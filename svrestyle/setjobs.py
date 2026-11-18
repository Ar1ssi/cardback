"""Sending a set to the simulator's custom-sets folder and publishing it to the site's bucket, as background jobs a
page polls (the Jobs class, also used by merged.py), and the push-all plan: what pushing every saved project as one
set would change there, worked out without rendering.
"""
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

from . import cache, edit, export, projects, rendering, tcg
from .rendering import UserError

SIM_SETS = ("client", "src", "assets", "custom-sets")   # where the simulator keeps custom sets, in its checkout
PUBLISH_SCRIPT = ("scripts", "custom-sets", "publish.mjs")    # the simulator's bucket uploader, in its checkout
PUBLISH_TIMEOUT = 20 * 60             # s: past the script's own 15-minute upload limit, so it reports first
SET_JOB_KEEP = 8                      # finished sends and publishes kept for a page that asks late
CANCEL_POLL = 0.5                     # s between looks at a cancel while the publish script runs
SITE_CHECK_TIMEOUT = 20               # s the bucket's index may take to read before a publish is checked without it
CARD_FILE = re.compile(r"(\d{3})/high\.webp")    # a card of a published set, among the files of the bucket's index

_push_lock = threading.Lock()         # one write into the simulator's folder at a time


class PublishError(Exception):
    """The upload to the site's bucket failed (rclone, the network, the bucket), or timed out."""


class DropsCards(Exception):
    """A send or publish would take cards out of a set that is already there: `info` says which copies and how many
    (drops_for). The page asks the user, and asks the server again with allowShrink; a 409."""
    def __init__(self, info):
        super().__init__(info["message"])
        self.info = info


class Jobs:
    """Background jobs a page polls (sends and publishes; all-projects zips), the last keep() by id. A job is a dict
    with at least id, state ("running", then "done", "failed" or "cancelled") and cancel (asked to stop: its thread
    looks). view(job) is what the page reads; dropped(job) cleans up after a job that falls out; `missing` names the
    kind in the answer to an id it does not have."""
    def __init__(self, keep, view, missing, dropped=None):
        self.jobs = OrderedDict()
        self.lock = threading.RLock()     # a starter holds it from its look for a running job to add
        self.keep, self.view, self.missing, self.dropped = keep, view, missing, dropped

    def add(self, **fields):
        job = {"id": secrets.token_hex(8), "state": "running", "cancel": False, **fields}
        with self.lock:
            self.jobs[job["id"]] = job
            while len(self.jobs) > self.keep():
                old = self.jobs.popitem(last=False)[1]
                if self.dropped:
                    self.dropped(old)
        return job

    def running(self, cancelling=True):
        """The job running now, or None; with cancelling False, not one that was asked to stop."""
        with self.lock:
            return next((j for j in self.jobs.values()
                         if j["state"] == "running" and (cancelling or not j["cancel"])), None)

    def find(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def progress(self, job_id):
        job = self.find(job_id)
        if not job:
            raise LookupError(f"no such {self.missing} (the server restarted since?): start it again")
        return self.view(job)

    def cancel(self, job_id):
        """Ask a job to stop; one that is over is left as it is. -> its progress"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise LookupError(f"no such {self.missing} (the server restarted since?)")
            if job["state"] == "running":
                job["cancel"] = True
        return self.view(job)


def sim_sets_dir():
    """The simulator's custom-sets folder, from SVRESTYLE_PTCG_SIM: the simulator's checkout (its
    client/src/assets/custom-sets is used, made if missing) or that folder itself; None when unset or neither."""
    raw = os.environ.get("SVRESTYLE_PTCG_SIM", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if (path / "client" / "src").is_dir():
        return path.joinpath(*SIM_SETS)
    if path.name == SIM_SETS[-1] and path.is_dir():
        return path
    return None


def _sim_root():
    """The simulator's custom-sets folder; a UserError saying how to set it when it is not known."""
    root = sim_sets_dir()
    if root is None:
        raise UserError("no ptcg-sim folder: start the GUI with SVRESTYLE_PTCG_SIM set to the simulator's folder "
                        "(start-site.bat sets it when the simulator is in Downloads)")
    return root


def _write_set(items, title, code, root, force, progress=None, cancelled=None):
    """export.write_set_dir into `root`; the caller holds _push_lock."""
    try:
        return export.write_set_dir(items, title, code, root, read=rendering.read, render=rendering.render,
                                    apply=edit.apply, changelog=rendering.changelog, force=bool(force),
                                    progress=progress, cancelled=cancelled, is_ace=rendering._is_ace)
    except ValueError as e:
        raise UserError(str(e)) from e


def site_remote():
    """The rclone remote of the site's bucket (CUSTOM_SETS_R2_REMOTE, e.g. r2:ptcg-foil/custom-sets), or None."""
    return os.environ.get("CUSTOM_SETS_R2_REMOTE", "").strip() or None


def sim_checkout():
    """The simulator's checkout holding its publish script, or None (sim_sets_dir is <checkout>/client/src/...)."""
    root = sim_sets_dir()
    if root is None:
        return None
    checkout = root.parents[len(SIM_SETS) - 1]
    return checkout if checkout.joinpath(*PUBLISH_SCRIPT).is_file() else None


def _kill_tree(proc):
    """Stops the publish script and the rclone it started: killing node alone would leave rclone uploading."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)      # its own process group (_run_publish starts it in one)
        except ProcessLookupError:
            pass
    proc.communicate()


def _run_publish(node, checkout, set_id, remote, cancelled=None):
    """The publish script's --json report; PublishError on a failure, a timeout or no report. cancelled() -> True
    stops the script and the rclone under it (export.Cancelled): the bucket's index is replaced only when the upload
    is over, so the site keeps the copy it had."""
    proc = subprocess.Popen([node, str(checkout.joinpath(*PUBLISH_SCRIPT)), set_id, "--remote", remote, "--json"],
                            cwd=checkout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                            errors="replace", start_new_session=os.name != "nt")
    deadline = time.monotonic() + PUBLISH_TIMEOUT
    while True:
        try:
            out, err = proc.communicate(timeout=CANCEL_POLL)
            break
        except subprocess.TimeoutExpired:
            if cancelled and cancelled():
                _kill_tree(proc)
                raise export.Cancelled from None
            if time.monotonic() > deadline:
                _kill_tree(proc)
                raise PublishError(f"publishing gave no answer in {PUBLISH_TIMEOUT} s; stopped it") from None
    if proc.returncode != 0:
        lines = (err or out or "").strip().splitlines()
        raise PublishError("publishing failed: " + (" | ".join(lines[-5:]) or f"exit status {proc.returncode}"))
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as e:
        raise PublishError(f"the publish script gave no report: {out.strip()[-300:]!r}") from e


# ---------------------------------------------------------------- send and publish, in the background
def site_card_count(remote, set_id):
    """How many cards the site's copy of `set_id` has (the card folders among the files of the bucket's index.json),
    or None: it is not published, or the bucket could not be read in SITE_CHECK_TIMEOUT s (a publish then reports a
    bucket out of reach by itself)."""
    rclone = shutil.which("rclone")
    if rclone is None:
        return None
    try:
        done = subprocess.run([rclone, "cat", f"{remote}/index.json"], capture_output=True, encoding="utf-8",
                              errors="replace", timeout=SITE_CHECK_TIMEOUT)
        index = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    sets = index.get("sets") if isinstance(index, dict) else None
    entry = sets.get(set_id) if isinstance(sets, dict) else None
    files = entry.get("files") if isinstance(entry, dict) else None
    if not isinstance(files, dict):
        return None
    return len({m.group(1) for name in files if (m := CARD_FILE.fullmatch(name))})


def _list_of(names, shown=5):
    return ", ".join(names[:shown]) + (f" and {len(names) - shown} more" if len(names) > shown else "")


def drops_for(kind, items, code, set_id):
    """What sending (kind "push") or publishing (kind "publish") this list would take out of the copies of the set
    that are already there, or None when it takes nothing out. The simulator's copy is compared card by card (the
    cards it was made from, export.dropped_cards); the site's, for a publish, by size: the folders past the end of
    the list go. -> {setId, code, incoming, copies: [{where, cards, dropped}], names: [first few], message}"""
    ids = [it["id"] for it in items]
    existing = export.existing_cards(sim_sets_dir() / set_id)
    names = [name for _, name in export.dropped_cards(existing, ids)] if existing is not None else []
    site = site_card_count(site_remote(), set_id) if kind == "publish" else None
    site_drops = site - len(ids) if site is not None and site > len(ids) else 0
    if not names and not site_drops:
        return None
    copies, lines = [], []
    if names:
        copies.append({"where": "ptcg-sim", "cards": len(existing), "dropped": len(names)})
        lines.append(f"{code} has {len(existing)} cards in ptcg-sim; this list has {len(ids)}: "
                     f"{len(names)} would be removed ({_list_of(names)}).")
    if site_drops:
        copies.append({"where": "the site", "cards": site, "dropped": site_drops})
        lines.append(f"The site has {site} cards of {code}" + ("" if names else f"; this list has {len(ids)}")
                     + f": {site_drops} would be removed" + (" there too." if names else "."))
    return {"setId": set_id, "code": code, "incoming": len(ids), "copies": copies, "names": names[:5],
            "message": " ".join(lines)}


def _set_job_view(job):
    end = job.get("ended") or time.time()
    return {"id": job["id"], "kind": job["kind"], "code": job["code"], "setId": job["setId"], "state": job["state"],
            "phase": job["phase"], "done": job["done"], "total": job["total"], "current": job["current"],
            "elapsed": int(end - job["started"]), "cancelling": job["cancel"], "result": job["result"],
            "error": job["error"], "pushed": job["pushed"]}


SET_JOBS = Jobs(lambda: SET_JOB_KEEP, _set_job_view, "send or publish")    # sends and publishes in the background
_set_jobs = SET_JOBS.jobs


def _running_set_job():
    return SET_JOBS.running()


def start_set_job(kind, items, title, code, force=False, allow_shrink=False):
    """Send (kind "push": export.write_set_dir into the simulator's folder, which the running simulator reloads) or
    publish (kind "publish": that, then the simulator's publish script uploads the set to the site's bucket) in the
    background: a first render takes 10-20 s a card, an upload minutes, and tunnels cut a request at ~100 s, so the
    page polls set_job_progress. Only cards changed since the last send are rendered (all of them with force). A
    set that would lose cards is refused (DropsCards) unless allow_shrink. A job already running for the same set and
    kind is joined, not started again; another set's is a UserError: one write into the folder at a time. -> its
    progress"""
    root = _sim_root() if kind == "push" else sim_sets_dir()
    remote = checkout = node = None
    if kind == "publish":
        remote = site_remote()
        if remote is None:
            raise UserError("set CUSTOM_SETS_R2_REMOTE to the site's bucket (e.g. r2:ptcg-foil/custom-sets) to publish")
        checkout = sim_checkout()
        if checkout is None:
            raise UserError("no ptcg-sim checkout with scripts/custom-sets/publish.mjs: set SVRESTYLE_PTCG_SIM to it")
        node = shutil.which("node")
        if node is None:
            raise UserError("node not found on PATH: the publish script runs on Node.js")
    if not isinstance(items, list) or not items:
        raise UserError("items: a non-empty list")
    if len(items) > export.MAX_CARDS:
        raise UserError(f"items: at most {export.MAX_CARDS} cards in a set")
    if not all(isinstance(it, dict) and isinstance(it.get("id"), str) for it in items):
        raise UserError("items: expected objects with an id")
    try:
        set_id = export.set_id_for(code)
    except ValueError as e:
        raise UserError(str(e)) from e
    with SET_JOBS.lock:
        running = _running_set_job()
        if running:
            return _joined_or_refused(running, kind, set_id)
    # The list's order is only how it looks here: set.json lists the cards in display order (export.display_order).
    items = push_items(items, sim_sets_dir() / set_id)
    if len(items) > export.MAX_CARDS:
        raise UserError(f"items: at most {export.MAX_CARDS} cards in a set")
    if not allow_shrink:
        info = drops_for(kind, items, code, set_id)
        if info:
            raise DropsCards(info)
    with SET_JOBS.lock:
        running = _running_set_job()
        if running:                                  # another page started one while the check ran
            return _joined_or_refused(running, kind, set_id)
        job = SET_JOBS.add(kind=kind, code=code, setId=set_id, phase="rendering", done=0, total=len(items), current="",
                           started=time.time(), ended=None, result=None, error=None, pushed=None)
        threading.Thread(target=_run_set_job, args=(job, items, title, code, force, (root, checkout, node, remote)),
                         daemon=True).start()
        return _set_job_view(job)


def _joined_or_refused(running, kind, set_id):
    if (running["kind"], running["setId"]) == (kind, set_id):
        return _set_job_view(running)
    raise UserError(f"a {'publish' if running['kind'] == 'publish' else 'send'} of {running['code']} is still "
                    "running: wait for it, or cancel it")


def _run_set_job(job, items, title, code, force, plan):
    """The job's thread: the push, then for a publish the upload, with the push lock held through both so no other
    write changes the files while they go up. The job always ends in state done, failed or cancelled."""
    root, checkout, node, remote = plan

    def cancelled():
        return job["cancel"]

    def progress(phase, done, total, current):
        job.update(phase=phase, done=done, total=total, current=current)
    try:
        with _push_lock:
            pushed = _write_set(items, title, code, root, force, progress, cancelled)
            job["pushed"] = pushed
            if job["kind"] == "publish":
                if cancelled():
                    raise export.Cancelled
                job.update(phase="uploading", done=0, total=0, current="")
                report = _run_publish(node, checkout, pushed["setId"], remote, cancelled)
                job["result"] = {**pushed, "published": report}
            else:
                job["result"] = pushed
        job["state"] = "done"
    except export.Cancelled:
        job["state"] = "cancelled"
    except PublishError as e:
        job["error"], job["state"] = f"sent to ptcg-sim, but {e}", "failed"
    except UserError as e:
        job["error"], job["state"] = str(e), "failed"
    except (Exception, SystemExit) as e:                                    # noqa: BLE001 - the page shows it
        job["error"], job["state"] = f"{type(e).__name__}: {e}", "failed"
    finally:
        job["ended"] = time.time()


def set_job_progress(job_id):
    return SET_JOBS.progress(job_id)


def cancel_set_job(job_id):
    """Stop a send or publish. Before the first file is written nothing changes; a publish stops its upload (the site
    keeps the copy it had); a send already writing finishes. Cancelling a job that is over changes nothing. -> its
    progress"""
    return SET_JOBS.cancel(job_id)


# ---------------------------------------------------------------- push all: what it would change
def keep_unlisted(items, set_dir):
    """`items` with the cards the set already in set_dir holds that the list does not name put back where they are in
    the set (export.unlisted_items): a push adds and changes cards, it never takes one out. A card made by hand in
    the simulator's folder, or sent from another list, stays with its number and its files."""
    out = list(items)
    for place, item in export.unlisted_items(set_dir, {it["id"] for it in items}):
        out.insert(min(place, len(out)), item)
    return out


MULTI_PRIZE = (" ex", " EX", " GX", " V", " VMAX", " VSTAR")   # name endings of Pokemon worth more than 1 prize


def takes_reverse(card_id):
    """Whether a card always gets the Gen 8 reverse holo (foil.gen8_plate): a 1-prize Pokemon, a Trainer but an ACE
    SPEC card (its own foil). Energy and multi-prize Pokemon keep the holofoil switch's choice."""
    try:
        layout = rendering.layout_of(card_id)
        name = tcg.card(card_id).get("name") or ""
    except (UserError, ValueError, cache.DownloadError):
        return False
    if layout == "trainer":
        return rendering._is_ace(card_id) is False  # an ACE SPEC card TCGdex cannot vouch for keeps the switch's choice
    return layout in ("me", "modern") and not name.endswith(MULTI_PRIZE)


def standard_foil(items):
    """`items` with holofoil on for every card that takes the Gen 8 reverse holo as standard (takes_reverse): a
    push sets it whatever the holofoil switch says. A card the old print made holo keeps its holo."""
    return [it if it.get("foil") or not takes_reverse(it["id"]) else {**it, "foil": True} for it in items]


def keep_numbers(items, existing):
    """`items` in the order of the set already there (export.existing_cards): a card that is in it keeps its number,
    the others follow in the order they came. The merge of the projects lists the newest project first, so its own
    order would renumber the set whenever another project is saved."""
    place = {}
    for i, (source, _) in enumerate(existing or []):
        if source:
            place.setdefault(source, i)
    return (sorted((it for it in items if it["id"] in place), key=lambda it: place[it["id"]])
            + [it for it in items if it["id"] not in place])


def push_items(items, set_dir):
    """What a push of `items` into set_dir sends: a card in the set already keeps its number whatever its place in the
    list (keep_numbers), the cards of the set the list does not name stay (keep_unlisted: a push never takes one out),
    and every 1-prize Pokemon and Trainer goes with its holofoil (standard_foil)."""
    return standard_foil(keep_unlisted(keep_numbers(items, export.existing_cards(set_dir)), set_dir))


def _push_changes(original, after, layout, old, new, stale=False):
    """What differs between a card as it was pushed (its options `old`) and as the list has it now (`new`, from
    export.card_options): the edits' changes, then the options'. stale: True when its images on disk are not the ones
    the set's record names, "foil" when its holofoil there is not known to match it (made outside a push, or its files
    differ from the record): rendered again whatever changed."""
    try:
        before = edit.apply(original, old["edits"]) if old.get("edits") else original
    except ValueError:                                   # edits this version no longer takes
        before = original
    lines = edit.changes(before, after)
    if layout == "ex" and (old.get("style") or "hybrid") != new["style"]:
        lines.append(f"Style: {old.get('style') or 'hybrid'} → {new['style']}")
    if bool(old.get("attacks", True)) != new["attacks"]:
        lines.append("Abilities and attacks: " + ("shown" if new["attacks"] else "left off"))
    if bool(old.get("foil")) != new["foil"]:
        lines.append("Holofoil: " + ("on" if new["foil"] else "off"))
    if stale == "foil":
        lines.append("Its holofoil in the set is not known to match the card: made again")
    elif stale:
        lines.append("Its images in the set are not the ones its record names: rendered again")
    return lines or ["Set again (its options changed, the card reads the same)"]


def _plan_card(item, old=None, stale=False):
    """A card of the push plan: {id, name (as the set will name it), lines}. lines: for a new card what differs from
    the printed card (rendering.card_changes); for a changed one (old: the options it was pushed with) what differs
    from that push. A card that cannot be read is listed with the reason: the push would fail on it."""
    try:
        layout, original = rendering.read(item["id"])
        card = edit.apply(original, item["edits"]) if item.get("edits") else original
        opts = export.card_options(item)
        lines = (rendering.card_changes(item["id"], item.get("edits"), opts["style"], opts["attacks"])[1] if old is None
                 else _push_changes(original, card, layout, old, opts, stale))
        return {"id": item["id"], "name": export.card_name(card, layout), "lines": lines}
    except (UserError, ValueError) as e:
        return {"id": item["id"], "name": item["id"], "lines": [f"cannot be read: {e}"]}


def push_all_plan(code):
    """What pushing every saved project, merged (projects.merged_items; the cards already in the set keep their
    numbers), as the set `code` would change in the simulator's folder, worked out from its set.json without rendering
    anything.
    -> {setId, code, title (the set's name now), projects, repeats, incoming, existing (cards there, None: no set yet),
    added / changed: [{id, name, lines}], removed: [{id, name}] (cards no list can send again: none in a set this editor
    wrote), kept: [{id, name}] (in the set, in no saved project: they stay), moved (cards that get another number),
    unchanged, unreadable: [{name, why}] (projects that could not be read: their cards are in none of the lists),
    notes: [str] (what the merge left off, projects.merged_items),
    items (what to send to /api/push-set or /api/publish-set to push exactly this)}"""
    root = _sim_root()
    try:
        set_id = export.set_id_for(code)
    except ValueError as e:
        raise UserError(str(e)) from e
    merged = projects.merged_items()
    if not merged.items:
        raise UserError("no saved project has a card to push"
                        + (f" ({projects._unreadable_note(merged.unreadable)})" if merged.unreadable else ""))
    set_dir = root / set_id
    listed = {it["id"] for it in merged.items}
    items = push_items(merged.items, set_dir)
    plan = export.plan_set(items, set_dir)
    names = {source: name for source, name in reversed(export.existing_cards(set_dir) or []) if source}
    kept = [{"id": it["id"], "name": names.get(it["id"], it["id"])} for it in items if it["id"] not in listed]
    return {"setId": set_id, "code": code, "title": export.existing_title(set_dir), "projects": merged.projects,
            "repeats": merged.repeats, "incoming": len(items), "existing": plan["existing"],
            "added": [_plan_card(it) for it in plan["added"]],
            "changed": [_plan_card(it, old, stale) for it, old, stale in plan["changed"]],
            "removed": [{"id": source, "name": name} for source, name in plan["removed"]],
            "kept": kept, "moved": plan["moved"], "unchanged": plan["unchanged"], "unreadable": merged.unreadable,
            "notes": merged.notes, "items": items}
