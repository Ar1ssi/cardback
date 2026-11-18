"""Saved projects: the page's card list, edits and options as JSON files, the patches pages send as they work, each
project's history and the trash, every project merged into one list, and the long polls that tell open pages what
happened to their project.

Projects are JSON files under the cache folder (SVRESTYLE_CACHE/projects), so work resumes after a restart. Each
save or patch raises the project's rev; a page that missed some (reconnecting) reads the project again. Before a
project is written over, its file is copied to projects/.history/<project>/: always before a whole save or a
restore, and while patches come in once every HISTORY_EVERY s and once more when work starts again after a break
(the newest HISTORY_KEEP stay). Delete moves the file to projects/.trash/ (the newest TRASH_KEEP stay).
"""
import datetime
import itertools
import json
import os
import re
import shutil
import threading
import time
from collections import deque, namedtuple

from . import cache, edit, rendering, tcg
from .rendering import UserError

PROJECT_NAME = re.compile(r"\w[\w .,'()&+#-]{0,59}")
MAX_PROJECT = 4 << 20                 # bytes of one saved project
POLL_WAIT = 20                        # s a poll waits for news before answering with none (tunnels cut at ~100 s)
VIEWER_GONE = POLL_WAIT + 25          # s without a poll after which a page no longer counts as having it open
EVENT_LOG = 200                       # events each project keeps for pages catching up
HISTORY_EVERY = 600                   # s between versions kept while patches come in; also the break that keeps one
HISTORY_KEEP = 100                    # versions kept of each project
TRASH_KEEP = 100                      # deleted projects kept
VERSION_ID = re.compile(r"\d{8}-\d{6}-r\d{6,9}")
TRASH_ID = re.compile(r"\d{8}-\d{6}-[\w-]{1,80}")


# ---------------------------------------------------------------- projects
_project_lock = threading.Lock()


def projects_dir():
    return cache.CACHE / "projects"


def project_file(name):
    if not isinstance(name, str) or not PROJECT_NAME.fullmatch(name.strip()):
        raise UserError("project name: 1-60 letters, digits, spaces or . , ' ( ) & + # -")
    slug = re.sub(r"[^\w-]+", "-", name.strip().lower()).strip("-")
    return projects_dir() / f"{slug}.json"


def _cards(p):
    return len((p.get("data") or {}).get("cards") or [])


def _read_projects():
    """-> ([{name, saved, cards}] newest first, [{name, why}]: the project files that cannot be read)"""
    out, unreadable = [], []
    for f in projects_dir().glob("*.json") if projects_dir().exists() else ():
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            unreadable.append({"name": f.stem, "why": f"{type(e).__name__}: {e}"})
            continue
        out.append({"name": p.get("name", f.stem), "saved": p.get("saved", ""), "cards": _cards(p)})
    return sorted(out, key=lambda p: p["saved"], reverse=True), unreadable


def list_projects():
    return {"projects": _read_projects()[0]}


def load_project(name):
    f = project_file(name)
    if not f.exists():
        raise UserError(f"no saved project {name.strip()!r}")
    p = json.loads(f.read_text(encoding="utf-8"))
    p.setdefault("rev", 0)
    if isinstance(p.get("data"), dict) and isinstance(p["data"].get("cards"), list):
        p["data"]["cards"] = rendering._current_entries(p["data"]["cards"])
    return p


def _check_data(data):
    if not isinstance(data, dict) or not isinstance(data.get("cards", []), list) \
            or not isinstance(data.get("edits", {}), dict) or not isinstance(data.get("options", {}), dict):
        raise UserError("project data: an object with a \"cards\" list and \"edits\" and \"options\" objects")


def _rev(f):
    try:
        return int(json.loads(f.read_text(encoding="utf-8")).get("rev", 0))
    except (OSError, ValueError, AttributeError):
        return 0


def _write(f, name, data, rev):
    """Write the project atomically (a half-written file would lose the work it was saving). Under _project_lock."""
    saved = datetime.datetime.now().isoformat(timespec="seconds")
    text = json.dumps({"name": name, "saved": saved, "rev": rev, "data": data}, ensure_ascii=False)
    if len(text.encode()) > MAX_PROJECT:
        raise UserError(f"project too large (over {MAX_PROJECT >> 20} MB)")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(f".tmp{threading.get_ident()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, f)
    return {"name": name, "saved": saved, "rev": rev}


def save_project(name, data, client=None):
    _check_data(data)
    f = project_file(name)
    with _project_lock:
        _keep_version(f, always=True)
        out = _write(f, name.strip(), data, _rev(f) + 1 if f.exists() else 1)
        HUB.publish(f.stem, "replaced", {"rev": out["rev"], "client": client})
    return out


def apply_patch(data, patch):
    """The project data with a page's changes made. patch: {cards?: {removed: [id], added: [{index, card}],
    order?: [id]}, edits?: {id: {field: value} | null}, options?: {id: object | null}, setName?, foil?, current?}.
    code: the set code for the ptcg-sim export (export.py).
    A card's edits change field by field, so two people editing different fields of one card both keep theirs; null
    drops a card's edits or options. order is the list as the page has it: the cards it names take the places those
    cards hold, in its order, and cards it does not name (someone else's, just added) keep theirs. The page's
    applyPatch does the same."""
    if not isinstance(patch, dict):
        raise UserError("patch: an object")
    out = json.loads(json.dumps(data))
    out.setdefault("cards", [])
    cards = patch.get("cards")
    if cards is not None:
        if not isinstance(cards, dict) or not isinstance(cards.get("removed", []), list) \
                or not isinstance(cards.get("added", []), list) or not isinstance(cards.get("order", []), list):
            raise UserError("patch cards: {removed: [id], added: [{index, card}], order: [id]}")
        gone = set(cards.get("removed", []))
        out["cards"] = [c for c in out["cards"] if c.get("id") not in gone]
        for add in cards.get("added", []):
            card = add.get("card") if isinstance(add, dict) else None
            if not isinstance(card, dict) or not isinstance(card.get("id"), str):
                raise UserError("patch cards added: {index, card: {id, ...}}")
            if any(c.get("id") == card["id"] for c in out["cards"]):
                continue
            index = add.get("index") if isinstance(add.get("index"), int) else len(out["cards"])
            out["cards"].insert(max(0, min(index, len(out["cards"]))), card)
        out["cards"] = reorder(out["cards"], cards.get("order", []))
    for key in ("edits", "options"):
        changes = patch.get(key)
        if changes is None:
            continue
        if not isinstance(changes, dict):
            raise UserError(f"patch {key}: an object by card id")
        held = out.setdefault(key, {})
        for card_id, value in changes.items():
            if value is None:
                held.pop(card_id, None)
            elif not isinstance(value, dict):
                raise UserError(f"patch {key} {card_id}: an object or null")
            elif key == "edits":
                held[card_id] = {**(held.get(card_id) or {}), **value}
            else:
                held[card_id] = value
    for key in ("setName", "foil", "current", "code"):
        if key in patch:
            out[key] = patch[key]
    return out


def reorder(cards, order):
    rank = {card_id: i for i, card_id in enumerate(order) if isinstance(card_id, str)}
    named = iter(sorted((c for c in cards if c.get("id") in rank), key=lambda c: rank[c["id"]]))
    return [next(named) if c.get("id") in rank else c for c in cards]


def patch_project(name, patch, client=None):
    f = project_file(name)
    with _project_lock:
        held = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"data": {"version": 1}}
        data = apply_patch(held.get("data") or {}, patch)
        _check_data(data)
        _keep_version(f)
        out = _write(f, name.strip(), data, int(held.get("rev", 0)) + 1)
        HUB.publish(f.stem, "patch", {"rev": out["rev"], "client": client, "patch": patch})
    return out


def delete_project(name, renamed=None, client=None):
    """renamed: the name the work was saved under instead, so the others with it open move there, and its
    history with them. A plain delete moves the project to the trash."""
    f = project_file(name)
    to = project_file(renamed) if renamed is not None else None      # checks the name
    with _project_lock:
        if not f.exists():
            raise UserError(f"no saved project {name.strip()!r}")
        if to is not None:
            _move_history(f, to)
            f.unlink()
            HUB.publish(f.stem, "renamed", {"to": renamed.strip(), "client": client})
        else:
            _to_trash(f)
            HUB.publish(f.stem, "deleted", {"client": client})
    return {"deleted": name.strip()}


# ---------------------------------------------------------------- project history and trash
def _history_dir(f):
    return projects_dir() / ".history" / f.stem


def _trash_dir():
    return projects_dir() / ".trash"


def _stamp(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y%m%d-%H%M%S")


def _time_of(file_id):
    """The time a version's or a trashed project's id starts with (_stamp), or None."""
    try:
        return datetime.datetime.strptime(file_id[:15], "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def _prune(folder, keep):
    for old in sorted(folder.glob("*.json"))[:-keep]:
        old.unlink(missing_ok=True)


def _saved(f):
    """(time, rev) the project file was saved with; its file time when it does not say (a copy has the copy's)."""
    try:
        p = json.loads(f.read_text(encoding="utf-8"))
        return datetime.datetime.fromisoformat(p["saved"]).timestamp(), int(p.get("rev", 0))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return f.stat().st_mtime, _rev(f)


def _keep_version(f, always=False):
    """Copy the project file into its history before it is written over. Pages patch 0.3 s after each change, so
    not each time: when `always` (a whole save, a restore), when the newest version is HISTORY_EVERY older than the
    project, or when the project has not changed for HISTORY_EVERY (the work a break ended on). Versions are named
    by the project's saved time and rev, so their names sort oldest first. Under _project_lock."""
    if not f.exists():
        return
    folder = _history_dir(f)
    versions = sorted(folder.glob("*.json")) if folder.exists() else []
    changed, rev = _saved(f)
    newest = _time_of(versions[-1].stem) if versions else None
    if not (always or newest is None or changed - newest.timestamp() >= HISTORY_EVERY
            or time.time() - changed >= HISTORY_EVERY):
        return
    to = folder / f"{_stamp(changed)}-r{rev:06d}.json"
    if to.exists():                              # kept already (a restore right after another)
        return
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(f, to)
    _prune(folder, HISTORY_KEEP)


def _move_history(f, to):
    """A renamed project's versions join those of its new name."""
    src, dst = _history_dir(f), _history_dir(to)
    if src == dst or not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for v in src.glob("*.json"):
        if not (dst / v.name).exists():
            os.replace(v, dst / v.name)
    shutil.rmtree(src, ignore_errors=True)
    _prune(dst, HISTORY_KEEP)


def _to_trash(f):
    folder = _trash_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stamp = _stamp(time.time())
    to = folder / f"{stamp}-{f.stem}.json"
    for n in itertools.count(2):
        if not to.exists():
            break
        to = folder / f"{stamp}-{f.stem}-{n}.json"
    os.replace(f, to)
    _prune(folder, TRASH_KEEP)


def list_versions(name):
    f = project_file(name)
    folder = _history_dir(f)
    out = []
    for v in sorted(folder.glob("*.json"), reverse=True) if folder.exists() else ():
        try:
            p = json.loads(v.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": v.stem, "saved": p.get("saved", ""), "rev": p.get("rev", 0), "cards": _cards(p)})
    return {"name": name.strip(), "versions": out}


def restore_version(name, version, client=None):
    f = project_file(name)
    if not isinstance(version, str) or not VERSION_ID.fullmatch(version):
        raise UserError("version: an id from the project's history")
    v = _history_dir(f) / f"{version}.json"
    with _project_lock:
        if not v.exists():
            raise UserError(f"no version {version} of {name.strip()!r}")
        data = json.loads(v.read_text(encoding="utf-8")).get("data")
        _check_data(data)
        _keep_version(f, always=True)
        out = _write(f, name.strip(), data, _rev(f) + 1 if f.exists() else 1)
        HUB.publish(f.stem, "replaced", {"rev": out["rev"], "client": client})
    return out


def list_trash():
    out = []
    for t in sorted(_trash_dir().glob("*.json"), reverse=True) if _trash_dir().exists() else ():
        deleted = _time_of(t.stem)
        try:
            p = json.loads(t.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": t.stem, "name": p.get("name", t.stem[16:]), "saved": p.get("saved", ""),
                    "deleted": deleted.isoformat() if deleted else "", "cards": _cards(p)})
    return {"trash": out}


def _free_name(name):
    """name, or "name (restored)", "name (restored 2)"... when a project has it now."""
    if not project_file(name).exists():
        return name
    for n in itertools.count(1):
        tail = " (restored)" if n == 1 else f" (restored {n})"
        free = name[:60 - len(tail)].rstrip() + tail
        if not project_file(free).exists():
            return free


def undelete_project(trash_id):
    if not isinstance(trash_id, str) or not TRASH_ID.fullmatch(trash_id):
        raise UserError("id: a deleted project's id")
    t = _trash_dir() / f"{trash_id}.json"
    with _project_lock:
        if not t.exists():
            raise UserError("no such deleted project (restored already, or emptied from the trash)")
        held = json.loads(t.read_text(encoding="utf-8"))
        _check_data(held.get("data"))
        name = _free_name(str(held.get("name") or "").strip())
        out = _write(project_file(name), name, held["data"], int(held.get("rev", 0)) + 1)
        t.unlink()
    return out


# ---------------------------------------------------------------- every project as one list
Merged = namedtuple("Merged", "items projects repeats unreadable notes")


def merged_items():
    """Every saved project's cards as one list of zip items (like the page's own, see gui.zip_renders). A card in
    several projects comes from the most recently saved one, with that project's edits and options; cards a project
    lists as skipped stay out. -> Merged: items, projects read, repeats left out, unreadable: [{name, why}] the
    projects that could not be read (their cards are not in the list: whoever uses it says so), notes: [str] what the
    list leaves out of what a project asks for (its holofoil, when there is no TCG Live extract to make it)."""
    items, seen, projects, repeats, notes = [], set(), 0, 0, []
    listed_projects, unreadable = _read_projects()
    for listed in listed_projects:                                    # newest saved first
        try:
            data = load_project(listed["name"]).get("data") or {}
        except (UserError, OSError, ValueError, AttributeError) as e:
            unreadable.append({"name": listed["name"], "why": str(e) or type(e).__name__})
            continue
        projects += 1
        foil = bool(data.get("foil"))
        if foil and not tcg.TCGLIVE:       # rendering.render() refuses it: the cards go without, and the note says so
            foil = False
            notes.append(f"{listed['name']}: holofoil left off (it needs a Pokemon TCG Live extract: start the GUI "
                         "with --tcglive DIR)")
        for c in data.get("cards") or []:
            if not isinstance(c, dict) or c.get("skip") or not edit.valid_card_id(c.get("id")):
                continue
            if c["id"] in seen:
                repeats += 1
                continue
            seen.add(c["id"])
            opt = (data.get("options") or {}).get(c["id"]) or {}
            items.append({"id": c["id"], "edits": (data.get("edits") or {}).get(c["id"]),
                          "style": opt.get("style", "hybrid"), "attacks": opt.get("attacks", True),
                          "foil": foil})
    return Merged(items, projects, repeats, unreadable, notes)


def _unreadable_note(unreadable):
    return f"{len(unreadable)} project{'' if len(unreadable) == 1 else 's'} could not be read: " + "; ".join(
        f"{u['name']}: {u['why']}" for u in unreadable[:3]) + (" …" if len(unreadable) > 3 else "")


# ---------------------------------------------------------------- who has a project open
class Room:
    def __init__(self):
        self.seq = 0
        self.log = deque(maxlen=EVENT_LOG)    # (seq, event, data)
        self.viewers = {}                     # client -> time of its last poll


class Hub:
    """What happens to each open project (by project file stem), for the pages polling it."""

    def __init__(self):
        self._cond = threading.Condition()
        self._rooms = {}

    def _room(self, key):
        return self._rooms.setdefault(key, Room())

    def _append(self, room, event, data):
        room.seq += 1
        room.log.append((room.seq, event, data))
        self._cond.notify_all()

    def _viewers(self, room, client=None):
        """Count the pages polling, `client` now one of them; a change in the count is news too."""
        now = time.monotonic()
        before = len(room.viewers)
        for c, seen in list(room.viewers.items()):
            if now - seen > VIEWER_GONE:
                del room.viewers[c]
        if client is not None:
            room.viewers[client] = now
        if len(room.viewers) != before:
            self._append(room, "viewers", {"viewers": len(room.viewers)})
        return len(room.viewers)

    def publish(self, key, event, data):
        with self._cond:
            self._append(self._room(key), event, data)

    def leave(self, key, client):
        with self._cond:
            room = self._room(key)
            if room.viewers.pop(client, None) is not None:
                self._append(room, "viewers", {"viewers": len(room.viewers)})

    def poll(self, key, after, client, rev, wait=POLL_WAIT):
        """-> {seq, viewers, events} once there are events past `after` or `wait` s pass; with hello {rev: rev()}
        at once when `after` is not a point this log can go on from (a new page, a restarted server, a page that
        was away longer than the log is long)."""
        deadline = time.monotonic() + wait
        with self._cond:
            room = self._room(key)
            self._viewers(room, client)
            oldest = room.log[0][0] if room.log else room.seq + 1
            if after < 0 or after > room.seq or after < oldest - 1:
                return {"seq": room.seq, "viewers": len(room.viewers), "events": [], "hello": {"rev": rev()}}
            while True:
                news = [{"event": e, "data": d} for n, e, d in room.log if n > after]
                left = deadline - time.monotonic()
                if news or left <= 0:
                    break
                self._cond.wait(left)
            room.viewers[client] = time.monotonic()
            return {"seq": room.seq, "viewers": len(room.viewers), "events": news}


HUB = Hub()


def poll_project(name, after, client):
    if not isinstance(client, str) or not 1 <= len(client) <= 80:
        raise UserError("poll: a client id")
    try:
        after = int(after)
    except (TypeError, ValueError):
        raise UserError("poll: after=SEQ") from None
    f = project_file(name)
    return HUB.poll(f.stem, after, client, lambda: _rev(f) if f.exists() else 0)


def leave_project(name, client):
    HUB.leave(project_file(name).stem, client)
    return {"left": name.strip()}
