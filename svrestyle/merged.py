"""Every saved project as one download: the merged list (projects.merged_items) rendered into one zip in the
background, which the page polls and then downloads; the last few kept under the cache folder (exports/).
"""
import threading
import zipfile

from . import cache, export, projects, rendering
from .rendering import UserError
from .setjobs import Jobs

MERGED_KEEP = 3                       # finished all-projects downloads kept on disk (exports/)


# ---------------------------------------------------------------- every project as one download
def exports_dir():
    return cache.CACHE / "exports"


def _export_view(job):
    return {**{k: job[k] for k in ("id", "state", "done", "total", "current", "projects", "repeats")},
            "failed": list(job["failed"]), "unreadable": list(job["unreadable"]), "notes": list(job["notes"]),
            "error": job.get("error"), "cancelling": job["cancel"]}


def _drop_export(job):
    if job["file"]:
        job["file"].unlink(missing_ok=True)


MERGED = Jobs(lambda: MERGED_KEEP, _export_view, "download", _drop_export)     # all-projects zips in the background
_exports = MERGED.jobs


def start_merged_zip():
    """Build every saved project, merged (projects.merged_items), as one zip in the background: a card takes 10-20 s to
    render the first time and tunnels cut a request at ~100 s, so the page polls merged_progress. A build already
    running is joined, not started again. -> its progress"""
    with MERGED.lock:
        running = MERGED.running(cancelling=False)
        if running:
            return _export_view(running)
        merged = projects.merged_items()
        if not merged.items:
            raise UserError("no saved project has a card to download"
                            + (f" ({projects._unreadable_note(merged.unreadable)})" if merged.unreadable else ""))
        job = MERGED.add(done=0, total=len(merged.items), current="", projects=merged.projects,
                         repeats=merged.repeats, failed=[], unreadable=merged.unreadable, notes=merged.notes,
                         file=None)
        for f in exports_dir().glob("*.zip") if exports_dir().exists() else ():       # left by an earlier run
            if f.stem not in MERGED.jobs:
                f.unlink(missing_ok=True)
        threading.Thread(target=_build_merged, args=(job, merged.items), daemon=True).start()
        return _export_view(job)


def _not_included(job):
    """NOT-INCLUDED.txt of an all-projects zip: the cards that could not be rendered, the projects that could not be
    read, and what was left off a card (projects.merged_items' notes); "" when nothing was."""
    parts = []
    if job["failed"]:
        parts.append("These cards could not be rendered:\n\n" + "\n".join(f"{f['id']}: {f['why']}" for f in job["failed"]))
    if job["unreadable"]:
        parts.append("These projects could not be read, so their cards are not here:\n\n"
                     + "\n".join(f"{u['name']}: {u['why']}" for u in job["unreadable"]))
    if job["notes"]:
        parts.append("Left off:\n\n" + "\n".join(job["notes"]))
    return "\n\n".join(parts) + "\n" if parts else ""


def _build_merged(job, items):
    """Render each card and zip it (renders are cached, so a card done before is quick). A card that fails does
    not stop the others: it is listed in NOT-INCLUDED.txt and in the progress."""
    path = exports_dir() / f"{job['id']}.zip"
    made = []
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for it in items:
                if job["cancel"]:
                    raise export.Cancelled
                job["current"] = it["id"]
                try:
                    png, _, layout, holo = rendering.render(it["id"], it["edits"], it["style"], it["attacks"],
                                                            it["foil"])
                except (Exception, SystemExit) as e:             # noqa: BLE001 - keep the cards that do render
                    job["failed"].append({"id": it["id"], "why": str(e) or type(e).__name__})
                else:
                    z.writestr(f"{it['id']}-{layout}.png", png)
                    for name, data in (holo.files.items() if holo else ()):
                        z.writestr(name, data)
                    made.append(it)
                job["done"] += 1
            if not made:
                raise UserError("no card could be rendered: " + "; ".join(f"{f['id']}: {f['why']}" for f in job["failed"][:3]))
            z.writestr("CHANGELOG.md", rendering.changelog(made, "All projects"))
            if _not_included(job):
                z.writestr("NOT-INCLUDED.txt", _not_included(job))
        job["file"], job["state"] = path, "done"
    except export.Cancelled:
        job["state"] = "cancelled"
        path.unlink(missing_ok=True)
    except Exception as e:                                       # noqa: BLE001 - the page shows it
        job["error"], job["state"] = (str(e) if isinstance(e, UserError) else f"{type(e).__name__}: {e}"), "failed"
        path.unlink(missing_ok=True)


def merged_progress(job_id):
    return MERGED.progress(job_id)


def cancel_merged_zip(job_id):
    """Stop a build: it ends after the card being rendered (a render cannot be stopped halfway) and its zip is
    removed. Cancelling a build that is over changes nothing. -> its progress"""
    return MERGED.cancel(job_id)


def merged_file(job_id):
    job = MERGED.find(job_id)
    if not job or job["state"] != "done" or not job["file"].exists():
        raise LookupError("that download is not ready, or was removed: start it again")
    return job["file"]
