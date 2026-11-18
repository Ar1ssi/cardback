"""Download cache. Everything fetched or built lives under SVRESTYLE_CACHE (default ~/.cache/svrestyle), so card
scans, the LaMa model and the built frame assets are never committed."""
import functools
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CACHE = Path(os.environ.get("SVRESTYLE_CACHE", Path.home() / ".cache" / "svrestyle"))
USER_AGENT = "svrestyle/1.0"
# Scans, built assets and cleaned art depend on where the scans come from (tcg.TCGLIVE): keep the two apart.
BY_SOURCE = ("scans", "assets", "art")
SOURCE = "tcglive" if os.environ.get("SVRESTYLE_TCGLIVE") else ""


class DownloadError(RuntimeError):
    pass


# One lock for every built asset, not one a module: the builders call across modules (megaretype's into fairy's), and
# a single re-entrant lock leaves no lock order to get wrong. A built asset is read from lru_cache or its file after.
_build = threading.RLock()


def locked(build):
    """An asset builder that runs one call at a time with every other one: the GUI serves requests on threads, and two
    first calls would both write its file while the other reads it (they call one another: the lock is re-entrant)."""
    @functools.wraps(build)
    def run(*args, **kwargs):
        with _build:
            return build(*args, **kwargs)
    return run


def path(*parts):
    p = CACHE.joinpath(SOURCE, *parts) if SOURCE and parts[0] in BY_SOURCE else CACHE.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sha256(file):
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, timeout=60, retries=4):
    """Download url to dest once (atomic write); later calls reuse the file."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise DownloadError(f"{url}: not found (404)") from e
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    raise DownloadError(f"{url}: {last}")


def fetch_json(url, dest):
    return json.loads(Path(fetch(url, dest)).read_text(encoding="utf-8"))
