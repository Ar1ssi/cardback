"""Where the Pokemon TCG Live extract is read from: a local folder, or a Cloudflare R2 bucket.

SVRESTYLE_TCGLIVE (or --tcglive) names either a folder or r2://<bucket>[/<prefix>]: a bucket whose keys mirror the
extract's folders (<prefix>/cardart/bundles/..., <prefix>/cardart/masks/..., <prefix>/cardart/manifests.jsonl,
<prefix>/WebApp/..., <prefix>/cosmetics/..., <prefix>/pokemon_Pokemon TCG Live/shadersbundle/...). A file read from
the bucket is downloaded once into <cache>/r2/<bucket>/<prefix>/ and read from there afterwards, so the code that
reads the extract only ever sees ordinary files: path() returns a Path that exists exactly when the extract has the
file. Delete <cache>/r2/ to pick up files changed in the bucket.

The bucket is reached through R2's S3 API (needs `pip install boto3`):
  SVRESTYLE_R2_ENDPOINT                 the account's S3 endpoint (default: ENDPOINT below)
  R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY an R2 API token's keys (Object Read is enough); unset, boto3's own
                                        chain is used (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, AWS_PROFILE, ...)
"""
import os
import threading
from pathlib import Path, PurePosixPath

from . import cache

SCHEME = "r2://"
ENDPOINT = "https://e24075f3d9dc9fbd325f42aa3f946d38.r2.cloudflarestorage.com"
MIRROR = "r2"                 # <cache>/r2/<bucket>/<prefix>/<key>

_client = None
_client_lock = threading.Lock()
_missing = set()              # (bucket, key) the bucket does not have: asked once per run


class ExtractError(RuntimeError):
    """The bucket could not be read (credentials, network, permissions): not the same as a file it lacks."""


def is_remote(root):
    return str(root or "").startswith(SCHEME)


def parse(root):
    """r2://bucket/some/prefix -> ('bucket', 'some/prefix'); the prefix may be ''."""
    bucket, _, prefix = str(root)[len(SCHEME):].strip("/").partition("/")
    if not bucket:
        raise ExtractError(f"{root}: no bucket name (expected r2://<bucket>[/<prefix>])")
    return bucket, prefix.strip("/")


def path(root, *parts):
    """The extract's file at parts: Path(root, *parts) for a folder; for a bucket, its local copy, downloaded on first
    use. Either way the returned Path exists only if the extract has the file."""
    if not is_remote(root):
        return Path(root, *parts)
    bucket, prefix = parse(root)
    rel = PurePosixPath(*parts)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"{rel}: not a path inside the extract")
    key = str(PurePosixPath(prefix, rel)) if prefix else str(rel)
    dest = cache.CACHE.joinpath(MIRROR, bucket, *PurePosixPath(key).parts)
    if not dest.exists() and (bucket, key) not in _missing:
        if not _download(bucket, key, dest):
            _missing.add((bucket, key))
    return dest


def _s3():
    global _client
    with _client_lock:
        if _client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as e:
                raise ExtractError("reading the TCG Live extract from R2 needs boto3: pip install boto3") from e
            keys = {}
            if os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
                keys = {"aws_access_key_id": os.environ["R2_ACCESS_KEY_ID"],
                        "aws_secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"]}
            _client = boto3.client("s3", endpoint_url=os.environ.get("SVRESTYLE_R2_ENDPOINT") or ENDPOINT,
                                   region_name="auto", config=Config(retries={"max_attempts": 5, "mode": "standard"}),
                                   **keys)
        return _client


def _download(bucket, key, dest):
    """Fetch bucket/key to dest (atomic write). False if the bucket has no such key."""
    from botocore.exceptions import BotoCoreError, ClientError
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}-{threading.get_ident()}.part")
    try:
        _s3().download_file(bucket, key, str(tmp))
    except ClientError as e:
        tmp.unlink(missing_ok=True)
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise ExtractError(f"r2://{bucket}/{key}: {e}") from e
    except BotoCoreError as e:
        tmp.unlink(missing_ok=True)
        raise ExtractError(f"r2://{bucket}/{key}: {e}") from e
    tmp.replace(dest)
    return True
