"""The TCG Live extract read from a Cloudflare R2 bucket (extract.py), against a fake S3 client: no network."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from svrestyle import extract, tcg

try:
    from botocore.exceptions import ClientError
except ImportError:                       # boto3 is optional
    ClientError = None


class FakeS3:
    """download_file from a dict of key -> bytes; a missing key raises S3's 404, a denied one its 403."""

    def __init__(self, objects, denied=()):
        self.objects, self.denied, self.calls = objects, set(denied), []

    def download_file(self, bucket, key, filename):
        self.calls.append((bucket, key))
        if key in self.denied:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject")
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        Path(filename).write_bytes(self.objects[key])


class ParseTest(unittest.TestCase):
    def test_remote_roots(self):
        self.assertTrue(extract.is_remote("r2://ptcg-foil/tcglive"))
        self.assertFalse(extract.is_remote("E:/TCGLive_Extract"))
        self.assertFalse(extract.is_remote(None))
        self.assertEqual(extract.parse("r2://ptcg-foil/tcglive/"), ("ptcg-foil", "tcglive"))
        self.assertEqual(extract.parse("r2://ptcg-foil"), ("ptcg-foil", ""))
        self.assertEqual(extract.parse("r2://b/a/b"), ("b", "a/b"))
        with self.assertRaises(extract.ExtractError):
            extract.parse("r2://")

    def test_local_root_is_a_plain_path(self):
        self.assertEqual(extract.path("E:/x", "cardart", "manifests.jsonl"), Path("E:/x", "cardart", "manifests.jsonl"))


@unittest.skipIf(ClientError is None, "needs boto3 (botocore)")
class BucketTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.cache = Path(tmp.name)
        for patch in (mock.patch.object(extract.cache, "CACHE", self.cache),
                      mock.patch.object(extract, "_missing", set())):
            patch.start()
            self.addCleanup(patch.stop)

    def use(self, s3):
        patch = mock.patch.object(extract, "_s3", return_value=s3)
        patch.start()
        self.addCleanup(patch.stop)
        return s3

    def test_downloads_once_into_the_mirror(self):
        s3 = self.use(FakeS3({"tcglive/cardart/manifests.jsonl": b"[]\n"}))
        p = extract.path("r2://ptcg-foil/tcglive", "cardart", "manifests.jsonl")
        self.assertEqual(p, self.cache / "r2" / "ptcg-foil" / "tcglive" / "cardart" / "manifests.jsonl")
        self.assertEqual(p.read_bytes(), b"[]\n")
        extract.path("r2://ptcg-foil/tcglive", "cardart", "manifests.jsonl")
        self.assertEqual(s3.calls, [("ptcg-foil", "tcglive/cardart/manifests.jsonl")])
        self.assertEqual([f.name for f in p.parent.iterdir()], ["manifests.jsonl"])     # no .part left over

    def test_empty_prefix_and_spaces(self):
        key = "pokemon_Pokemon TCG Live/shadersbundle/Texture2D/T.png"
        s3 = self.use(FakeS3({key: b"png"}))
        p = extract.path("r2://bucket", "pokemon_Pokemon TCG Live", "shadersbundle", "Texture2D", "T.png")
        self.assertTrue(p.exists())
        self.assertEqual(s3.calls, [("bucket", key)])

    def test_missing_key_is_a_missing_file_asked_once(self):
        s3 = self.use(FakeS3({}))
        for _ in range(2):
            self.assertFalse(extract.path("r2://b/p", "cardart", "masks", "sv1", "x.png").exists())
        self.assertEqual(len(s3.calls), 1)

    def test_unreadable_bucket_is_an_error(self):
        self.use(FakeS3({}, denied={"p/cardart/manifests.jsonl"}))
        with self.assertRaises(extract.ExtractError):
            extract.path("r2://b/p", "cardart", "manifests.jsonl")

    def test_no_escaping_the_mirror(self):
        self.use(FakeS3({}))
        with self.assertRaises(ValueError):
            extract.path("r2://b/p", "..", "secret")

    def test_tcglive_front_from_the_bucket(self):
        buf = Path(self.cache, "front.webp")
        Image.new("RGB", (1024, 1024), (10, 200, 30)).save(buf, "WEBP", lossless=True)
        s3 = self.use(FakeS3({"tl/WebApp/assets/cards/swsh6_99.webp": buf.read_bytes()}))
        with mock.patch.dict("sys.modules", {"UnityPy": None}):        # no UnityPy: the bundle is not fetched
            front = tcg.tcglive_front("swsh6-99", root="r2://ptcg-foil/tl")
        self.assertEqual(front.size, (1024, 1024))
        self.assertEqual(front.getpixel((0, 0)), (10, 200, 30))
        self.assertEqual(s3.calls, [("ptcg-foil", "tl/WebApp/assets/cards/swsh6_99.webp")])


if __name__ == "__main__":
    unittest.main()
