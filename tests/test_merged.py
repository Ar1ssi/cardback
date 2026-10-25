"""Every saved project as one download (gui.merged_items, the background zip, /api/projects/merged). Rendering and
TCGdex are mocked. Run: python -m unittest discover tests"""
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

from svrestyle import gui


def entry(card_id, layout="me", **extra):
    return {"id": card_id, "name": card_id, "layout": layout, **extra}


class Merged(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = gui.make_server(0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for p in (mock.patch.object(gui.cache, "CACHE", Path(self.tmp.name)),
                  mock.patch.object(gui, "changelog", return_value="# log"),
                  mock.patch.object(gui.tcg, "TCGLIVE", "x"),
                  mock.patch.object(gui.tcg, "card", return_value={"category": "Energy"})):   # for the skipped entries
            p.start()
            self.addCleanup(p.stop)
        gui._exports.clear()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.finish)

    def finish(self):                                  # no build still writing under the folder that goes
        for job in list(gui._exports.values()):
            self.wait(job["id"])

    def put(self, name, saved, data):
        gui.save_project(name, data)
        f = gui.project_file(name)
        held = json.loads(f.read_text(encoding="utf-8"))
        f.write_text(json.dumps({**held, "saved": saved}), encoding="utf-8")

    def two_projects(self):
        self.put("Older", "2026-10-01T10:00:00", {
            "cards": [entry("swsh3-88"), entry("swsh6-99")], "edits": {"swsh3-88": {"hp": "1"}, "swsh6-99": {"hp": "9"}},
            "options": {}, "foil": False})
        self.put("Newer", "2026-10-02T10:00:00", {
            "cards": [entry("swsh6-99"), entry("sv03.5-065", "modern"), entry("sma-SV33", "trainer"),
                      entry("swsh1-1", skip="not a Pokémon or Trainer")],
            "edits": {"swsh6-99": {"hp": "2"}}, "options": {"swsh6-99": {"style": "panel", "attacks": False}},
            "foil": True})

    def wait(self, job_id, seconds=10):
        end = time.time() + seconds
        while time.time() < end:
            view = gui.merged_progress(job_id)
            if view["state"] != "running":
                return view
            time.sleep(0.02)
        self.fail("the build did not finish")

    def test_one_list_newest_project_wins(self):
        self.two_projects()
        items, projects, repeats = gui.merged_items()
        self.assertEqual((projects, repeats), (2, 1))
        self.assertEqual([i["id"] for i in items], ["swsh6-99", "sv03.5-065", "sma-SV33", "swsh3-88"])   # no Energy
        by = {i["id"]: i for i in items}
        self.assertEqual(by["swsh6-99"], {"id": "swsh6-99", "edits": {"hp": "2"}, "style": "panel", "attacks": False,
                                          "foil": True})                       # the newer project's copy
        self.assertEqual(by["swsh3-88"], {"id": "swsh3-88", "edits": {"hp": "1"}, "style": "hybrid", "attacks": True,
                                          "foil": False})
        self.assertFalse(by["sma-SV33"]["foil"])                               # a Trainer has no foil
        with mock.patch.object(gui.tcg, "TCGLIVE", None):                      # no extract: no foil asked for
            self.assertFalse(any(i["foil"] for i in gui.merged_items()[0]))

    def test_nothing_to_download(self):
        with self.assertRaises(gui.UserError):
            gui.start_merged_zip()
        self.put("Empty", "2026-10-01T10:00:00", {"cards": [entry("swsh1-1", skip="not a Pokémon or Trainer")]})
        with self.assertRaises(gui.UserError):
            gui.start_merged_zip()

    def test_zip_has_every_render_and_names_the_failures(self):
        self.two_projects()

        def render(card_id, edits, style, attacks, holo):
            if card_id == "sv03.5-065":
                raise gui.UserError("TCGdex has no scan")
            return gui.Render(f"png-{card_id}-{edits}-{style}".encode(), [], "me",
                              ({}, {f"{card_id}-foil.json": b"{}"}, {}, "t") if holo else None)
        with mock.patch.object(gui, "render", side_effect=render):
            job = gui.start_merged_zip()
            self.assertEqual((job["total"], job["projects"], job["repeats"]), (4, 2, 1))
            done = self.wait(job["id"])
        self.assertEqual((done["state"], done["done"], done["failed"]), ("done", 4, [
            {"id": "sv03.5-065", "why": "TCGdex has no scan"}]))
        with zipfile.ZipFile(gui.merged_file(job["id"])) as z:
            self.assertEqual(sorted(z.namelist()), ["CHANGELOG.md", "NOT-INCLUDED.txt", "sma-SV33-me.png",
                                                    "swsh3-88-me.png", "swsh6-99-foil.json", "swsh6-99-me.png"])
            self.assertEqual(z.read("swsh6-99-me.png"), b"png-swsh6-99-{'hp': '2'}-panel")
            self.assertIn("sv03.5-065: TCGdex has no scan", z.read("NOT-INCLUDED.txt").decode())

    def test_a_build_running_is_joined(self):
        self.two_projects()
        release = threading.Event()

        def render(*args):
            release.wait(10)
            return gui.Render(b"png", [], "me")
        with mock.patch.object(gui, "render", side_effect=render):
            first = gui.start_merged_zip()
            self.assertEqual(gui.start_merged_zip()["id"], first["id"])
            self.assertEqual(gui.merged_progress(first["id"])["state"], "running")
            with self.assertRaises(LookupError):                               # not ready to download yet
                gui.merged_file(first["id"])
            release.set()
            self.assertEqual(self.wait(first["id"])["state"], "done")
        self.assertNotEqual(gui.start_merged_zip()["id"], first["id"])         # a finished one is not reused
        self.finish()

    def test_every_card_failing_is_a_failed_build(self):
        self.two_projects()
        with mock.patch.object(gui, "render", side_effect=gui.UserError("no fonts")):
            done = self.wait(gui.start_merged_zip()["id"])
        self.assertEqual(done["state"], "failed")
        self.assertIn("no card could be rendered", done["error"])
        self.assertIn("no fonts", done["error"])
        self.assertEqual(list(gui.exports_dir().glob("*.zip")), [])
        with self.assertRaises(LookupError):
            gui.merged_file(done["id"])

    def test_only_the_last_builds_are_kept(self):
        self.two_projects()
        with mock.patch.object(gui, "render", return_value=gui.Render(b"png", [], "me")):
            ids = []
            for _ in range(gui.MERGED_KEEP + 1):
                ids.append(gui.start_merged_zip()["id"])
                self.wait(ids[-1])
        with self.assertRaises(LookupError):
            gui.merged_progress(ids[0])
        self.assertEqual(sorted(f.stem for f in gui.exports_dir().glob("*.zip")), sorted(ids[1:]))

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def test_http(self):
        self.assertEqual(self.call("/api/projects/merged", {})[0], 400)        # nothing saved
        self.two_projects()
        with mock.patch.object(gui, "render", return_value=gui.Render(b"png", [], "me")):
            status, _, body = self.call("/api/projects/merged", {})
            job = json.loads(body)
            self.assertEqual((status, job["total"]), (200, 4))
            self.wait(job["id"])
        status, _, body = self.call(f"/api/projects/merged?job={job['id']}")
        self.assertEqual((status, json.loads(body)["state"]), (200, "done"))
        status, headers, body = self.call(f"/api/projects/merged/file?job={job['id']}")
        self.assertEqual((status, headers["Content-Type"], int(headers["Content-Length"])), (200, "application/zip", len(body)))
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn("CHANGELOG.md", zipfile.ZipFile(io.BytesIO(body)).namelist())
        self.assertEqual(self.call("/api/projects/merged?job=nope")[0], 404)
        self.assertEqual(self.call("/api/projects/merged/file?job=nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
