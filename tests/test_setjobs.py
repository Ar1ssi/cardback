"""Send to ptcg-sim and Publish to site as background jobs (gui.start_set_job, /api/push-set, /api/publish-set,
/api/set-job) and the check that refuses to take cards out of a set that is already there. Rendering and TCGdex are
mocked; the publish script is a stand-in. Run: python -m unittest discover tests"""
import functools
import hashlib
import io
import json
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from PIL import Image

from svrestyle import gui

SET_ID = "customhbw"
A, B, C = "swsh3-88", "base1-91", "swsh3-89"
NAMES = {A: "Tyranitar", B: "Bill", C: "Larvitar"}
# stands in for the simulator's scripts/custom-sets/publish.mjs: a report, or (FAKE_PUBLISH_HANG) a stalled upload
# whose child process would write the marker file if it outlived the kill
PUBLISH = ("const args = process.argv.slice(2);\n"
           "if (process.env.FAKE_PUBLISH_HANG) {\n"
           "  process.getBuiltinModule('child_process').spawn(process.execPath, ['-e',\n"
           "    `setTimeout(() => require('fs').writeFileSync(${JSON.stringify(process.env.FAKE_PUBLISH_HANG)}, 'x'), "
           "2500)`], { stdio: 'inherit' });\n"
           "  setTimeout(() => {}, 60000);\n"
           "} else {\n"
           "  console.log(JSON.stringify({ sets: [{ setId: args[0], uploaded: 3, uploadedBytes: 4096, reused: 1, "
           "skipped: [], unchanged: false }], deleted: 0, dryRun: false, args }));\n"
           "}\n")


@functools.lru_cache(maxsize=None)
def png(seed="x"):
    color = tuple(hashlib.md5(seed.encode()).digest()[:3]) + (255,)
    buf = io.BytesIO()
    Image.new("RGBA", (733, 1024), color).save(buf, "PNG")
    return buf.getvalue()


def card(card_id):
    return {"id": card_id, "family": "regular", "type": "Fighting", "prefix": "", "name": NAMES.get(card_id, card_id),
            "hp": "100", "stage": 0, "evolves_from": "", "illustrator": "x", "reg_mark": "D", "rarity": "rare",
            "flavor": "", "abilities": [], "attacks": [{"cost": ["Fighting"], "name": "Hit", "damage": "10", "effect": ""}],
            "weakness": None, "resistance": None, "retreat": 1}


def fake_render(card_id, *rest):
    return gui.Render(png(card_id), [], "me")


class JobCase(unittest.TestCase):
    """A simulator checkout in a temp folder with the stand-in publish script, and a bucket remote."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkout = Path(self.tmp.name)
        (self.checkout / "client" / "src").mkdir(parents=True)
        script = self.checkout.joinpath(*gui.PUBLISH_SCRIPT)
        script.parent.mkdir(parents=True)
        script.write_text(PUBLISH, encoding="utf-8")
        self.set_dir = self.checkout.joinpath(*gui.SIM_SETS) / SET_ID
        for patcher in (mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": str(self.checkout),
                                                       "CUSTOM_SETS_R2_REMOTE": "r2:bucket/sets"}),
                        mock.patch.object(gui, "read", side_effect=lambda cid: ("me", card(cid))),
                        mock.patch.object(gui, "render", side_effect=fake_render),
                        mock.patch.object(gui, "changelog", return_value="# log"),
                        mock.patch.object(gui, "site_card_count", return_value=None),
                        mock.patch.object(gui, "takes_reverse", return_value=False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        gui._set_jobs.clear()
        self.addCleanup(self.finish)          # runs first: no job still writing under the folder that goes

    def finish(self):
        for job in list(gui._set_jobs.values()):
            gui.cancel_set_job(job["id"])
            self.wait(job["id"])

    def start(self, kind="push", items=None, code="HBW", **extra):
        return gui.start_set_job(kind, items or [{"id": A}], "T", code, **extra)

    def wait(self, job_id, seconds=15):
        end = time.time() + seconds
        while time.time() < end:
            view = gui.set_job_progress(job_id)
            if view["state"] != "running":
                return view
            time.sleep(0.02)
        self.fail("the job did not finish")

    def until(self, condition, seconds=10):
        end = time.time() + seconds
        while time.time() < end:
            if condition():
                return
            time.sleep(0.02)
        self.fail("the job did not get there")

    def send(self, ids, kind="push", **extra):
        """A send or publish of these cards, waited for -> its last progress."""
        return self.wait(self.start(kind, [{"id": i} for i in ids], **extra)["id"])

    def snapshot(self):
        return {str(p.relative_to(self.set_dir)): p.read_bytes() for p in self.set_dir.rglob("*") if p.is_file()}


class Jobs(JobCase):
    def test_a_send_runs_in_the_background_and_shows_its_progress(self):
        release = threading.Event()

        def render(card_id, *rest):
            if card_id == B:
                release.wait(10)
            return fake_render(card_id)
        with mock.patch.object(gui, "render", side_effect=render):
            job = self.start(items=[{"id": A}, {"id": B}])
            self.assertEqual((job["kind"], job["state"], job["setId"], job["total"]), ("push", "running", SET_ID, 2))
            self.until(lambda: gui.set_job_progress(job["id"])["current"] == B)
            view = gui.set_job_progress(job["id"])
            self.assertEqual((view["phase"], view["done"], view["total"], view["result"]), ("rendering", 1, 2, None))
            release.set()
            done = self.wait(job["id"])
        self.assertEqual((done["state"], done["phase"], done["done"], done["total"]), ("done", "writing", 6, 6))
        self.assertEqual((done["result"]["rendered"], done["result"]["reused"]), (2, 0))
        self.assertEqual(done["pushed"], done["result"])
        self.assertIsNone(done["error"])

    def test_asking_again_follows_the_job_running_but_another_set_is_refused(self):
        release = threading.Event()
        with mock.patch.object(gui, "render", side_effect=lambda *a: release.wait(10) and fake_render(A)):
            first = self.start()
            self.assertEqual(self.start()["id"], first["id"], "the same send is followed, not started twice")
            for kwargs in ({"code": "ABC"}, {"kind": "publish"}):
                with self.subTest(kwargs=kwargs), self.assertRaisesRegex(gui.UserError, "send of HBW is still running"):
                    self.start(**kwargs)
            release.set()
            self.wait(first["id"])
            self.assertNotEqual(self.start()["id"], first["id"], "a finished job is not joined")

    def test_a_cancel_while_rendering_changes_nothing(self):
        done = self.send([A, B])
        self.assertEqual(gui.cancel_set_job(done["id"])["cancelling"], False, "cancelling a finished job does nothing")
        before = self.snapshot()
        release, started = threading.Event(), threading.Event()

        def render(card_id, *rest):
            started.set()
            release.wait(10)
            return fake_render(card_id)
        with mock.patch.object(gui, "render", side_effect=render):
            job = self.start(items=[{"id": A, "style": "panel"}, {"id": B}])
            started.wait(10)
            stopping = gui.cancel_set_job(job["id"])
            self.assertEqual((stopping["state"], stopping["cancelling"]), ("running", True))
            release.set()
            cancelled = self.wait(job["id"])
        self.assertEqual((cancelled["state"], cancelled["pushed"], cancelled["result"], cancelled["error"]),
                         ("cancelled", None, None, None))
        self.assertEqual(self.snapshot(), before)

    def test_a_cancel_while_uploading_stops_the_script_and_keeps_the_push(self):
        marker = Path(self.tmp.name, "child-outlived-the-kill")
        with mock.patch.dict("os.environ", {"FAKE_PUBLISH_HANG": str(marker)}):
            job = self.start("publish")
            self.until(lambda: gui.set_job_progress(job["id"])["phase"] == "uploading")
            time.sleep(0.5)                                  # node is running, its child started
            gui.cancel_set_job(job["id"])
            cancelled = self.wait(job["id"])
        self.assertEqual((cancelled["state"], cancelled["error"], cancelled["result"]), ("cancelled", None, None))
        self.assertEqual(cancelled["pushed"]["setId"], SET_ID, "the files are in ptcg-sim")
        time.sleep(3.5)
        self.assertFalse(marker.exists(), "the script's child process was killed with it")

    def test_a_publish_reports_its_upload(self):
        done = self.send([A, B], "publish")
        self.assertEqual((done["state"], done["phase"], done["kind"]), ("done", "uploading", "publish"))
        self.assertEqual(done["result"]["published"]["args"], [SET_ID, "--remote", "r2:bucket/sets", "--json"])
        self.assertEqual(done["result"]["published"]["sets"][0]["uploaded"], 3)
        self.assertEqual(done["pushed"]["rendered"], 2)

    def test_failures_end_the_job_with_their_message(self):
        with mock.patch.object(gui, "read", side_effect=gui.UserError("TCGdex has no scan")):
            failed = self.send([A])
        self.assertEqual((failed["state"], failed["error"], failed["pushed"]), ("failed", "TCGdex has no scan", None))
        for boom, text in ((RuntimeError("boom"), "RuntimeError: boom"), (SystemExit("no scan"), "SystemExit: no scan")):
            with self.subTest(boom=boom), mock.patch.object(gui, "render", side_effect=boom):
                failed = self.send([A])
                self.assertEqual((failed["state"], failed["error"]), ("failed", text))
        self.assertFalse(self.set_dir.exists(), "nothing was written")

    def test_only_the_last_jobs_are_kept(self):
        with mock.patch.object(gui, "SET_JOB_KEEP", 2):
            ids = [self.send([A])["id"] for _ in range(3)]
        with self.assertRaises(LookupError):
            gui.set_job_progress(ids[0])
        self.assertEqual(gui.set_job_progress(ids[2])["state"], "done")
        with self.assertRaises(LookupError):
            gui.cancel_set_job("nope")

    def test_what_the_page_sends_is_checked_before_a_job_starts(self):
        for kwargs, why in (({"items": []}, "items: a non-empty list"), ({"items": ["swsh3-88"]}, "expected objects"),
                            ({"items": [{"id": 5}]}, "expected objects"), ({"code": "bad code"}, "set code"),
                            ({"code": None}, "set code")):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(gui.UserError, why):
                gui.start_set_job("push", kwargs.get("items", [{"id": A}]), "T", kwargs.get("code", "HBW"))
        self.assertEqual(gui._set_jobs, {})


class Drops(JobCase):
    def sources(self):
        record = json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))
        return [c["custom"]["source"] for c in sorted(record["cards"], key=lambda c: c["localId"])]   # set.json is in display order

    def unsource(self):
        """The set as a hand-made one: its cards name no card they were made from, so a push cannot send them again."""
        record = json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))
        for c in record["cards"]:
            c["custom"].pop("source")
        (self.set_dir / "set.json").write_text(json.dumps(record), encoding="utf-8")

    def test_cards_the_list_does_not_name_stay_in_the_set(self):
        self.send([A, B, C])
        done = self.send([A])                                # a list with one card: B and C are not taken out
        self.assertEqual((done["state"], done["result"]["removed"]), ("done", []))
        self.assertEqual(self.sources(), [A, B, C])
        self.assertEqual(self.send([C])["state"], "done")    # reordered: each keeps its number
        self.assertEqual(self.sources(), [A, B, C])
        done = self.send([A, "sv1-1"])                       # a new card goes after them
        self.assertEqual((done["state"], self.sources()), ("done", [A, B, C, "sv1-1"]))

    def test_the_cards_that_stay_keep_their_edits_and_options(self):
        job = self.start(items=[{"id": A, "edits": {"hp": "190"}}, {"id": B, "style": "panel", "attacks": False}])
        self.wait(job["id"])
        self.send([C])
        record = json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))
        custom = {c["custom"]["source"]: c["custom"] for c in sorted(record["cards"], key=lambda c: c["localId"])}
        self.assertEqual((custom[A]["edits"], custom[B]["style"], custom[B]["attacks"]),
                         ({"hp": "190"}, "panel", False))
        self.assertEqual(list(custom), [A, B, C])

    def test_a_set_whose_cards_cannot_be_sent_again_is_refused_until_confirmed(self):
        self.send([A, B, C])
        self.unsource()
        before = self.snapshot()
        with self.assertRaises(gui.DropsCards) as caught:
            self.start(items=[{"id": A}])
        info = caught.exception.info
        self.assertEqual(info["message"], "HBW has 3 cards in ptcg-sim; this list has 1: 2 would be removed (Bill, Larvitar).")
        self.assertEqual((info["setId"], info["code"], info["incoming"], info["names"]), (SET_ID, "HBW", 1, ["Bill", "Larvitar"]))
        self.assertEqual(info["copies"], [{"where": "ptcg-sim", "cards": 3, "dropped": 2}])
        self.assertEqual((self.snapshot(), gui._running_set_job()), (before, None), "nothing started, nothing written")
        done = self.send([A], allow_shrink=True)
        self.assertEqual((done["state"], done["result"]["removed"]), ("done", ["002", "003"]))

    def test_the_names_are_a_few(self):
        names = {f"sv1-{n}": f"Card {n}" for n in range(1, 9)}
        with mock.patch.dict(NAMES, names):
            self.send(list(names))
            self.unsource()
            with self.assertRaises(gui.DropsCards) as caught:
                self.start(items=[{"id": "sv1-1"}])
        self.assertIn("(Card 2, Card 3, Card 4, Card 5, Card 6 and 2 more)", caught.exception.info["message"])
        self.assertEqual(caught.exception.info["names"], ["Card 2", "Card 3", "Card 4", "Card 5", "Card 6"])

    def test_nothing_taken_out_nothing_asked(self):
        self.send([A, B])                                    # the first send of a set: there is nothing to lose
        for ids in ([A, B], [B, A], [A, B, C]):             # as it is, reordered, longer
            with self.subTest(ids=ids):
                self.assertEqual(self.send(ids)["state"], "done")
        self.assertEqual(self.send([A, B, C])["result"]["unchanged"], 8)
        (self.set_dir / "set.json").write_text("{not json", encoding="utf-8")        # nothing readable to compare with
        self.assertEqual(self.send([A])["state"], "done")

    def test_a_publish_is_checked_against_the_site_copy_too(self):
        self.send([A, B, C])
        with mock.patch.object(gui, "site_card_count", return_value=5) as counted:
            with self.assertRaises(gui.DropsCards) as caught:
                self.start("publish", [{"id": i} for i in (A, B, C)])      # ptcg-sim loses nothing; the site loses 2
            info = caught.exception.info
            self.assertEqual(info["copies"], [{"where": "the site", "cards": 5, "dropped": 2}])
            self.assertEqual(info["message"], "The site has 5 cards of HBW; this list has 3: 2 would be removed.")
            counted.assert_called_once_with("r2:bucket/sets", SET_ID)
            with self.assertRaises(gui.DropsCards) as caught:
                self.start("publish", [{"id": A}])                           # B and C stay in ptcg-sim: the site is the same
            self.assertEqual(caught.exception.info["message"],
                             "The site has 5 cards of HBW; this list has 3: 2 would be removed.")
            counted.reset_mock()
            self.assertEqual(self.send([A, B, C], "push")["state"], "done")
            self.assertEqual(self.send([A, B, C], "publish", allow_shrink=True)["state"], "done")
            counted.assert_not_called()                                  # a send never asks the site; a confirmed publish skips it
        for site in (None, 3, 2):                                        # unknown, as long, shorter than the list: nothing to ask
            with self.subTest(site=site), mock.patch.object(gui, "site_card_count", return_value=site):
                self.assertEqual(self.send([A, B, C], "publish")["state"], "done")

    def test_over_http_the_refusal_is_a_409_the_page_can_answer(self):
        server = gui.make_server(0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def post(path, body):
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}{path}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                with e:
                    return e.code, json.loads(e.read())
        self.send([A, B, C])
        status, answer = post("/api/push-set", {"items": [{"id": A}], "title": "T", "code": "HBW"})
        self.assertEqual(status, 200, "the cards the list does not name stay: nothing to ask")
        self.wait(answer["id"])
        self.unsource()
        status, answer = post("/api/push-set", {"items": [{"id": A}], "title": "T", "code": "HBW"})
        self.assertEqual(status, 409)
        self.assertEqual((answer["error"], answer["drops"]["copies"]), (answer["drops"]["message"],
                                                                         [{"where": "ptcg-sim", "cards": 3, "dropped": 2}]))
        self.assertEqual(post("/api/push-set", {"items": [{"id": A}], "code": "HBW", "allowShrink": "no"})[0], 409,
                         "only true confirms")
        status, answer = post("/api/push-set", {"items": [{"id": A}], "code": "HBW", "allowShrink": True})
        self.assertEqual((status, answer["kind"]), (200, "push"))
        self.wait(answer["id"])
        for path, body in (("/api/push-set", {"items": [], "code": "HBW"}), ("/api/publish-set", {"items": [{"id": A}], "code": "x"}),
                           ("/api/set-job/cancel", {"job": "nope"})):
            with self.subTest(path=path):
                self.assertIn(post(path, body)[0], (400, 404))


class SiteCount(unittest.TestCase):
    INDEX = {"sets": {SET_ID: {"files": {name: {} for name in ("001/high.webp", "001/low.webp", "002/high.webp", "002/low.webp",
                                                              "set.json", "CHANGELOG.md", "foil/001.webp")}},
                      "customother": {"files": {"001/high.webp": {}, "002/high.webp": {}, "003/high.webp": {}}}}}

    def count(self, out=None, code=0, set_id=SET_ID, which="rclone-path", error=None):
        done = subprocess.CompletedProcess([], code, json.dumps(self.INDEX) if out is None else out, "")
        with mock.patch.object(gui.shutil, "which", return_value=which), \
                mock.patch.object(gui.subprocess, "run", return_value=done, side_effect=error) as ran:
            return gui.site_card_count("r2:b/s", set_id), ran

    def test_counts_the_card_folders_of_that_set(self):
        counted, ran = self.count()
        self.assertEqual(counted, 2)
        self.assertEqual(ran.call_args.args[0], ["rclone-path", "cat", "r2:b/s/index.json"])
        self.assertEqual(self.count(set_id="customother")[0], 3)

    def test_anything_unreadable_is_unknown(self):
        for what, kwargs in (("set not published", {"set_id": "customnone"}), ("rclone failed", {"code": 1}),
                             ("not json", {"out": "{nope"}), ("not an index", {"out": "[]"}),
                             ("no files", {"out": json.dumps({"sets": {SET_ID: {}}})}),
                             ("no rclone", {"which": None}), ("too slow", {"error": subprocess.TimeoutExpired("rclone", 20)}),
                             ("cannot run", {"error": OSError("not found")}), ("empty", {"out": ""})):
            with self.subTest(what):
                self.assertIsNone(self.count(**kwargs)[0])


if __name__ == "__main__":
    unittest.main()
