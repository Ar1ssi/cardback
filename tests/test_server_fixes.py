"""Server regressions: an ACE SPEC lookup that fails is not stored as "not ACE SPEC"; a project that cannot be read,
and holofoil left off for want of an extract, are said wherever the merged list is used; the render cache is safe
across the server's threads; a failed lookup is not reported as a missing Pokemon; an unexpected error is logged.
TCGdex and rendering are mocked. Run: python -m unittest discover tests"""
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from svrestyle import export, gui, merged, projects, rendering, setjobs

offline = gui.cache.DownloadError("offline")


class AceUnknown(unittest.TestCase):
    def test_a_failed_lookup_is_not_an_answer(self):
        with mock.patch.object(gui.tcg, "card", side_effect=offline):
            self.assertIsNone(rendering._is_ace("sv05-144"))
            entry = rendering._set_entry("sv05-144", "Prime Catcher", "Trainer")
            self.assertEqual(entry, {"id": "sv05-144", "name": "Prime Catcher", "layout": "trainer", "gen9": True})
            saved = [{"id": "sv05-144", "name": "Prime Catcher", "layout": "trainer", "gen9": True}]
            self.assertEqual(rendering._current_entries(saved), saved, "nothing stored while TCGdex cannot say")
            self.assertFalse(setjobs.takes_reverse("sv05-144"), "holofoil stays the switch's choice")
        self.assertFalse(rendering._is_ace("swsh1-156"), "an older Trainer is no ACE SPEC card, no lookup needed")
        with mock.patch.object(gui.tcg, "card", return_value={"rarity": "ACE SPEC Rare"}):
            self.assertEqual(rendering._current_entries(saved)[0]["ace"], True, "the next load asks again")

    def test_an_unknown_ace_sorts_with_the_other_trainers(self):
        recs = [{"category": "Trainer", "trainerType": "Item", "name": n, "localId": f"00{i}",
                 "custom": {"source": f"sv0{i}-1"}} for i, n in enumerate(["B", "A"], 1)]
        order = export.display_order(recs, lambda source: None)
        self.assertEqual([r["name"] for r in order], ["A", "B"])


class Unreadable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for p in (mock.patch.object(gui.cache, "CACHE", Path(self.tmp.name)),
                  mock.patch.object(rendering, "changelog", return_value="# log"),
                  mock.patch.object(gui.tcg, "TCGLIVE", None)):
            p.start()
            self.addCleanup(p.stop)
        merged._exports.clear()
        projects.save_project("Good", {"cards": [{"id": "swsh3-88", "name": "Tyranitar", "layout": "me"}], "foil": True})
        (projects.projects_dir() / "broken.json").write_text("{not json", encoding="utf-8")

    def test_merged_items_carries_what_it_left_out(self):
        merged = projects.merged_items()
        self.assertEqual((merged.projects, [i["id"] for i in merged.items]), (1, ["swsh3-88"]))
        self.assertEqual([u["name"] for u in merged.unreadable], ["broken"])
        self.assertFalse(merged.items[0]["foil"])
        self.assertEqual(len(merged.notes), 1)
        self.assertIn("Good: holofoil left off", merged.notes[0])

    def test_the_zip_and_its_progress_say_so(self):
        with mock.patch.object(rendering, "render", return_value=rendering.Render(b"png", [], "me")):
            job = merged.start_merged_zip()
            end = time.time() + 10
            while merged.merged_progress(job["id"])["state"] == "running" and time.time() < end:
                time.sleep(0.02)
        done = merged.merged_progress(job["id"])
        self.assertEqual(done["state"], "done")
        self.assertEqual([u["name"] for u in done["unreadable"]], ["broken"])
        self.assertTrue(done["notes"])
        with zipfile.ZipFile(merged.merged_file(job["id"])) as z:
            text = z.read("NOT-INCLUDED.txt").decode()
        self.assertIn("could not be read", text)
        self.assertIn("broken: JSONDecodeError", text)
        self.assertIn("holofoil left off", text)

    def test_nothing_to_download_names_the_unreadable(self):
        projects.project_file("Good").unlink()
        with self.assertRaisesRegex(rendering.UserError, "1 project could not be read: broken"):
            merged.start_merged_zip()

    def test_the_push_plan_says_so(self):
        checkout = Path(self.tmp.name, "sim")
        (checkout / "client" / "src").mkdir(parents=True)
        with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": str(checkout)}), \
                mock.patch.object(setjobs, "takes_reverse", return_value=False), \
                mock.patch.object(setjobs, "_plan_card", side_effect=lambda it, *a: {"id": it["id"]}):
            plan = setjobs.push_all_plan("HBW")
        self.assertEqual(([u["name"] for u in plan["unreadable"]], len(plan["notes"])), (["broken"], 1))


class RenderCache(unittest.TestCase):
    def setUp(self):
        rendering._renders.clear()
        self.addCleanup(rendering._renders.clear)
        blank = np.zeros((4, 4, 4), np.float32)
        for p in (mock.patch.object(rendering, "read", return_value=("energy", {"id": "x"})),
                  mock.patch.object(rendering.energy, "compose_card", return_value=(blank, {}))):
            p.start()
            self.addCleanup(p.stop)

    def test_a_hit_does_not_wait_for_a_render_in_progress(self):
        first = rendering.render("sm1-1")
        got = []
        with rendering._render_lock:                        # a render in progress elsewhere
            t = threading.Thread(target=lambda: got.append(rendering.render("sm1-1")), daemon=True)
            t.start()
            t.join(2)
        self.assertEqual(got, [first])

    def test_threads_reading_and_filling_the_cache(self):
        errors, stop = [], threading.Event()

        def fill():
            try:
                for i in range(3000):
                    rendering.render(f"sm1-{i}")
            except Exception as e:                    # noqa: BLE001
                errors.append(e)
            finally:
                stop.set()

        def look():
            while not stop.is_set():
                try:
                    gui.foil_file("t", "a.png")
                except LookupError:
                    pass
                except Exception as e:                # noqa: BLE001 - mutated during iteration
                    errors.append(e)
                    return
        old = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, old)
        threads = [threading.Thread(target=fill)] + [threading.Thread(target=look) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.assertEqual(errors, [])
        self.assertLessEqual(len(rendering._renders), rendering._RENDER_CACHE)


class MegaStage(unittest.TestCase):
    def test_a_failed_lookup_is_not_a_missing_pokemon(self):
        mega = {"name": "Tyranitar", "prefix": "Mega", "evolves_from": "Pupitar"}
        with mock.patch.object(export.tcg, "search", return_value=[{"id": "swsh3-87"}]), \
                mock.patch.object(export.tcg, "card", side_effect=offline), \
                self.assertWarnsRegex(UserWarning, "could not look up 1 of the TCGdex cards named 'Pupitar'"):
            self.assertEqual(export.mega_stage(mega), "Stage1")


class Guard(unittest.TestCase):
    def test_an_unexpected_error_is_logged_a_user_error_is_not(self):
        answers = []
        handler = mock.Mock(_json=lambda obj, status=200: answers.append((status, obj)))

        def fail(e):
            def go():
                raise e
            return go
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            gui.Handler._guard(handler, fail(rendering.UserError("bad id")))
            self.assertEqual(err.getvalue(), "")
            gui.Handler._guard(handler, fail(RuntimeError("boom")))
        self.assertIn("Traceback", err.getvalue())
        self.assertIn("RuntimeError: boom", err.getvalue())
        self.assertEqual([s for s, _ in answers], [400, 500])
        self.assertEqual(answers[1][1], {"error": "RuntimeError: boom"})


if __name__ == "__main__":
    unittest.main()
