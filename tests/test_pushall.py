"""Push all: every saved project merged into one set, and what pushing it would change (gui.push_all_plan,
/api/push-plan, gui.keep_numbers). TCGdex and rendering are mocked. Run: python -m unittest discover tests"""
import functools
import hashlib
import io
import json
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

A, B, C, D, M = "swsh3-88", "base1-91", "swsh3-89", "swsh4-43", "sv01-001"
NAMES = {A: "Tyranitar", B: "Bill", C: "Larvitar", D: "Pikachu", M: "Sprigatito"}


@functools.lru_cache(maxsize=None)
def png(seed="x"):
    buf = io.BytesIO()
    Image.new("RGBA", (733, 1024), tuple(hashlib.md5(seed.encode()).digest()[:3]) + (255,)).save(buf, "PNG")
    return buf.getvalue()


def card(card_id):
    return {"id": card_id, "family": "regular", "type": "Fighting", "prefix": "", "name": NAMES[card_id], "hp": "180",
            "stage": 0, "evolves_from": "", "portrait": "", "illustrator": "x", "reg_mark": "D", "rarity": "rare",
            "flavor": "", "abilities": [], "attacks": [{"cost": ["Fighting"], "name": "Hit", "damage": "10", "effect": ""}],
            "weakness": None, "resistance": None, "retreat": 1}


def entry(card_id):
    return {"id": card_id, "name": NAMES[card_id], "layout": "me"}


class PushAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.checkout = Path(self.tmp.name, "sim")
        (self.checkout / "client" / "src").mkdir(parents=True)
        self.set_dir = self.checkout.joinpath(*gui.SIM_SETS) / "customhbw"
        for patcher in (mock.patch.object(gui.cache, "CACHE", Path(self.tmp.name, "cache")),
                        mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": str(self.checkout)}),
                        mock.patch.object(gui, "read", side_effect=lambda cid: ("me", card(cid))),
                        mock.patch.object(gui, "render", side_effect=lambda cid, *a: gui.Render(png(cid), [], "me")),
                        mock.patch.object(gui, "changelog", return_value="# log"),
                        mock.patch.object(gui.tcg, "TCGLIVE", None),
                        mock.patch.object(gui.tcg, "card", side_effect=lambda cid: {"name": NAMES[cid]})):
            patcher.start()
            self.addCleanup(patcher.stop)
        gui._set_jobs.clear()
        self.addCleanup(self.finish)

    def finish(self):
        for job in list(gui._set_jobs.values()):
            gui.cancel_set_job(job["id"])
            self.wait(job["id"])

    def wait(self, job_id, seconds=15):
        end = time.time() + seconds
        while time.time() < end:
            view = gui.set_job_progress(job_id)
            if view["state"] != "running":
                return view
            time.sleep(0.02)
        self.fail("the job did not finish")

    def put(self, name, saved, cards, edits=None, options=None):
        gui.save_project(name, {"cards": [entry(c) for c in cards], "edits": edits or {}, "options": options or {},
                                "foil": False})
        f = gui.project_file(name)
        f.write_text(json.dumps({**json.loads(f.read_text(encoding="utf-8")), "saved": saved}), encoding="utf-8")

    def push(self, items, code="HBW", title="Homebrew Wave 1", **extra):
        return self.wait(gui.start_set_job("push", items, title, code, **extra)["id"])

    def sources(self):
        record = json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))
        return [c["custom"]["source"] for c in record["cards"]]

    def test_the_cards_already_in_the_set_keep_their_numbers(self):
        existing = [(A, "x"), (B, "y"), (C, "z")]
        items = [{"id": i} for i in (D, C, B, M, A)]
        self.assertEqual([i["id"] for i in gui.keep_numbers(items, existing)], [A, B, C, D, M])
        self.assertEqual([i["id"] for i in gui.keep_numbers(items, None)], [D, C, B, M, A])
        self.assertEqual([i["id"] for i in gui.keep_numbers(items, [(None, "x"), (None, "y")])], [D, C, B, M, A])

    def test_the_plan_lists_new_changed_and_removed_cards_with_what_changed(self):
        self.push([{"id": A}, {"id": B}, {"id": C}])
        self.put("Older", "2026-10-01T10:00:00", [A, B], edits={A: {"hp": "190"}})
        self.put("Newer", "2026-10-02T10:00:00", [B, D], options={B: {"style": "hybrid", "attacks": False}})
        plan = gui.push_all_plan("HBW")
        self.assertEqual((plan["setId"], plan["code"], plan["title"], plan["projects"], plan["repeats"]),
                         ("customhbw", "HBW", "Homebrew Wave 1", 2, 1))
        self.assertEqual([i["id"] for i in plan["items"]], [A, B, D], "the set's order, the new card last")
        self.assertEqual((plan["incoming"], plan["existing"], plan["moved"], plan["unchanged"]), (3, 3, 0, 0))
        self.assertEqual(plan["added"], [{"id": D, "name": "Pikachu", "lines": ["Restyled as a Mega Evolution card"]}])
        self.assertEqual(plan["changed"], [{"id": A, "name": "Tyranitar", "lines": ["HP: 180 → 190"]},
                                           {"id": B, "name": "Bill", "lines": ["Abilities and attacks: left off"]}])
        self.assertEqual(plan["removed"], [{"id": C, "name": "Larvitar"}])
        self.assertEqual(self.sources(), [A, B, C], "planning writes nothing")
        done = self.push(plan["items"], title=plan["title"], allow_shrink=True)
        self.assertEqual((done["state"], done["result"]["rendered"], done["result"]["removed"]), ("done", 3, []),
                         "the new card takes the removed one's number: no folder is left over")
        self.assertEqual(self.sources(), [A, B, D])
        again = gui.push_all_plan("HBW")
        self.assertEqual((again["added"], again["changed"], again["removed"], again["moved"], again["unchanged"]),
                         ([], [], [], 0, 3), "nothing left to push")

    def test_the_push_renders_exactly_the_cards_the_plan_lists(self):
        self.push([{"id": A}, {"id": B}, {"id": C}, {"id": M}])
        record = json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))
        for card in record["cards"][:2]:                              # A and B: a set from before digests were kept
            card["custom"].pop("files")
        (self.set_dir / "set.json").write_text(json.dumps(record), encoding="utf-8")
        (self.set_dir / "003" / "high.webp").write_bytes(b"damaged")  # C's image is not the one its record names
        self.put("Only", "2026-10-02T10:00:00", [A, B, C, D], edits={B: {"hp": "190"}})
        plan = gui.push_all_plan("HBW")
        self.assertEqual(sorted(c["id"] for c in plan["added"]), [D])
        self.assertEqual(sorted(c["id"] for c in plan["changed"]), [B, C])
        self.assertEqual((plan["unchanged"], [c["id"] for c in plan["removed"]]), (1, [M]), "A is kept as it is")
        self.assertIn("Its images in the set are not the ones its record names: rendered again",
                      next(c for c in plan["changed"] if c["id"] == C)["lines"])
        gui.render.reset_mock()
        done = self.push(plan["items"], allow_shrink=True)
        self.assertEqual(sorted(call.args[0] for call in gui.render.call_args_list), [B, C, D], "what the plan listed")
        self.assertEqual((done["result"]["rendered"], done["result"]["reused"]), (3, 1))
        after = gui.push_all_plan("HBW")
        self.assertEqual((after["added"], after["changed"], after["removed"], after["unchanged"]), ([], [], [], 4))

    def test_a_first_push_has_everything_new_and_a_card_that_cannot_be_read_says_so(self):
        self.put("Only", "2026-10-02T10:00:00", [A, D, M])

        def read(card_id):
            if card_id == D:
                raise gui.UserError("TCGdex has no such card")
            return "me", card(card_id)
        with mock.patch.object(gui, "read", side_effect=read):
            plan = gui.push_all_plan("HBW")
        self.assertEqual((plan["existing"], plan["title"], plan["changed"], plan["removed"], plan["moved"]), (None, "", [], [], 0))
        self.assertEqual([(c["id"], c["name"]) for c in plan["added"]], [(A, "Tyranitar"), (D, D), (M, "Sprigatito")])
        self.assertEqual(plan["added"][1]["lines"], ["cannot be read: TCGdex has no such card"])
        self.assertEqual(plan["added"][2]["lines"], [], "a Scarlet & Violet card left as printed has nothing to list")

    def test_requests_the_plan_cannot_answer(self):
        with self.assertRaisesRegex(gui.UserError, "no saved project has a card"):
            gui.push_all_plan("HBW")
        self.put("Only", "2026-10-02T10:00:00", [A])
        for code in ("bad code", None, "H"):
            with self.subTest(code=code), self.assertRaisesRegex(gui.UserError, "set code"):
                gui.push_all_plan(code)
        with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": ""}), self.assertRaisesRegex(gui.UserError, "SVRESTYLE_PTCG_SIM"):
            gui.push_all_plan("HBW")

    def test_over_http_only_a_trusted_browser_gets_the_plan(self):
        server = gui.make_server(0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def post(body, headers=None):
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/api/push-plan", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                with e:
                    return e.code, json.loads(e.read())
        self.put("Only", "2026-10-02T10:00:00", [A, D])
        status, plan = post({"code": "HBW"})
        self.assertEqual((status, plan["incoming"], [i["id"] for i in plan["items"]]), (200, 2, [A, D]))
        self.assertEqual(post({"code": "HBW"}, {"Cf-Connecting-Ip": "203.0.113.9"})[0], 403)
        self.assertEqual(post({"code": "nope"})[0], 400)


if __name__ == "__main__":
    unittest.main()
