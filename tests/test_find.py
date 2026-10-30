"""Card search (tcg.find, /api/find): the TCGdex query it builds and how it sorts and trims the answer. TCGdex is
mocked. Run: python -m unittest discover tests"""
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from svrestyle import gui, tcg

SETS = [
    {"id": "base1", "name": "Base Set", "logo": "https://assets.tcgdex.net/en/base/base1/logo"},
    {"id": "sv03.5", "name": "151", "logo": "https://assets.tcgdex.net/en/sv/sv03.5/logo"},
    {"id": "A1", "name": "Genetic Apex", "logo": "https://assets.tcgdex.net/en/tcgp/A1/logo"},
    {"id": "30th", "name": "30th Celebration", "logo": "https://assets.tcgdex.net/en/me/30th/logo"},
]
BRIEFS = [
    {"id": "sv03.5-065", "localId": "065", "name": "Alakazam ex", "image": "https://assets.tcgdex.net/en/sv/sv03.5/065"},
    {"id": "base1-58", "localId": "58", "name": "Pikachu", "image": "https://assets.tcgdex.net/en/base/base1/58"},
    {"id": "base1-9", "localId": "9", "name": "Blastoise"},
    {"id": "A1-094", "localId": "094", "name": "Pikachu ex", "image": "https://assets.tcgdex.net/en/tcgp/A1/094"},
    {"id": "30th-053", "localId": "053", "name": "Pikachu ex", "image": "https://assets.tcgdex.net/en/me/30th/053"},
    {"id": "sv03.5-007", "localId": "007", "name": "Squirtle", "image": "https://assets.tcgdex.net/en/sv/sv03.5/007"},
]


def answer(url, name, days=1):
    return {"sets.json": SETS, "rarities.json": ["Common", "One Diamond", "Crown", "Double rare"]}.get(name, BRIEFS)


class Find(unittest.TestCase):
    def run_find(self, *args, **kwargs):
        with mock.patch.object(tcg, "_fresh_json", side_effect=answer) as fresh:
            out = tcg.find(*args, **kwargs)
        url = next(c.args[0] for c in fresh.call_args_list if "/cards?" in c.args[0])
        return out, url

    def test_query(self):
        _, url = self.run_find("Farfetch’d", {"rarity": ["Double rare", "Ultra Rare"], "set": ["sv03.5"],
                                               "types": ["Fire"], "hpMin": 100, "hpMax": 200})
        parts = url.split("?", 1)[1].split("&")
        self.assertIn("name=Farfetch%27d", parts)
        self.assertIn("rarity=eq:Double%20rare%7CUltra%20Rare", parts)       # exact values, any of them
        self.assertIn("set.id=eq:sv03.5", parts)
        self.assertIn("types=eq:Fire", parts)
        self.assertIn("hp=gte:100", parts)
        self.assertIn("hp=lte:200", parts)
        self.assertIn("category=eq:Pokemon%7CTrainer%7CEnergy", parts)

    def test_energy_kind_filter(self):
        _, url = self.run_find("", {"category": ["Energy"], "energyType": ["Special"]})
        self.assertIn("energyType=eq:Special", url)
        self.assertIn("category=eq:Energy", url)

    def test_category_picked(self):
        _, url = self.run_find("", {"category": ["Trainer"]})
        self.assertIn("category=eq:Trainer", url)
        self.assertNotIn("Pokemon", url)

    def test_rejections(self):
        for args in [("", {}), ("  ", None), ("x", {"category": ["Stadium"]}), ("x", {"hpMin": "a"}),
                     ("x", {"hpMin": True}), ("x", {"hpMax": 5000}), ("x", {"rarity": "Rare"}), ("x", {"set": [""]}),
                     ("x", [1])]:
            with self.subTest(args=args), mock.patch.object(tcg, "_fresh_json", side_effect=answer):
                with self.assertRaises(ValueError):
                    tcg.find(*args)

    def test_pocket_and_unreadable_ids_left_out_newest_set_first(self):
        out, _ = self.run_find("pikachu")
        # A1 is TCG Pocket, 30th-053 is an id the editor cannot take; the Pokemon TCG's newer set comes first,
        # numbers in order within a set (9 before 58)
        self.assertEqual([c["id"] for c in out["cards"]], ["sv03.5-007", "sv03.5-065", "base1-9", "base1-58"])
        self.assertEqual(out["total"], 4)
        self.assertEqual(out["cards"][1], {"id": "sv03.5-065", "name": "Alakazam ex", "localId": "065", "set": "151",
                                           "thumb": "https://assets.tcgdex.net/en/sv/sv03.5/065/low.webp"})
        self.assertEqual(out["cards"][2]["thumb"], "")                       # no art on TCGdex

    def test_pages(self):
        out, _ = self.run_find("pikachu", offset=1, limit=2)
        self.assertEqual((out["total"], [c["id"] for c in out["cards"]]), (4, ["sv03.5-065", "base1-9"]))
        out, _ = self.run_find("pikachu", offset=3, limit=500)
        self.assertEqual([c["id"] for c in out["cards"]], ["base1-58"])

    def test_sets_and_rarities_for_the_filters(self):
        with mock.patch.object(tcg, "_fresh_json", side_effect=answer):
            self.assertEqual([s["id"] for s in tcg.sets()], ["sv03.5", "base1"])   # newest first, no Pocket, no 30th
            self.assertEqual(tcg.rarities(), ["Common", "Double rare"])


class FreshJson(unittest.TestCase):
    def test_asks_again_when_old_and_keeps_the_old_answer_when_tcgdex_is_down(self):
        import os
        import tempfile
        import time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tcg.cache, "CACHE", Path(tmp)):
            dest = tcg.cache.path("data", "sets.json")
            dest.write_text('["old"]')
            self.assertEqual(tcg._fresh_json("u", "sets.json"), ["old"])         # young: no request
            os.utime(dest, (time.time() - 3 * 86400,) * 2)
            with mock.patch.object(tcg.cache, "fetch", side_effect=tcg.cache.DownloadError("down")):
                self.assertEqual(tcg._fresh_json("u", "sets.json"), ["old"])
            self.assertEqual(json.loads(dest.read_text()), ["old"])
            os.utime(dest, (time.time() - 3 * 86400,) * 2)

            def fetch(url, to, **kw):
                Path(to).write_text('["new"]')
                return to
            with mock.patch.object(tcg.cache, "fetch", side_effect=fetch):
                self.assertEqual(tcg._fresh_json("u", "sets.json"), ["new"])


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = gui.make_server(0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def post(self, body):
        req = urllib.request.Request(self.base + "/api/find", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_find_and_its_errors(self):
        with mock.patch.object(tcg, "_fresh_json", side_effect=answer):
            status, out = self.post({"name": "pikachu", "limit": 2})
            self.assertEqual((status, out["total"], len(out["cards"])), (200, 4, 2))
            with urllib.request.urlopen(self.base + "/api/find/options", timeout=10) as r:
                options = json.loads(r.read())
            self.assertEqual(([s["id"] for s in options["sets"]], options["rarities"]),
                             (["sv03.5", "base1"], ["Common", "Double rare"]))
        self.assertEqual(self.post({})[0], 400)
        self.assertEqual(self.post({"name": "x", "filters": {"category": ["Stadium"]}})[0], 400)
        self.assertEqual(self.post({"name": "x", "limit": "many"})[0], 400)
        with mock.patch.object(tcg, "_fresh_json", side_effect=tcg.cache.DownloadError("TCGdex is down")):
            status, out = self.post({"name": "x"})
            self.assertEqual((status, out["error"]), (400, "TCGdex is down"))


if __name__ == "__main__":
    unittest.main()
