"""GUI editing model and HTTP API. No network, model or fonts: card reading and rendering are mocked.
Run: python -m unittest discover tests"""
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from unittest import mock

from svrestyle import edit, gui


def card(**over):
    base = {"id": "swsh3-88", "type": "Fighting", "prefix": "", "name": "Tyranitar", "hp": "180", "illustrator": "x",
            "evolves_from": "Pupitar", "flavor": "f", "abilities": [],
            "attacks": [{"cost": ["Fighting"], "name": "Bedrock Breaker", "damage": 120, "effect": "Discard."}],
            "weakness": {"type": "Grass", "value": "×2"}, "resistance": None, "retreat": 4}
    base.update(over)
    return base


class Apply(unittest.TestCase):
    def test_round_trip_keeps_card(self):
        c = card()
        out = edit.apply(c, edit.editable(c))
        self.assertEqual(out["attacks"][0]["damage"], "120")
        self.assertEqual(out["weakness"], {"type": "Grass", "value": "×2"})
        self.assertIsNot(out, c)

    def test_pokemon_gets_its_accent(self):
        self.assertEqual(edit.printed("Your opponent's Active Pokemon"), "Your opponent’s Active Pokémon")
        self.assertEqual(edit.printed("POKEMON CHECKUP, Pokemon's, pokemon-ex"), "POKÉMON CHECKUP, Pokémon’s, pokémon-ex")
        self.assertEqual(edit.printed("Pokémon stays"), "Pokémon stays")
        self.assertEqual(edit.printed("Slowpokemon"), "Slowpokemon")        # only the word itself
        out = edit.apply(card(), {"attacks": [{"cost": [], "name": "Pokemon Power", "effect": "Heal each Pokemon."}]})
        self.assertEqual((out["attacks"][0]["name"], out["attacks"][0]["effect"]), ("Pokémon Power", "Heal each Pokémon."))

    def test_attack_edits(self):
        out = edit.apply(card(), {"attacks": [
            {"cost": ["Darkness", "Colorless"], "name": "Big Bite", "damage": "50x", "effect": "Don't flip."},
            {"cost": [], "name": "Nap", "damage": "", "effect": ""}]})
        self.assertEqual([a["damage"] for a in out["attacks"]], ["50×", ""])
        self.assertEqual(out["attacks"][0]["effect"], "Don’t flip.")      # printed apostrophe
        self.assertEqual(out["attacks"][1]["cost"], [])

    def test_weakness_resistance_retreat(self):
        out = edit.apply(card(), {"weakness": {"type": "Fire", "value": "+20"},
                                  "resistance": {"type": "Water", "value": "−30"}, "retreat": 0})
        self.assertEqual(out["weakness"]["value"], "+20")
        self.assertEqual(out["resistance"], {"type": "Water", "value": "-30"})
        self.assertEqual(out["retreat"], 0)
        self.assertIsNone(edit.apply(card(), {"weakness": None})["weakness"])

    def test_rejections(self):
        bad = [
            {"attacks": [{"cost": ["Fairy"], "name": "A", "damage": "10"}]},
            {"attacks": [{"cost": [], "name": "A", "damage": "+"}]},
            {"attacks": [{"cost": [], "name": "A", "damage": "ten"}]},
            {"attacks": [{"cost": [], "name": "", "damage": "10"}]},
            {"attacks": [{"cost": ["Fire"] * 6, "name": "A", "damage": "10"}]},
            {"attacks": [{"cost": [], "name": "A"}] * 5},
            {"abilities": [{"name": "A", "effect": ""}]},
            {"weakness": {"type": "Fire", "value": "2"}},
            {"resistance": {"type": "Fire", "value": "+30"}},
            {"retreat": 6}, {"retreat": True}, {"hp": "185"}, {"name": ""},
        ]
        for edits in bad:
            with self.subTest(edits), self.assertRaises(ValueError):
                edit.apply(card(), edits)

    def test_card_ids(self):
        for ok in ("swsh6-99", "sv03.5-065", "sma-SV33", "swshp-SWSH055", "swsh12.5-065", "sm10-19"):
            self.assertTrue(edit.valid_card_id(ok), ok)
        for bad in ("../etc", "swsh6/99", "", None, "a-b-c", "x" * 40):
            self.assertFalse(edit.valid_card_id(bad), bad)


class ImportSet(unittest.TestCase):
    def test_sorts_cards(self):
        data = {"name": "Mine", "cards": [
            {"id": "sv09-080", "name": "Larvitar", "supertype": "Pokémon"},
            {"id": "swsh6-97", "name": "Galarian Moltres V", "supertype": "Pokémon"},
            {"id": "swsh3-88", "name": "Tyranitar", "supertype": "Pokémon"},
            {"id": "swsh3-88", "name": "Tyranitar", "supertype": "Pokémon"},
            {"id": "me01-114", "name": "Boss's Orders", "supertype": "Trainer"},
            {"id": "../x", "name": "bad"}]}
        out = gui.import_set(data)["cards"]
        self.assertEqual([(c["id"], c.get("layout"), c.get("skip")) for c in out],
                         [("sv09-080", "modern", None), ("swsh6-97", "ex", None), ("swsh3-88", "me", None),
                          ("me01-114", None, "not a Pokémon")])

    def test_rejects_other_json(self):
        with self.assertRaises(gui.UserError):
            gui.import_set([1, 2])


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

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_page_and_meta(self):
        status, ctype, body = self.call("/")
        self.assertEqual((status, ctype.split(";")[0]), (200, "text/html"))
        self.assertIn(b"Card Restyle", body)
        status, _, body = self.call("/api/meta")
        meta = json.loads(body)
        self.assertEqual(meta["energy"], list(edit.ENERGY))
        self.assertIn("+20", meta["weakness"])
        self.assertIn("-30", meta["resistance"])
        self.assertIn("foil", meta)

    def test_bad_requests(self):
        self.assertEqual(self.call("/api/card?id=../../etc/passwd")[0], 400)
        self.assertEqual(self.call("/api/icon/Fairy.png")[0], 400)
        self.assertEqual(self.call("/nope")[0], 404)
        self.assertEqual(self.call("/api/render", {"id": "swsh3-88", "style": "comic"})[0], 400)

    def test_holofoil_needs_an_extract(self):
        with mock.patch.object(gui.tcg, "TCGLIVE", None):
            status, _, body = self.call("/api/render", {"id": "swsh3-88", "foil": True})
        self.assertEqual(status, 400)
        self.assertIn("--tcglive", json.loads(body)["error"])

    def test_render_and_zip(self):
        gui._renders.clear()
        fake = gui.Render(b"\x89PNG fake", ["a warning"], "me")
        with mock.patch.object(gui, "render", return_value=fake) as r:
            status, _, body = self.call("/api/render", {"id": "swsh3-88", "edits": {"retreat": 2}})
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual((data["layout"], data["warnings"]), ("me", ["a warning"]))
            self.assertNotIn("foil", data)
            r.assert_called_once_with("swsh3-88", {"retreat": 2}, "hybrid", True, False)
            status, ctype, body = self.call("/api/zip", {"items": [{"id": "swsh3-88"}, {"id": "sm10-19"}]})
        self.assertEqual((status, ctype), (200, "application/zip"))
        self.assertEqual(zipfile.ZipFile(io.BytesIO(body)).namelist(), ["swsh3-88-me.png", "sm10-19-me.png"])

    def test_render_and_zip_with_holofoil(self):
        gui._renders.clear()
        record = {"base": None, "reverse": {"shader": "TPCi/Cards3D/HoloFoil/FlatSilver",
                                            "material": "MAT_Cards3D_FlatSilver"}}
        files = {"swsh3-51-me-front.png": b"f", "swsh3-51-me-wp-ph.png": b"p", "swsh3-51-me-foil.json": b"{}"}
        fake = gui.Render(b"\x89PNG fake", [], "me", (record, files, {"reverse": b"plate"}))
        with mock.patch.object(gui, "render", return_value=fake) as r:
            status, _, body = self.call("/api/render", {"id": "swsh3-51", "foil": True})
            data = json.loads(body)
            self.assertIsNone(data["foil"]["base"])
            self.assertEqual(data["foil"]["reverse"]["shader"], "FlatSilver")
            r.assert_called_once_with("swsh3-51", None, "hybrid", True, True)
            status, _, body = self.call("/api/zip", {"items": [{"id": "swsh3-51", "foil": True}]})
        self.assertEqual(zipfile.ZipFile(io.BytesIO(body)).namelist(), ["swsh3-51-me.png", *files])

    def test_edit_errors_come_back_as_400(self):
        with mock.patch.object(gui, "read", return_value=("me", card())):
            status, _, body = self.call("/api/render", {"id": "swsh3-88", "edits": {"retreat": 9}})
        self.assertEqual(status, 400)
        self.assertIn("retreat", json.loads(body)["error"])


if __name__ == "__main__":
    unittest.main()


class Projects(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(gui.cache, "CACHE", Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_save_list_open_delete(self):
        data = {"cards": [{"id": "swsh3-88", "name": "Tyranitar", "layout": "me"}],
                "edits": {"swsh3-88": {"hp": "190"}}, "current": "swsh3-88"}
        self.assertEqual(gui.list_projects(), {"projects": []})
        out = gui.save_project(" My Set (v2) ", data)
        self.assertEqual(out["name"], "My Set (v2)")
        listed = gui.list_projects()["projects"]
        self.assertEqual([(p["name"], p["cards"]) for p in listed], [("My Set (v2)", 1)])
        self.assertEqual(gui.load_project("My Set (v2)")["data"], data)
        gui.save_project("My Set (v2)", {**data, "cards": []})          # overwrites
        self.assertEqual(gui.list_projects()["projects"][0]["cards"], 0)
        self.assertEqual(gui.delete_project("My Set (v2)"), {"deleted": "My Set (v2)"})
        self.assertEqual(gui.list_projects(), {"projects": []})

    def test_rejections(self):
        for name in ("", "../x", "a/b", None, "x" * 61, " "):
            with self.subTest(name), self.assertRaises(gui.UserError):
                gui.save_project(name, {"cards": []})
        for data in (None, [], {"cards": "x"}, {"edits": []}):
            with self.subTest(data), self.assertRaises(gui.UserError):
                gui.save_project("ok", data)
        with self.assertRaises(gui.UserError):
            gui.load_project("missing")
        with mock.patch.object(gui, "MAX_PROJECT", 50), self.assertRaises(gui.UserError):
            gui.save_project("big", {"cards": [{"id": "x" * 100}]})

    def test_names_stay_inside_the_folder(self):
        self.assertEqual(gui.project_file("Base Set, 1999").parent, gui.projects_dir())
        self.assertEqual(gui.project_file("Base Set, 1999").name, "base-set-1999.json")
