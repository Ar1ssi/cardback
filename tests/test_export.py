"""The ptcg-sim custom-set export (export.py) and its re-import. No network: TCGdex lookups are mocked."""
import datetime
import io
import json
import unittest
import zipfile
from unittest import mock

from PIL import Image

from svrestyle import edit, export, gui

SET_ID = "customhbw"


def pokemon(**over):
    base = {"id": "swsh3-88", "family": "regular", "type": "Fighting", "prefix": "", "name": "Tyranitar", "hp": "180",
            "stage": 2, "evolves_from": "Pupitar", "portrait": "", "illustrator": "x", "reg_mark": "D", "rarity": "rare",
            "flavor": "Its roar.", "abilities": [{"name": "Sand Veil", "effect": "Prevent all effects."}],
            "attacks": [{"cost": ["Fighting", "Colorless"], "name": "Bedrock Breaker", "damage": "120",
                         "effect": "Discard a Stadium in play."},
                        {"cost": ["Fighting"], "name": "Growl", "damage": "", "effect": ""}],
            "weakness": {"type": "Grass", "value": "×2"}, "resistance": None, "retreat": 4}
    base.update(over)
    return base


def trainer(**over):
    base = {"id": "base1-91", "kind": "Supporter", "printed_kind": "Item", "prefix": "", "name": "Bill", "character": "",
            "hp": "", "effect": "Draw 2 cards.\nThen, shuffle your deck.", "abilities": [], "attacks": [],
            "illustrator": "y", "reg_mark": "", "rarity": "common"}
    base.update(over)
    return base


def png_bytes(color=(200, 30, 30, 255)):
    buf = io.BytesIO()
    Image.new("RGBA", (733, 1024), color).save(buf, "PNG")
    return buf.getvalue()


class Records(unittest.TestCase):
    def test_pokemon_record_is_tcgdex_shaped(self):
        rec = export.tcgdex_record(pokemon(), "me", SET_ID, "001", {"id": "swsh3-88", "edits": {"hp": "180"}})
        self.assertEqual((rec["id"], rec["localId"], rec["category"], rec["name"]), ("customhbw-001", "001", "Pokemon", "Tyranitar"))
        self.assertEqual((rec["hp"], rec["types"], rec["stage"], rec["evolveFrom"]), (180, ["Fighting"], "Stage2", "Pupitar"))
        self.assertEqual(rec["attacks"][0], {"cost": ["Fighting", "Colorless"], "name": "Bedrock Breaker", "damage": 120,
                                             "effect": "Discard a Stadium in play."})
        self.assertEqual(rec["attacks"][1]["damage"], "")
        self.assertEqual(rec["abilities"], [{"type": "Ability", "name": "Sand Veil", "effect": "Prevent all effects."}])
        self.assertEqual((rec["weaknesses"], rec["resistances"], rec["retreat"]), ([{"type": "Grass", "value": "×2"}], [], 4))
        self.assertEqual((rec["regulationMark"], rec["rarity"], rec["description"]), ("D", "Rare", "Its roar."))
        self.assertEqual(rec["image"], "/src/assets/custom-sets/customhbw/001")
        self.assertEqual(rec["legal"], {"standard": False, "expanded": False})
        self.assertEqual(rec["custom"], {"source": "swsh3-88", "layout": "me", "edits": {"hp": "180"}, "style": "hybrid",
                                         "attacks": True, "foil": False})
        self.assertNotIn("suffix", rec)

    def test_ex_cards_get_their_suffix_and_plus_damage_stays_text(self):
        v = {k: v for k, v in pokemon(prefix="Galarian", name="Slowking", stage=0, evolves_from="").items() if k not in ("family", "rarity", "flavor")}
        v["attacks"][0]["damage"] = "90+"
        rec = export.tcgdex_record(v, "ex", SET_ID, "002", {"id": "swsh6-99", "style": "panel", "attacks": False, "foil": True})
        self.assertEqual((rec["name"], rec["suffix"], rec["stage"], rec["rarity"]), ("Galarian Slowking ex", "ex", "Basic", "Double rare"))
        self.assertEqual(rec["attacks"][0]["damage"], "90+")
        self.assertNotIn("evolveFrom", rec)
        self.assertEqual(rec["custom"]["style"], "panel")
        modern_ex = export.tcgdex_record(pokemon(family="ex"), "modern", SET_ID, "003", {"id": "sv03.5-065"})
        self.assertEqual(modern_ex["name"], "Tyranitar ex")
        regular_modern = export.tcgdex_record(pokemon(family="regular"), "modern", SET_ID, "004", {"id": "sv01-001"})
        self.assertEqual(regular_modern["name"], "Tyranitar")

    def test_mega_stage_follows_the_pre_evolution(self):
        mega = pokemon(stage=3, name="Tyranitar", evolves_from="Pupitar")
        with mock.patch.object(export.tcg, "search", return_value=[{"id": "swsh3-87"}]), \
                mock.patch.object(export.tcg, "card", return_value={"stage": "Stage1"}):
            self.assertEqual(export.tcgdex_record(mega, "me", SET_ID, "001", {"id": "x-1"})["stage"], "Stage2")
        with mock.patch.object(export.tcg, "search", return_value=[{"id": "a-1"}]), \
                mock.patch.object(export.tcg, "card", return_value={"stage": "Basic"}):
            self.assertEqual(export.mega_stage(mega), "Stage1")
        with mock.patch.object(export.tcg, "search", side_effect=OSError("offline")), self.assertWarns(UserWarning):
            self.assertEqual(export.mega_stage(mega), "Stage1")
        with mock.patch.object(export.tcg, "search", return_value=[]), self.assertWarns(UserWarning):
            self.assertEqual(export.mega_stage(mega), "Stage1")

    def test_trainer_record(self):
        rec = export.tcgdex_record(trainer(), "trainer", SET_ID, "005", {"id": "base1-91"})
        self.assertEqual((rec["category"], rec["trainerType"], rec["name"]), ("Trainer", "Supporter", "Bill"))
        self.assertEqual(rec["effect"], "Draw 2 cards.\n\nThen, shuffle your deck.")
        self.assertEqual(rec["rarity"], "Common")
        for key in ("hp", "attacks", "abilities", "types", "stage"):
            self.assertNotIn(key, rec)
        fossil = export.tcgdex_record(trainer(kind="Item", hp="50", abilities=[{"name": "Dig", "effect": "x"}]), "trainer",
                                      SET_ID, "006", {"id": "sm1-1"})
        self.assertEqual((fossil["trainerType"], fossil["hp"], fossil["abilities"][0]["type"]), ("Item", 50, "Ability"))

    def test_set_code_and_id(self):
        self.assertEqual(export.set_id_for("HBW"), "customhbw")
        self.assertEqual(export.set_id_for("A1"), "customa1")
        for bad in ("hbw", "H", "TOOLONG", "HB W", "", None, 12):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                export.set_id_for(bad)


class Zip(unittest.TestCase):
    def export(self, items, code="HBW"):
        reads = {"swsh3-88": ("me", pokemon()), "base1-91": ("trainer", trainer())}
        return export.export_zip(items, "Homebrew Wave 1", code, read=lambda i: reads[i],
                                 render=lambda *a: (png_bytes(), [], "me", None), apply=edit.apply,
                                 changelog=lambda items, title: f"# {title}", today=datetime.date(2026, 10, 23))

    def test_layout_and_set_json(self):
        data = self.export([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91"}])
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertEqual(sorted(z.namelist()), sorted([
            "customhbw/set.json", "customhbw/001/high.webp", "customhbw/001/low.webp",
            "customhbw/002/high.webp", "customhbw/002/low.webp", "customhbw/CHANGELOG.md"]))
        record = json.loads(z.read("customhbw/set.json"))
        self.assertEqual({k: record[k] for k in ("format", "version", "id", "code", "name", "releaseDate")},
                         {"format": "ptcg-sim.custom-set-data", "version": 1, "id": "customhbw", "code": "HBW",
                          "name": "Homebrew Wave 1", "releaseDate": "2026-10-23"})
        self.assertEqual([c["id"] for c in record["cards"]], ["customhbw-001", "customhbw-002"])
        self.assertEqual(record["cards"][0]["hp"], 190)                  # the edit applied
        self.assertEqual(record["cards"][0]["custom"]["edits"], {"hp": "190"})
        high = Image.open(io.BytesIO(z.read("customhbw/001/high.webp")))
        low = Image.open(io.BytesIO(z.read("customhbw/002/low.webp")))
        self.assertEqual((high.format, high.size, low.size), ("WEBP", (733, 1024), (245, 342)))
        self.assertTrue(z.read("customhbw/CHANGELOG.md").decode().startswith("# Homebrew Wave 1"))

    def test_rejections(self):
        with self.assertRaises(ValueError):
            self.export([], "HBW")
        with self.assertRaises(ValueError):
            self.export([{"id": "swsh3-88"}], "hbw")
        with self.assertRaises(ValueError):
            self.export(["swsh3-88"])

    def test_round_trip_import(self):
        data = self.export([{"id": "swsh3-88", "edits": {"hp": "190"}, "style": "panel", "attacks": False, "foil": True},
                            {"id": "base1-91"}])
        record = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("customhbw/set.json"))
        self.assertTrue(export.is_set_data(record))
        with mock.patch.object(gui.modern, "is_modern", return_value=False):
            back = gui.import_set(record)
        self.assertEqual((back["name"], back["code"], back["foil"]), ("Homebrew Wave 1", "HBW", True))
        self.assertEqual([(c["id"], c["layout"]) for c in back["cards"]], [("swsh3-88", "me"), ("base1-91", "trainer")])
        self.assertEqual(back["edits"], {"swsh3-88": {"hp": "190"}})
        self.assertEqual(back["options"], {"swsh3-88": {"style": "panel", "attacks": False},
                                           "base1-91": {"style": "hybrid", "attacks": True}})
        with self.assertRaises(gui.UserError):
            gui.import_set({**record, "version": 2})
        # the older Custom Set tab export still imports as before
        with mock.patch.object(gui.modern, "is_modern", return_value=False):
            old = gui.import_set({"format": "ptcg-sim.custom-set", "name": "Picks", "cards": [{"id": "swsh3-88", "name": "Tyranitar", "supertype": "Pokémon"}]})
        self.assertEqual([c["layout"] for c in old["cards"]], ["me"])
        self.assertNotIn("edits", old)


if __name__ == "__main__":
    unittest.main()


class Http(unittest.TestCase):
    """The two routes over HTTP: /api/export-set (rendering mocked) and /api/suggest (no key)."""
    @classmethod
    def setUpClass(cls):
        import threading
        cls.server = gui.make_server(0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, path, body):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_export_set(self):
        fake = gui.Render(png_bytes(), [], "me")
        with mock.patch.object(gui, "read", return_value=("me", pokemon())), \
                mock.patch.object(gui, "render", return_value=fake), \
                mock.patch.object(gui, "changelog", return_value="# log"):
            status, ctype, body = self.call("/api/export-set", {"items": [{"id": "swsh3-88"}], "title": "T", "code": "HBW"})
            self.assertEqual((status, ctype), (200, "application/zip"))
            names = zipfile.ZipFile(io.BytesIO(body)).namelist()
            self.assertEqual(sorted(names), ["customhbw/001/high.webp", "customhbw/001/low.webp", "customhbw/CHANGELOG.md",
                                             "customhbw/set.json"])
            status, _, body = self.call("/api/export-set", {"items": [{"id": "swsh3-88"}], "code": "bad code"})
            self.assertEqual(status, 400)
            self.assertIn("set code", json.loads(body)["error"])
            self.assertEqual(self.call("/api/export-set", {"items": [], "code": "HBW"})[0], 400)

    def test_suggest_without_a_key_or_cli_is_a_502_and_meta_says_so(self):
        import urllib.request
        from svrestyle import wording
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(wording.shutil, "which", return_value=None):
            status, _, body = self.call("/api/suggest", {"text": "flip coin", "kind": "attack"})
            self.assertEqual(status, 502)
            self.assertIn("ANTHROPIC_API_KEY", json.loads(body)["error"])
            with urllib.request.urlopen(self.base + "/api/meta", timeout=10) as r:
                meta = json.loads(r.read())
        self.assertFalse(meta["suggest"])
        self.assertIn("Active Pokémon", meta["terms"])
        self.assertEqual(self.call("/api/suggest", {"text": "", "kind": "attack"})[0], 400)

    def test_push_set_writes_into_the_simulator_folder(self):
        import tempfile
        import urllib.request
        from pathlib import Path
        fake = gui.Render(png_bytes(), [], "me")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gui, "read", return_value=("me", pokemon())), \
                mock.patch.object(gui, "render", return_value=fake), \
                mock.patch.object(gui, "changelog", return_value="# log"):
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": ""}):
                status, _, body = self.call("/api/push-set", {"items": [{"id": "swsh3-88"}], "code": "HBW"})
                self.assertEqual(status, 400)
                self.assertIn("SVRESTYLE_PTCG_SIM", json.loads(body)["error"])
            (Path(tmp) / "client" / "src").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp}):
                with urllib.request.urlopen(self.base + "/api/meta", timeout=10) as r:
                    self.assertTrue(json.loads(r.read())["ptcgSim"])
                status, _, body = self.call("/api/push-set", {"items": [{"id": "swsh3-88"}], "title": "T", "code": "HBW"})
                self.assertEqual(status, 200)
                self.assertEqual({k: v for k, v in json.loads(body).items() if k != "written"},
                                 {"setId": "customhbw", "rendered": 1, "reused": 0, "unchanged": 0, "removed": []})
                self.assertTrue((Path(tmp) / "client/src/assets/custom-sets/customhbw/001/high.webp").is_file())



class SimFolder(unittest.TestCase):
    def test_sim_sets_dir_takes_the_checkout_or_the_folder_itself(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sets = Path(tmp, "client", "src", "assets", "custom-sets")
            for value, want in (("", None), (str(Path(tmp, "missing")), None), (tmp, None)):
                with self.subTest(value=value), mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": value}):
                    self.assertEqual(gui.sim_sets_dir(), want)
            Path(tmp, "client", "src").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp}):
                self.assertEqual(gui.sim_sets_dir(), sets)     # made by the first push, not here
            sets.mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": str(sets)}):
                self.assertEqual(gui.sim_sets_dir(), sets)


class Push(unittest.TestCase):
    """write_set_dir: the set written into the simulator's folder, re-rendering only what changed."""
    COLORS = {"swsh3-88": (200, 30, 30, 255), "base1-91": (30, 30, 200, 255), "swsh3-89": (30, 200, 30, 255)}

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.set_dir = self.root / SET_ID
        self.renders = []

    def tearDown(self):
        self._tmp.cleanup()

    def push(self, items, force=False, title="Homebrew Wave 1"):
        reads = {"swsh3-88": ("me", pokemon()), "base1-91": ("trainer", trainer()),
                 "swsh3-89": ("me", pokemon(id="swsh3-89", name="Larvitar", stage=0, evolves_from=""))}

        def render(card_id, *rest):
            self.renders.append(card_id)
            return png_bytes(self.COLORS[card_id]), [], "me", None
        return export.write_set_dir(items, title, "HBW", self.root, read=lambda i: reads[i], render=render,
                                    apply=edit.apply, changelog=lambda items, title: f"# {title}", force=force,
                                    today=datetime.date(2026, 10, 25))

    def record(self):
        return json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))

    def color_of(self, local_id):
        """The card front's colour channel ("r", "g" or "b"): WebP is lossy, so exact pixels drift."""
        pixel = Image.open(self.set_dir / local_id / "high.webp").convert("RGB").getpixel((360, 500))
        return "rgb"[pixel.index(max(pixel))]

    def test_first_push_writes_every_file_and_a_second_changes_nothing(self):
        first = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((first["setId"], first["rendered"], first["reused"], first["removed"]), (SET_ID, 2, 0, []))
        self.assertEqual(first["written"][-1], "set.json")           # last: the simulator reloads on it
        self.assertEqual(sorted(first["written"]), sorted(["001/high.webp", "001/low.webp", "002/high.webp",
                                                           "002/low.webp", "CHANGELOG.md", "set.json"]))
        again = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((again["rendered"], again["reused"], again["written"], again["unchanged"]), (0, 2, [], 6))
        self.assertEqual(self.renders, ["swsh3-88", "base1-91"])
        self.assertEqual(list(self.set_dir.glob("**/.*.tmp")), [])

    def test_an_edit_re_renders_that_card_only(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.renders.clear()
        r = self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 1, ["swsh3-88"]))
        self.assertEqual(r["written"], ["set.json"])          # the mock renders the same pixels: equal WebPs stay
        self.assertEqual(self.record()["cards"][0]["hp"], 190)
        self.renders.clear()
        r = self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91", "style": "panel"}])
        self.assertEqual(self.renders, ["base1-91"])

    def test_reordering_moves_kept_cards_without_rendering(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.renders.clear()
        r = self.push([{"id": "base1-91"}, {"id": "swsh3-88"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (0, 2, []))
        cards = self.record()["cards"]
        self.assertEqual([(c["id"], c["custom"]["source"], c["image"]) for c in cards], [
            (f"{SET_ID}-001", "base1-91", f"/src/assets/custom-sets/{SET_ID}/001"),
            (f"{SET_ID}-002", "swsh3-88", f"/src/assets/custom-sets/{SET_ID}/002")])
        self.assertEqual((self.color_of("001"), self.color_of("002")), ("b", "r"))

    def test_a_shorter_list_removes_card_folders_and_keeps_other_files(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}, {"id": "swsh3-89"}])
        (self.set_dir / "foil.json").write_text("{}")
        r = self.push([{"id": "swsh3-89"}])
        self.assertEqual((r["rendered"], r["removed"]), (0, ["002", "003"]))
        self.assertEqual(sorted(p.name for p in self.set_dir.iterdir()), ["001", "CHANGELOG.md", "foil.json", "set.json"])
        self.assertEqual(self.color_of("001"), "g")

    def test_force_renders_every_card_and_a_broken_set_json_is_rebuilt(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.renders.clear()
        self.assertEqual(self.push([{"id": "swsh3-88"}, {"id": "base1-91"}], force=True)["rendered"], 2)
        (self.set_dir / "set.json").write_text("{not json")
        self.renders.clear()
        r = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], self.renders), (2, ["swsh3-88", "base1-91"]))
        self.assertEqual(self.record()["code"], "HBW")

    def test_rejects_like_the_zip(self):
        with self.assertRaises(ValueError):
            self.push([])
        with self.assertRaises(ValueError):
            export.write_set_dir([{"id": "swsh3-88"}], "", "hbw", self.root, read=None, render=None, apply=None,
                                 changelog=None)
        self.assertFalse(any(self.root.iterdir()))
