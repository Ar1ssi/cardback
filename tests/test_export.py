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
