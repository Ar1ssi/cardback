"""Energy conversion: card text, kinds, frames, the editor's Energy fields, the set export and the set list. No network
or scans: TCGdex answers are mocked with the cards' own TCGdex text. The window and composing checks need Gill Sans
(SVRESTYLE_FONTS) and TCG Live scans (SVRESTYLE_TCGLIVE) and run only when both are set.
Run: python -m unittest discover tests"""
import os
import unittest
import warnings
from unittest import mock

import numpy as np

from svrestyle import edit, energy, export, gui


def tcgdex(**over):
    """A TCGdex Energy record, sm3.5-69 Double Colorless Energy by default."""
    base = {"id": "sm3.5-69", "localId": "69", "category": "Energy", "energyType": "Special",
            "name": "Double Colorless Energy", "rarity": "Uncommon", "set": {"id": "sm3.5"},
            "effect": "Double Colorless Energy provides ColorlessColorless Energy."}
    base.update(over)
    return base


SET = {"cardCount": {"official": 73}, "releaseDate": "2017-08-11", "abbreviation": {"official": "SLG"}}


def read(record, s=SET):
    with mock.patch.object(energy.tcg, "card", return_value=record), \
            mock.patch.object(energy.tcg, "set_info", return_value=s):
        return energy.read_card(record["id"])


class Reading(unittest.TestCase):
    def test_double_colorless_energy_is_reworded_and_priced(self):
        card = read(tcgdex())
        self.assertEqual(card["energy"], "special")
        self.assertEqual(card["effect"], "As long as this card is attached to a Pokémon, it provides {C}{C} Energy.")
        self.assertEqual(card["cost"], ["Colorless", "Colorless"])
        self.assertEqual((card["frame"], card["set_code"], card["number"], card["rarity"]),
                         ("regular", "SLG", "69/73", "uncommon"))

    def test_a_secret_rare_takes_the_full_art_frame(self):
        self.assertEqual(read(tcgdex(rarity="Secret Rare"))["frame"], "full")
        self.assertEqual(read(tcgdex(rarity="Rare Holo"))["frame"], "regular")

    def test_old_wording_of_what_a_card_provides(self):
        # swsh1-186 keeps the sentence; xy6-97 and base-era Energy open with "Attach" / "While in play"
        self.assertEqual(energy.reworded(["Attach Rainbow Energy to 1 of your Pokémon. While in play, Rainbow Energy "
                                          "provides every type of Energy."], "Rainbow Energy"),
                         ["As long as this card is attached to a Pokémon, it provides every type of Energy."])
        self.assertEqual(energy.reworded(["Hyper Energy provides {C}{C} Energy. (Has no effect other than providing "
                                          "Energy.)"], "Hyper Energy"),
                         ["As long as this card is attached to a Pokémon, it provides {C}{C} Energy."])
        paragraphs = ["You can attach this card to 1 of your Pokémon only if you discard another card.",
                      "As long as this card is attached to a Pokémon, it provides {C} Energy."]
        self.assertEqual(energy.reworded(paragraphs, "Boost Energy"), paragraphs)

    def test_what_a_card_provides(self):
        self.assertEqual(energy.provided("As long as this card is attached to a Pokémon, it provides {R}{L} Energy."),
                         ["Fire", "Lightning"])
        self.assertEqual(energy.provided("it provides {C}{C}{C}{C} Energy"), ["Colorless"] * 3)       # the bar holds three
        self.assertEqual(energy.provided("it provides every type of Energy"), [])

    def test_no_icons_found_is_said(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            card = read(tcgdex(effect="This card provides every type of Energy."))
        self.assertEqual(card["cost"], [])
        self.assertTrue(any("name bar prints no icons" in str(w.message) for w in caught))

    def test_basic_energy_is_the_types_own(self):
        for name in ("Grass Energy", "Basic Grass Energy"):
            card = read(tcgdex(id="base1-99", localId="99", name=name, energyType="Normal", effect=None, rarity="Common",
                               set={"id": "base1"}), {"cardCount": {"official": 102}, "releaseDate": "1999-01-09",
                                                      "abbreviation": {"official": "BS"}})
            self.assertEqual((card["energy"], card["frame"], card["type"], card["number"]),
                             ("basic", "basic", "Grass", "099/102"))
        self.assertEqual(energy.template_id(card), "mee-001")
        self.assertEqual(sorted(set(energy.BASIC_TEMPLATES.values())), [f"mee-00{i}" for i in range(1, 9)])
        # Fairy, which Mega Evolution does not print, is Psychic's made Fairy's (fairy.basic_energy)
        card = read(tcgdex(name="Fairy Energy", effect=None, energyType="Normal"))
        self.assertEqual((card["energy"], card["frame"], card["type"]), ("basic", "basic", "Fairy"))
        self.assertEqual(energy.template_id(card), energy.BASIC_TEMPLATES["Psychic"])

    def test_refused(self):
        with self.assertRaises(SystemExit) as e:
            read(tcgdex(category="Trainer"))
        self.assertIn("not an Energy card", str(e.exception))
        with self.assertRaises(SystemExit) as e:                       # a Basic Energy of no Mega Evolution type
            read(tcgdex(name="Dragon Energy", effect=None, energyType="Normal"))
        self.assertIn("these types", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            read(tcgdex(effect=None))
        self.assertIn("no text", str(e.exception))
        with self.assertRaises(SystemExit) as e:                       # TCG Pocket
            read(tcgdex(), {**SET, "serie": {"id": "tcgp"}})
        self.assertIn("Pocket", str(e.exception))


    def test_energy_in_the_look_already_is_kept_as_printed(self):
        special = read(tcgdex(id="me03-086", name="Growing Grass Energy", set={"id": "me03"}, rarity="Rare",
                              effect="As long as this card is attached to a Pokémon, it provides {G} Energy."))
        self.assertEqual((special["energy"], special["frame"], special["cost"]), ("special", "print", ["Grass"]))
        basic = read(tcgdex(id="sv01-257", name="Basic Lightning Energy", energyType="Normal", effect=None,
                            rarity="Hyper rare", set={"id": "sv01"}))
        self.assertEqual((basic["energy"], basic["frame"], basic["type"]), ("basic", "print", "Lightning"))
        for card in (special, basic):
            self.assertEqual(edit.apply(card, edit.editable(card)), card)
            with self.assertRaises(ValueError):
                edit.apply(card, {"name": "Other"})
        self.assertEqual(edit.editable(special), {"energy": "special", "frame": "print"})


class Editing(unittest.TestCase):
    def setUp(self):
        self.special = read(tcgdex())
        self.basic = read(tcgdex(id="base1-99", name="Grass Energy", energyType="Normal", effect=None,
                                 set={"id": "base1"}))

    def test_editable_fields(self):
        self.assertEqual(edit.editable(self.special),
                         {"name": "Double Colorless Energy", "illustrator": "", "energy": "special",
                          "effect": self.special["effect"], "cost": ["Colorless", "Colorless"], "frame": "regular"})
        self.assertEqual(edit.editable(self.basic), {"energy": "basic", "frame": "basic", "type": "Grass"})
        self.assertTrue(edit.is_energy(self.special) and not edit.is_trainer(self.special))

    def test_edits_are_checked(self):
        out = edit.apply(self.special, {"name": "Twin Energy", "illustrator": "Someone",
                                        "effect": "As long as this card is attached to a pokemon, it provides {C} energy.",
                                        "cost": ["Fire"], "frame": "full"})
        self.assertEqual((out["name"], out["illustrator"], out["cost"], out["frame"]),
                         ("Twin Energy", "Someone", ["Fire"], "full"))
        self.assertIn("Pokémon", out["effect"])
        self.assertEqual(self.special["name"], "Double Colorless Energy")        # a copy
        for bad in ({"name": ""}, {"cost": ["Light"]}, {"cost": ["Fire"] * 4}, {"frame": "basic"},
                    {"energy": "basic"}, {"effect": "x" * 900}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                edit.apply(self.special, bad)

    def test_what_the_editor_sends_back_is_accepted(self):
        self.assertEqual(edit.apply(self.special, edit.editable(self.special))["cost"], ["Colorless", "Colorless"])
        self.assertEqual(edit.apply(self.basic, edit.editable(self.basic)), self.basic)

    def test_a_basic_energy_has_nothing_to_edit(self):
        for bad in ({"name": "Other Energy"}, {"type": "Fire"}, {"energy": "special"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                edit.apply(self.basic, bad)

    def test_an_older_basic_energy_can_take_the_gold_frame(self):
        gold = edit.apply(self.basic, {**edit.editable(self.basic), "frame": "gold"})
        self.assertEqual((gold["frame"], gold["rarity_name"]), ("gold", "Hyper rare"))     # as the gold prints are
        self.assertEqual(export.card_name(gold, "energy"), "Basic Grass Energy")
        self.assertEqual(edit.apply(gold, {"frame": "basic"})["frame"], "basic")
        with self.assertRaises(ValueError):
            edit.apply(self.basic, {"frame": "full"})                   # a Special Energy's frame
        printed = dict(self.basic, frame="print")                      # a Scarlet & Violet print keeps its own
        with self.assertRaises(ValueError):
            edit.apply(printed, {"frame": "gold"})

    def test_a_one_type_special_energy_can_take_the_typed_frame(self):
        typed = edit.apply(self.special, {"cost": ["Fairy"], "frame": "typed"})
        self.assertEqual((typed["frame"], typed["rarity_name"], typed["rarity"]), ("typed", "Rare", "rare"))
        self.assertEqual(edit.apply(self.special, {"cost": ["Water", "Water"], "frame": "typed"})["frame"], "typed")
        for bad in ({"frame": "typed"}, {"cost": ["Fire", "Water"], "frame": "typed"},
                    {"cost": ["Dragon"], "frame": "typed"}):          # Colorless, two types, no typed print of it
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                edit.apply(self.special, bad)

    def test_a_typed_name_prints_its_type_as_the_symbol(self):
        self.assertEqual(energy.TYPE_WORD.fullmatch("Charming Fairy Energy").groups(), ("Charming", "Fairy", "Energy"))
        self.assertIsNone(energy.TYPE_WORD.fullmatch("Wonder Energy"))              # set as it is
        self.assertEqual(energy.typed_type({"cost": ["Fairy", "Fairy"]}), "Fairy")
        self.assertIsNone(energy.typed_type({"cost": ["Colorless", "Colorless"]}))

    def test_changes_are_listed(self):
        after = edit.apply(self.special, {"name": "Twin Energy", "cost": ["Fire", "Water"], "frame": "full"})
        self.assertEqual(edit.changes(self.special, after),
                         ["Name: Double Colorless Energy → Twin Energy", "Provides: {C}{C} → {R}{W}",
                          "Frame: regular → full"])
        self.assertEqual(edit.changes(self.basic, self.basic), [])


class SetsAndExport(unittest.TestCase):
    def test_a_special_energy_is_a_record_the_simulator_reads(self):
        card = read(tcgdex(rarity="Secret Rare"))
        rec = export.tcgdex_record(card, "energy", "customhb", "387", {"id": "sm3.5-69", "foil": True})
        self.assertEqual((rec["category"], rec["energyType"], rec["rarity"], rec["name"]),
                         ("Energy", "Special", "Secret Rare", "Double Colorless Energy"))
        self.assertEqual(rec["effect"], "As long as this card is attached to a Pokémon, it provides {C}{C} Energy.")
        self.assertEqual((rec["custom"]["layout"], rec["custom"]["foil"], rec["image"]),
                         ("energy", True, "/src/assets/custom-sets/customhb/387"))

    def test_a_basic_energy_is_a_normal_energy_record(self):
        card = read(tcgdex(id="sv01-257", name="Basic Lightning Energy", energyType="Normal", effect=None,
                           rarity="Hyper rare", set={"id": "sv01"}))
        rec = export.tcgdex_record(card, "energy", "customhb", "389", {"id": "sv01-257", "foil": True})
        self.assertEqual((rec["category"], rec["energyType"], rec["rarity"], rec["name"], "effect" in rec),
                         ("Energy", "Normal", "Hyper rare", "Basic Lightning Energy", False))

    def test_layout_of_and_the_set_list(self):
        with mock.patch.object(gui.tcg, "card", return_value=tcgdex()):
            self.assertEqual(gui.layout_of("sm3.5-69"), "energy")
        self.assertEqual(gui._set_entry("sm3.5-69", "Double Colorless Energy", "Energy"),
                         {"id": "sm3.5-69", "name": "Double Colorless Energy", "layout": "energy"})
        self.assertEqual(gui._set_entry("me02-124", "Ignition Energy", "Energy")["layout"], "energy")

    def test_the_editor_view_and_the_note(self):
        card = read(tcgdex())
        with mock.patch.object(gui, "read", return_value=("energy", card)), \
                mock.patch.object(gui.tcg, "card", return_value=tcgdex()):
            summary = gui.card_summary("sm3.5-69")
            self.assertEqual((summary["layout"], summary["energy"], summary["name"]),
                             ("energy", "special", "Double Colorless Energy"))
            name, notes = gui.card_changes("sm3.5-69", {"name": "Twin Energy"})
        self.assertEqual(notes, ["Restyled as a Mega Evolution Energy", "Name: Double Colorless Energy → Twin Energy"])


@unittest.skipUnless(os.environ.get("SVRESTYLE_FONTS") and os.environ.get("SVRESTYLE_TCGLIVE"),
                     "needs Gill Sans (SVRESTYLE_FONTS) and a TCG Live extract (SVRESTYLE_TCGLIVE)")
class Composing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from svrestyle import typeset
        typeset.configure(os.environ["SVRESTYLE_FONTS"])

    def test_an_old_window_is_found_on_the_scan(self):
        for card_id, (width, height) in (("sm3.5-69", (675, 528)), ("sm1-136", (675, 528)), ("swsh1-186", (675, 520))):
            x0, y0, x1, y1 = energy.old_window({"id": card_id, "frame": "regular"})
            with self.subTest(card_id=card_id):
                self.assertLess(abs((x1 - x0) - width), 12)
                self.assertLess(abs((y1 - y0) - height), 20)

    def test_a_frame_with_no_window_is_refused(self):
        with self.assertRaises(SystemExit):
            energy.old_window({"id": "base1-97", "frame": "regular"})

    def test_a_card_is_composed_with_its_foil(self):
        from svrestyle import layers
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with layers.recording() as drawn:
                img, report = energy.compose_card("sm2-166")
            built = energy.build_foil(report["card"], img, drawn)
        self.assertEqual(img.shape, (1024, 733, 4))
        self.assertTrue(built["base"] and built["base"][1].shape == (1024, 1024))
        self.assertIsNone(built["reverse"])
        plate = np.asarray(built["base"][1])
        self.assertEqual(int(plate[0, 0]), 0)               # outside the card: no foil

    def test_a_fairy_basic_energy_on_the_gold_frame(self):
        """sm1-172 Fairy Energy made the gold Hyper Rare one: Psychic's gold print (sv03.5-207) with Fairy's art and
        symbols, its gold as printed, its footer this card's, and the print's own foil (SvUltra, plate and etch)."""
        from svrestyle import layers, tcg
        card = edit.apply(energy.read_card("sm1-172"), {"frame": "gold"})
        with layers.recording() as drawn:
            img, _ = energy.compose_card("sm1-172", card=card)
        built = energy.build_foil(card, img, drawn)
        gold = tcg.scan("sv03.5-207")
        self.assertLess(float(np.abs(img[500, 12, :3] - gold[500, 12, :3]).max()), 2)    # the gold border
        self.assertLess(float(np.abs(img[60, 500, :3] - gold[60, 500, :3]).max()), 2)    # the gold title line
        self.assertGreater(float(np.abs(img[300, 100, :3] - gold[300, 100, :3]).max()), 20)    # the art: Fairy's
        spec, plate, extra = built["base"]
        self.assertEqual(spec[0], "TPCi/Cards3D/HoloFoil/SvUltra")
        self.assertIsNotNone(extra["etch"])
        self.assertIsNone(built["reverse"])
        stars = drawn["footer_stars"].max(axis=0) > 0.5
        number_end = np.flatnonzero(drawn["footer"].max(axis=0) > 0.5)[-1]
        self.assertTrue(0 < np.flatnonzero(stars)[0] - number_end < 14)      # the stars after the number

    def test_a_fairy_special_energy_on_the_typed_frame(self):
        """xy5-144 Wonder Energy on the typed frame: Telepathic Psychic Energy's print (me03-088) made Fairy's (its
        sphere with Fairy's symbol where the eye was), its frame as printed, the name's type set as its symbol between
        the words, and the print's own foil (SvHolo on the swirl)."""
        from svrestyle import layers, tcg
        card = edit.apply(energy.read_card("xy5-144"), {"name": "Charming Fairy Energy", "frame": "typed"})
        with layers.recording() as drawn:
            img, _ = energy.compose_card("xy5-144", card=card)
        built = energy.build_foil(card, img, drawn)
        psychic = tcg.scan("me03-088")
        self.assertLess(float(np.abs(img[30, 300, :3] - psychic[30, 300, :3]).max()), 2)     # the header
        r, g, b = img[300, 250, :3]                         # the eye's outline on the print, the sphere's pink here
        self.assertTrue(r > g + 60 and r > b and float(psychic[300, 250, :3].max()) < 20)
        cols = np.flatnonzero(drawn["name"][70:140].max(axis=0) > 0.5)
        groups = 1 + int((np.diff(cols) > energy.NAME_GAP).sum())
        self.assertEqual(groups, 3)                         # "Charming", the symbol, "Energy"
        spec, plate, _ = built["base"]
        self.assertEqual(spec[0], "TPCi/Cards3D/HoloFoil/SvHolo")
        self.assertIsNone(built["reverse"])
        self.assertGreater(int(np.asarray(plate)[200, 60 + tcg.BLEED_X]), 100)      # the swirl foils
        self.assertLess(int(np.asarray(plate)[397, 366 + tcg.BLEED_X]), 60)         # the sphere does not


if __name__ == "__main__":
    unittest.main()
