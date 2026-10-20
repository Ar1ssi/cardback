"""Scarlet & Violet / Mega Evolution cards edited in place, and energy symbols in effect text: pure logic.
No network, model, fonts or extract. Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

from svrestyle import compose, edit, modern
from svrestyle import typeset as ts

W, H = 733, 1024


def card(**over):
    c = {"id": "sv09-080", "family": "regular", "type": "Fighting", "prefix": "", "name": "Larvitar", "hp": "70",
         "stage": 0, "abilities": [], "retreat": 1,
         "attacks": [{"cost": ["Colorless", "Colorless"], "name": "Crunch", "damage": "20", "effect": "Flip a coin."}],
         "weakness": {"type": "Grass", "value": "×2"}, "resistance": None}
    c.update(over)
    return c


class Groups(unittest.TestCase):
    def test_nothing_changed(self):
        self.assertEqual(modern.changed(card(), card()), ())

    def test_each_group(self):
        self.assertEqual(modern.changed(card(), card(hp="80")), ("name",))
        self.assertEqual(modern.changed(card(), card(retreat=2)), ("row",))
        moved = card(attacks=[{"cost": ["Fighting"], "name": "Crunch", "damage": "20", "effect": "Flip a coin."}])
        self.assertEqual(modern.changed(card(), moved), ("moves",))
        both = card(name="Pupitar", resistance={"type": "Fire", "value": "-30"})
        self.assertEqual(modern.changed(card(), both), ("name", "row"))

    def test_edits_round_trip_unchanged(self):
        c = card()
        self.assertEqual(modern.changed(c, edit.apply(c, edit.editable(c))), ())


class ReadCard(unittest.TestCase):
    def tcgdex(self, **over):
        c = {"category": "Pokemon", "name": "Gengar ex", "stage": "Stage2", "types": ["Darkness"], "hp": 310,
             "abilities": [{"name": "Gnawing Curse", "effect": "Put 2 damage counters on that Pokemon."}],
             "attacks": [], "weaknesses": [{"type": "Fighting", "value": "×2"}], "retreat": 2}
        c.update(over)
        return c

    def test_ex(self):
        with mock.patch.object(modern.tcg, "card", return_value=self.tcgdex()):
            c = modern.read_card("sv05-104")
        self.assertEqual((c["family"], c["name"], c["stage"], c["retreat"]), ("ex", "Gengar", 2, 2))
        self.assertEqual(c["abilities"][0]["effect"], "Put 2 damage counters on that Pokémon.")

    def test_refuses_what_it_cannot_edit(self):
        for over in ({"category": "Trainer"}, {"name": "Mewtwo V", "stage": "Basic"}, {"stage": "VSTAR"},
                     {"types": ["Fairy"]}):
            with self.subTest(over), mock.patch.object(modern.tcg, "card", return_value=self.tcgdex(**over)):
                with self.assertRaises(SystemExit):
                    modern.read_card("sv05-104")

    def test_is_modern(self):
        self.assertTrue(modern.is_modern("sv03.5-065"))
        self.assertTrue(modern.is_modern("me01-054"))
        self.assertFalse(modern.is_modern("swsh6-99"))
        self.assertFalse(modern.is_modern(None))


class FindingPrint(unittest.TestCase):
    def test_ink_in_finds_text_not_panel(self):
        scan = np.full((H, W, 4), 200, np.float32)
        scan[600:615, 100:300, :3] = 20             # a line of dark text
        ink = modern.ink_in(scan, (36, 512, 697, 868))
        self.assertTrue(ink[607, 200])
        self.assertFalse(ink[700, 200])
        self.assertFalse(ink[607, 20])               # outside the zone

    def test_ink_colour(self):
        scan = np.full((H, W, 4), 60, np.float32)
        where = np.zeros((H, W), bool)
        where[600:604, 100:200] = True
        scan[where, :3] = 250
        self.assertIs(modern.ink_colour(scan, where), compose.WHITE)
        scan[:, :, :3] = 230
        scan[where, :3] = 30
        self.assertFalse(np.array_equal(modern.ink_colour(scan, where), compose.WHITE))

    def test_fill_band_blends_along_rows(self):
        img = np.zeros((H, W, 4), np.float32)
        img[880:900, :300, :3] = 100
        img[880:900, 300:, :3] = 200
        where = np.zeros((H, W), bool)
        where[885:895, 280:320] = True
        img[where, :3] = 0                           # an icon
        out = modern.fill_band(img, where, np.zeros((H, W), bool))
        self.assertTrue(100 <= out[890, 285, 0] < out[890, 315, 0] <= 200)
        self.assertEqual(out[870, 285, 0], 0)        # rows outside the gap untouched


class EnergyInText(unittest.TestCase):
    def test_codes_become_marks(self):
        text, types = ts.energy_text("Move a Basic {G} Energy. Discard {R}{R}. Keep {Y}.")
        self.assertEqual(types, ["Grass", "Fire", "Fire"])
        self.assertEqual(text.count(ts.ENERGY_MARK), 3)
        self.assertIn("{Y}", text)                    # no Fairy symbol: left as written

    def test_marks_take_room_and_report_where(self):
        text, _ = ts.energy_text("a {G} b")
        ts.font.cache_clear()
        with mock.patch.object(ts, "font") as font:
            font.return_value.getlength = lambda s: len(s) * 10 * ts.SS
            marks = []
            items = ts.line_items([(text, ts.MEDIUM)], 0.0, 20.0, 100, 500, marks)
        self.assertEqual([i["text"] for i in items], ["a ", " b"])
        advance = ts.ENERGY_ADVANCE * 20.0
        self.assertAlmostEqual(marks[0], 100 + 20 + advance / 2)
        self.assertAlmostEqual(items[1]["x"], 100 + 20 + advance)


if __name__ == "__main__":
    unittest.main()
