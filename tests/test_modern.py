"""Scarlet & Violet / Mega Evolution cards edited in place, and energy symbols in effect text: pure logic.
No network, model, fonts or extract. Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

from svrestyle import compose, edit, frame, modern
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

    def test_row_edits_touch_only_what_changes(self):
        fire = {"type": "Fire", "value": "×2"}
        # TCGdex leaves the printed weakness out (sv10.5b-046): it is looked for where the edit sets it
        gap, edited = card(weakness=None, retreat=3), card(weakness=fire, retreat=2)
        off, on = modern.row_off(gap, edited), modern.row_on(gap, edited)
        self.assertEqual((off["weakness"], off["resistance"], off["retreat_span"]), (fire, None, (2, 3)))
        self.assertEqual((on["weakness"], on["resistance"], on["retreat_span"]), (fire, None, (2, 2)))
        more = card(retreat=3)
        self.assertEqual(modern.row_off(card(), more)["retreat_span"], (1, 1))      # the printed circle stays
        self.assertEqual(modern.row_on(card(), more)["retreat_span"], (1, 3))
        self.assertIsNone(modern.row_on(card(), more)["weakness"])


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
                     {"types": ["Plasma"]}):
            with self.subTest(over), mock.patch.object(modern.tcg, "card", return_value=self.tcgdex(**over)):
                with self.assertRaises(SystemExit):
                    modern.read_card("sv05-104")

    def test_is_modern(self):
        self.assertTrue(modern.is_modern("sv03.5-065"))
        self.assertTrue(modern.is_modern("me01-054"))
        self.assertFalse(modern.is_modern("swsh6-99"))
        self.assertFalse(modern.is_modern(None))


def strokes(y0, y1, x0, x1, step=5):
    """Coverage like a line of glyphs: upright strokes 2 px wide, step px apart."""
    out = np.zeros((H, W), np.float32)
    for x in range(x0, x1 - 1, step):
        out[y0:y1, x:x + 2] = 1
    return out


def outline(y0, y1, x0, x1, t=3):
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = 1
    out[y0 + t:y1 - t, x0 + t:x1 - t] = 0
    return out


class MovesAsPrinted(unittest.TestCase):
    """The composers' layout of the old abilities and attacks moved onto the print, move by move: me02.5-152 prints
    its Ability 53 px above where the composers set it and its attack 23 px above."""
    LAYOUT = {"badge": outline(605, 640, 55, 229), "ability_name": strokes(612, 636, 240, 400),
              "text": np.maximum.reduce([strokes(655, 668, 50, 650), strokes(674, 687, 50, 500),
                                         strokes(736, 760, 160, 380), strokes(780, 793, 50, 450)]),
              "icon": outline(731, 765, 55, 89)}

    def printed(self, shifts):
        """A plain panel with the layout printed in dark ink, the Ability (above row 700) and the attack moved."""
        scan = np.full((H, W, 4), 230, np.float32)
        for kind, cover in self.LAYOUT.items():
            for (a, b), dy in zip(((0, 700), (700, H)), shifts):
                part = np.zeros((H, W), np.float32)
                part[a:b] = cover[a:b]
                ys, xs = np.nonzero(part > 0.5)
                scan[ys + dy, xs, :3] = 30
        return scan

    def moved(self, scan):
        with mock.patch.object(modern, "_drawn", return_value={k: v.copy() for k, v in self.LAYOUT.items()}):
            return modern.moves_as_printed(scan, card(family="ex"))

    @staticmethod
    def rows(cover):
        ys = np.flatnonzero((cover > 0.5).any(axis=1))
        return int(ys.min()), int(ys.max())

    def test_each_move_lands_on_its_print(self):
        out = self.moved(self.printed((-53, -23)))
        self.assertEqual(self.rows(out["badge"]), (552, 586))
        self.assertEqual(self.rows(out["ability_name"]), (559, 582))
        self.assertEqual(self.rows(out["icon"]), (708, 741))
        self.assertTrue(out["text"][602:615].any() and out["text"][757:770].any())     # body lines go with them
        self.assertFalse(out["text"][655:668].any())

    def test_a_print_set_where_the_composers_set_it_stays(self):
        out = self.moved(self.printed((0, 0)))
        for kind, cover in self.LAYOUT.items():
            np.testing.assert_array_equal(out[kind], cover)


class FindingPrint(unittest.TestCase):
    def test_ink_in_finds_text_not_panel(self):
        scan = np.full((H, W, 4), 200, np.float32)
        scan[600:615, 100:300, :3] = 20             # a line of dark text
        ink = modern.ink_in(scan, (36, 512, 697, 868))
        self.assertTrue(ink[607, 200])
        self.assertFalse(ink[700, 200])
        self.assertFalse(ink[607, 20])               # outside the zone

    def test_ink_near_takes_offset_print_whole_but_not_the_rule(self):
        scan = np.full((H, W, 4), 200, np.float32)
        scan[880:900, 350:362, :3] = 20             # a printed glyph, set further left than ours
        scan[870:872, 100:600, :3] = 20             # the bar's rule above it
        cover = np.zeros((H, W), np.float32)
        cover[880:900, 360:372] = 1                  # where the composer sets the glyph
        found = modern.ink_near(scan, cover, reach=4)
        self.assertTrue(found[890, 351])             # beyond the reach, but part of the glyph that reaches
        self.assertFalse(found[871, 120])            # the rule far from the glyph stays

    def test_ink_colour(self):
        scan = np.full((H, W, 4), 60, np.float32)
        where = np.zeros((H, W), bool)
        where[600:604, 100:200] = True
        scan[where, :3] = 250
        self.assertIs(modern.ink_colour(scan, where), compose.WHITE)
        scan[:, :, :3] = 230
        scan[where, :3] = 30
        self.assertFalse(np.array_equal(modern.ink_colour(scan, where), compose.WHITE))

    @staticmethod
    def text_on(panel, ink, outline=None):
        """A panel with two lines of glyph-sized marks, and a few art strokes of the other shade."""
        scan = np.full((H, W, 4), panel, np.float32)
        for base in (620, 660):
            for x in range(80, 600, 20):
                if outline is not None:
                    scan[base - 16:base + 2, x - 2:x + 6, :3] = outline
                scan[base - 14:base, x:x + 4, :3] = ink
        scan[720:724, 60:400, :3] = 255 - panel     # a long streak of art: not glyph-shaped
        return scan

    def test_print_ink_reads_dark_and_light_text(self):
        fill, edge = modern.print_ink(self.text_on(200, 30), modern.MOVES_ZONE["regular"])
        self.assertTrue(np.array_equal(fill, frame.DARK_INK))
        self.assertIsNone(edge)
        fill, edge = modern.print_ink(self.text_on(40, 250), modern.MOVES_ZONE["ex"])
        self.assertIs(fill, compose.WHITE)
        self.assertIsNone(edge)

    def test_print_ink_reads_a_keyline(self):
        fill, edge = modern.print_ink(self.text_on(120, 30, outline=250), modern.MOVES_ZONE["ex"])
        self.assertTrue(np.array_equal(fill, frame.DARK_INK))
        self.assertIs(edge[0], compose.WHITE)
        self.assertAlmostEqual(edge[1], 2.0, delta=0.3)             # the outline's own width: 2 px

    def test_print_ink_keeps_an_ink_it_does_not_know(self):
        fill, _ = modern.print_ink(self.text_on(220, (150, 20, 20)), modern.MOVES_ZONE["regular"])
        self.assertTrue(np.allclose(fill, (150, 20, 20), atol=12))

    def test_print_ink_finds_nothing_on_a_plain_panel(self):
        self.assertIsNone(modern.print_ink(np.full((H, W, 4), 200, np.float32), modern.MOVES_ZONE["regular"]))

    def test_fill_band_blends_along_rows(self):
        img = np.zeros((H, W, 4), np.float32)
        img[880:900, :300, :3] = 100
        img[880:900, 300:, :3] = 200
        where = np.zeros((H, W), bool)
        where[885:895, 280:320] = True
        img[where, :3] = 0                           # an icon
        out = modern.fill_band(img, where)
        self.assertTrue(100 <= out[890, 285, 0] < out[890, 315, 0] <= 200)
        self.assertEqual(out[870, 285, 0], 0)        # rows outside the gap untouched

    def test_fill_band_keeps_the_grain_and_the_rules(self):
        img = np.zeros((H, W, 4), np.float32)
        img[870:920, :, :3] = 150 + np.random.default_rng(1).normal(0, 12, (50, W, 1))   # the band's grain
        img[876:879, :, :3] = 220                    # a bar's rule
        where = np.zeros((H, W), bool)
        where[874:900, 300:330] = True               # an icon taken off, into the rule
        img[where, :3] = 0
        out = modern.fill_band(img, where)
        gap = out[882:898, 300:330, 0]
        self.assertGreater(float(gap.std()), 6)      # grain, not a flat blend
        self.assertLess(abs(float(gap.mean()) - 150), 8)
        self.assertTrue(np.all(np.abs(out[876:879, 300:330, 0] - 220) < 1))

    def test_fill_strip_rebuilds_the_strip_shading_not_the_ring(self):
        img = np.zeros((H, W, 4), np.float32)
        img[:95, :, :3] = (230, 120, 40)                             # the name bar
        img[116:, :, :3] = (40, 60, 160)                             # the art
        rows = np.linspace(255, 120, 21)[:, None, None]              # the strip's bands: light top, darker bottom
        cols = np.linspace(0, -12, W)[None, :, None]                  # a little darker along it
        img[95:116, :, :3] = rows + cols + np.random.default_rng(2).normal(0, 2, (21, W, 1))
        img[95:116, 128:136, :3] = 90                                # the ring's rim, left of the line
        where = np.zeros((H, W), bool)
        where[99:113, 140:300] = True                                # the old line
        img[where, :3] = 20
        strip = np.zeros((H, W), bool)
        strip[95:116, 128:520] = True
        with mock.patch.object(modern.stage, "strip_shape", return_value=strip):
            out = modern.fill_strip(img, where)
        want = np.broadcast_to(rows + cols, (21, W, 3))[4:18, 140:300]
        got = out[99:113, 140:300, :3]
        self.assertLess(float(np.abs(got - want).mean()), 4)
        self.assertLess(float(np.abs(got[:, :10] - want[:, :10]).mean()), 6)   # not the ring's grey at its left end
        self.assertTrue(np.array_equal(out[116:], img[116:]))        # the art below untouched

    def test_blot_on_plain_panel_is_patched_from_the_panel(self):
        art = np.full((H, W, 4), 200, np.float32)
        mask = np.zeros((H, W), bool)
        mask[580:660, 100:400] = True                # the taken-off area
        art[610:626, 240:256, :3] = 60               # a blot LaMa made up there
        blots = modern.ghosts(art, mask)
        self.assertTrue(blots[618, 248])
        self.assertFalse(blots[590, 120])
        out = modern.patch_blots(art, blots, mask)
        self.assertAlmostEqual(float(out[618, 248, 0]), 200, delta=2)

    def test_specks_are_left_as_grain(self):
        art = np.full((H, W, 4), 200, np.float32)
        mask = np.zeros((H, W), bool)
        mask[580:660, 100:400] = True
        art[618:621, 248:251, :3] = 60               # 9 px: under GHOST_MIN
        self.assertFalse(modern.ghosts(art, mask).any())

    def test_only_plain_panels_are_read_for_blots(self):
        def rarity(r):
            return mock.patch.object(modern.tcg, "card", return_value={"rarity": r})
        with rarity("Uncommon"):
            self.assertTrue(modern.plain_panel("sv04-128", card()))
            self.assertFalse(modern.plain_panel("sv08-076", card(family="ex")))
        with rarity("Special illustration rare"):
            self.assertFalse(modern.plain_panel("sv03.5-199", card()))


class EnergyInText(unittest.TestCase):
    def test_codes_become_marks(self):
        text, types = ts.energy_text("Move a Basic {G} Energy. Discard {R}{R}. Keep {Y}. Not {Z}.")
        self.assertEqual(types, ["Grass", "Fire", "Fire", "Fairy"])
        self.assertEqual(text.count(ts.ENERGY_MARK), 4)
        self.assertIn("{Z}", text)                    # no such type: left as written

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
