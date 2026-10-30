"""The XY Mega Evolution rule box on Gen 9 cards: the sprite's build, where it hangs, the text under it, the editor
field. No network, model or fonts: scans and the sprite are made up.
Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

from svrestyle import compose, edit, layers, mecompose, modern, xyrule

W, H = 733, 1024
PITCH = 29


def laid(*blocks):
    """'A3' = ability with 3 lines, 'E2' = attack with 2 lines of text, 'n' = attack without text."""
    return [("ability" if b[0] == "A" else "attack", {}, [None] * (int(b[1:]) if len(b) > 1 else 0)) for b in blocks]


def scan_with_tab(dy=0, art=0.0, tab=True, seed=0):
    """A made-up XY print: grey box rows, a tab pattern where REF has it (moved down dy), art noise around."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 4), 255, np.float32)
    img[..., :3] = rng.uniform(0, 255, (H, W, 3)) if art else 128
    x0, y0, x1, y1 = xyrule.CROP
    img[y0 + dy:y1 + dy, x0:x1, :3] = 150                         # the box's silver
    if tab:
        tx0, ty0, tx1, ty1 = xyrule.TAB
        stripes = (np.arange(tx1 - tx0) // 7 % 2 * 200 + 20)[None, :, None]
        img[ty0 + dy:ty1 + dy, tx0:tx1, :3] = stripes
    return img


class Sprite(unittest.TestCase):
    def test_outline_is_solid_inside_and_clear_outside(self):
        a = xyrule.outline_alpha()
        x0, y0, x1, y1 = xyrule.CROP
        self.assertEqual(a.shape, (y1 - y0, x1 - x0))
        self.assertEqual(a[640 - y0, 300 - x0], 1.0)                 # the gold panel
        self.assertEqual(a[600 - y0, 300 - x0], 0.0)                 # above the top rim
        self.assertEqual(a[686 - y0, 690 - x0], 0.0)                 # past the slanted right end
        self.assertEqual(a[686 - y0, 40 - x0], 0.0)                  # round the left end
        self.assertEqual(a[602 - y0, 400 - x0], 1.0)                 # top and bottom lie on pixel edges (601.5, 689.5)
        self.assertEqual(a[601 - y0, 400 - x0], 0.0)
        self.assertEqual((a[689 - y0, 300 - x0], a[690 - y0, 300 - x0]), (1.0, 0.0))
        slant = a[645 - y0, 645 - x0:665 - x0]                       # the right end crosses x 654 at y 645
        self.assertTrue(((slant > 0) & (slant < 1)).any(), slant)   # anti-aliased

    def test_build_moves_each_print_onto_the_reference_and_takes_the_median(self):
        prints = {cid: scan_with_tab(dy=(23 if i % 3 == 1 else 0), art=1, seed=i)
                  for i, cid in enumerate(xyrule.SOURCES)}
        prints[xyrule.SOURCES[2]] = scan_with_tab(tab=False, art=1, seed=99)     # art over the tab: left out
        with mock.patch.object(xyrule.tcg, "web_scan", side_effect=prints.__getitem__):
            box = xyrule.build()
        x0, y0, x1, y1 = xyrule.CROP
        self.assertEqual(box.shape, (y1 - y0, x1 - x0, 4))
        self.assertAlmostEqual(float(box[660 - y0, 600 - x0, 0]), 150, delta=1)  # silver, whatever the art
        self.assertAlmostEqual(float(box[620 - y0, 200 - x0, 3]), 255, delta=1)

    def test_build_refuses_when_too_few_prints_show_the_box(self):
        prints = {cid: scan_with_tab(tab=cid == xyrule.REF) for cid in xyrule.SOURCES}
        with mock.patch.object(xyrule.tcg, "web_scan", side_effect=prints.__getitem__):
            with self.assertRaisesRegex(ValueError, "show the rule box"):
                xyrule.build()


class Placement(unittest.TestCase):
    def test_box_hangs_from_the_strip_that_closes_the_art_window(self):
        self.assertEqual(xyrule.shift("regular"), -90)               # Pokedex strip ends at 511.5 (me01-056)
        self.assertEqual(xyrule.shift("ex"), -91)                    # window bar ends at 510.5 (me01-077)
        self.assertEqual(xyrule.floor("regular"), 689.5 - 90 + 20)
        self.assertEqual(xyrule.floor("ex"), 689.5 - 91 + 20)

    def test_draw_pastes_and_records_the_box(self):
        x0, y0, x1, y1 = xyrule.CROP
        box = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        box[..., 0] = 200
        box[..., 3] = 255 * xyrule.outline_alpha()
        canvas = np.zeros((H, W, 4), np.float32)
        with mock.patch.object(xyrule, "sprite", return_value=box), layers.recording() as drawn:
            xyrule.draw(canvas, "ex")
        top = int(xyrule.TOP) + 1 + xyrule.shift("ex")                # first whole row of the box
        self.assertEqual(canvas[top, 300, 0], 200)
        self.assertEqual(canvas[top - 2, 300, 0], 0)                  # the window bar above is left alone
        self.assertEqual(drawn[xyrule.KIND][top, 300], 1.0)
        self.assertEqual(drawn[xyrule.KIND][top, 700], 0.0)


class TextUnder(unittest.TestCase):
    def test_ex_blocks_start_under_the_box(self):
        free, _ = compose.sv_ex_layout(laid("E2", "E1"), PITCH)
        floor = xyrule.floor("ex") + compose.CAP_UNDER_ROW
        under, fits = compose.sv_ex_layout(laid("E2", "E1"), PITCH, floor=floor)
        self.assertLess(free[0], floor)
        self.assertEqual(under[0], np.ceil(floor))
        self.assertEqual(under[1] - under[0], free[1] - free[0])    # the stack keeps its gaps
        self.assertTrue(fits)

    def test_ex_blocks_lower_than_the_box_stay(self):
        free, _ = compose.sv_ex_layout(laid("n"), PITCH)
        self.assertEqual(compose.sv_ex_layout(laid("n"), PITCH, floor=600)[0], free)

    def test_me_blocks_start_under_the_box(self):
        floor = xyrule.floor("regular")
        caps, fits = mecompose.me_layout(laid("A2", "E1"), PITCH, floor=floor)
        self.assertEqual(caps[0] - 5, np.ceil(floor))                # the badge top, under the box
        self.assertTrue(fits)
        caps, fits = mecompose.me_layout(laid("A3", "E2"), PITCH, floor=floor)    # Alakazam: set smaller to fit
        self.assertEqual(caps[0] - 5, np.ceil(floor))
        self.assertFalse(fits)

    def test_layouts_follow_the_card(self):
        self.assertIs(compose.ex_layout({}), compose.sv_ex_layout)
        self.assertIs(mecompose.rule_layout({"mega_rule": False}), mecompose.me_layout)
        self.assertEqual(compose.ex_layout({"mega_rule": True}).keywords["floor"],
                         xyrule.floor("ex") + compose.CAP_UNDER_ROW)
        self.assertEqual(mecompose.rule_layout({"mega_rule": True}).keywords["floor"], xyrule.floor("regular"))


def card(**over):
    c = {"id": "me01-077", "family": "ex", "type": "Fighting", "prefix": "", "name": "Mega Lucario", "hp": "340",
         "stage": 1, "evolves_from": "Riolu", "portrait": "", "abilities": [], "retreat": 2,
         "attacks": [{"cost": ["Fighting"], "name": "Aura Jab", "damage": "130", "effect": ""}],
         "weakness": {"type": "Psychic", "value": "×2"}, "resistance": None}
    c.update(over)
    return c


class Editing(unittest.TestCase):
    def test_field_round_trips_off_by_default(self):
        c = card()
        self.assertIs(edit.editable(c)["mega_rule"], False)
        self.assertIs(edit.apply(c, edit.editable(c))["mega_rule"], False)
        self.assertIs(edit.apply(c, {"mega_rule": True})["mega_rule"], True)

    def test_field_takes_only_true_or_false(self):
        with self.assertRaisesRegex(ValueError, "mega rule"):
            edit.apply(card(), {"mega_rule": "yes"})

    def test_change_is_listed(self):
        self.assertEqual(edit.changes(card(), card(mega_rule=True)), ["Mega Evolution rule (XY): added"])
        self.assertEqual(edit.changes(card(mega_rule=True), card()), ["Mega Evolution rule (XY): taken off"])

    def test_in_place_the_box_moves_the_abilities_and_attacks(self):
        self.assertEqual(modern.changed(card(), card(mega_rule=True)), ("moves", "rule"))
        self.assertEqual(modern.changed(card(), card(mega_rule=True, hp="350")), ("name", "moves", "rule"))
        self.assertEqual(modern.changed(card(), card()), ())


if __name__ == "__main__":
    unittest.main()
