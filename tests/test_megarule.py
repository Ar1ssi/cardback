"""The Mega Evolution rule panel on Gen 9 cards: its outline, its two lines beside the badge, the badge rebuilt from
Mega ex prints, the drawing (ring and printed badge on top), the editor field. No network, model or fonts: scans,
the badge, the Stage sprite and the type are made up.
Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

from svrestyle import edit, layers, megarule, modern

W, H = 733, 1024
X0, Y0, X1, Y1 = megarule.BADGE
BAND = (232, 205, 226)          # a pastel: neither lettering test (white fill, saturated outline) takes it


def fake_width(text, *args, **kwargs):
    return 7.0 * len(text)


def no_text(items):
    return np.zeros((H, W), np.float32)


def print_scan(name_at, strip=True, seed=0):
    """A made-up Mega ex print: art noise, the badge (band, white first line, a saturated name at name_at), and the
    "Evolves from" strip under the badge's lobe."""
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 4), np.float32)
    img[..., :3] = rng.uniform(0, 255, (H, W, 3))
    img[..., 3] = 255
    img[95:140, 413:717, :3] = BAND
    for x in range(470, 680, 14):                                   # "The Mega-Evolved form of": the same on all
        img[103:114, x:x + 6, :3] = 255
    img[121:132, name_at:name_at + 40, :3] = (230, 40, 160)         # the name: another on each print
    if strip:
        img[96:117, 395:413, :3] = 200
    return img


def sprite():
    """A badge sprite over BADGE: band inside rows 95-139 from x 413, its first line's letters white."""
    out = np.zeros((Y1 - Y0, X1 - X0, 4), np.float32)
    out[95 - Y0:140 - Y0, 413 - X0:717 - X0, :3] = BAND
    out[103 - Y0:114 - Y0, 470 - X0:676 - X0, :3] = (200, 60, 170)          # the outline colour the panel takes
    out[95 - Y0:140 - Y0, 413 - X0:717 - X0, 3] = 255
    return out


class Outline(unittest.TestCase):
    def test_panel_hangs_under_the_strip_and_leans_on_the_right(self):
        a = megarule.coverage(megarule.outline(420.0))
        self.assertEqual(a[140, 300], 1.0)
        self.assertEqual(a[117, 300], 0.0)                          # the strip's row stays the strip's
        self.assertEqual(a[118, 300], 1.0)
        self.assertEqual((a[165, 300], a[166, 300], a[167, 300]), (1.0, 0.5, 0.0))     # bottom edge at 166.0
        self.assertEqual(a[140, 124], 0.0)                          # clear of the portrait ring's rim
        self.assertEqual(a[119, 126], 0.0)                          # the rounded top-left corner
        top_right = 420.0 + megarule.SLANT * (megarule.BOTTOM - megarule.TOP)
        self.assertEqual(a[119, int(top_right) - 3], 1.0)           # leaning like '/': wider at the top
        self.assertEqual(a[164, int(top_right) - 3], 0.0)
        edge = a[140, 430:445]
        self.assertTrue(((edge > 0) & (edge < 1)).any())            # anti-aliased

    def test_colours_run_along_the_panel(self):
        colours = np.zeros((W, 3), np.float32)
        colours[:, 0] = np.arange(W)
        laid = megarule.along(colours, megarule.outline(420.0))
        self.assertAlmostEqual(float(laid[int(megarule.LEFT), 0]), megarule.LETTER_X[0], delta=2)
        self.assertAlmostEqual(float(laid[470, 0]), megarule.LETTER_X[1], delta=1)


class Lines(unittest.TestCase):
    def test_first_line_fits_beside_the_badge(self):
        with mock.patch.object(megarule.ts, "text_width", side_effect=fake_width):
            first, second = megarule.split(250.0)
        self.assertLessEqual(fake_width(first), 250.0)
        self.assertGreater(fake_width(first + " " + second.split()[0]), 250.0)
        self.assertEqual(f"{first} {second}", megarule.RULE)

    def test_label_names_what_the_mega_evolved_from(self):
        self.assertEqual(megarule.label_of({"name": "Mega Alakazam", "stage": 3, "evolves_from": "Alakazam"}),
                         (["Alakazam"], True))
        self.assertEqual(megarule.label_of({"name": "Mega Lucario", "stage": 1, "evolves_from": "Riolu"}),
                         (["Lucario"], True))
        self.assertEqual(megarule.label_of({"name": "Slowking", "stage": 0}), (["Mega Evolution rule"], False))

    def test_mega_ex_prints_have_the_badge(self):
        self.assertTrue(megarule.prints_badge({"id": "me01-077", "family": "ex", "name": "Mega Lucario"}))
        self.assertTrue(megarule.prints_badge({"id": "mep-013", "family": "ex", "name": "Mega Venusaur"}))
        self.assertFalse(megarule.prints_badge({"id": "me01-056", "family": "regular", "name": "Alakazam"}))
        self.assertFalse(megarule.prints_badge({"id": "sv03-125", "family": "ex", "name": "Charizard"}))


class Badge(unittest.TestCase):
    def test_build_keeps_the_shared_line_and_takes_the_names_off(self):
        prints = {cid: print_scan(470 + 9 * i, strip=i % 5 != 0, seed=i) for i, cid in enumerate(megarule.SOURCES)}
        with mock.patch.object(megarule.tcg, "scan", side_effect=prints.__getitem__):
            badge = megarule.build_badge()
        self.assertEqual(badge.shape, (Y1 - Y0, X1 - X0, 4))
        alpha = badge[..., 3] / 255
        self.assertGreater(alpha[117 - Y0, 600 - X0], 0.99)
        self.assertGreater(alpha[126 - Y0, 520 - X0], 0.99)         # where the names disagree: enclosed, filled
        self.assertEqual(alpha[105 - Y0, 405 - X0], 0.0)            # the strip under the lobe is not the badge's
        self.assertEqual(alpha[92 - Y0, 600 - X0], 0.0)             # above it
        np.testing.assert_allclose(badge[126 - Y0, 520 - X0, :3], BAND, atol=2)    # the names: band again
        np.testing.assert_allclose(badge[108 - Y0, 472 - X0, :3], 255, atol=1)     # the shared line: kept

    def test_label_set_on_the_band(self):
        with mock.patch.object(megarule, "badge", return_value=sprite()), \
                mock.patch.object(megarule.ts, "text_width", side_effect=fake_width), \
                mock.patch.object(megarule.ts, "coverage", side_effect=no_text):
            rgb, alpha = megarule.rebuilt_badge({"name": "Slowking", "stage": 0})
        self.assertEqual(rgb.shape, (Y1 - Y0, X1 - X0, 3))
        np.testing.assert_allclose(rgb[108 - Y0, 500 - X0], BAND, atol=2)      # not a Mega: no "Mega-Evolved form"
        self.assertEqual(alpha[117 - Y0, 600 - X0], 1.0)

    def test_outline_colours_need_lettering(self):
        rgb = np.zeros((Y1 - Y0, X1 - X0, 3), np.float32) + BAND
        with self.assertRaisesRegex(ValueError, "no lettering"):
            megarule.outline_colours(rgb)
        rgb[110 - Y0, 500 - X0:502 - X0] = (200, 60, 170)
        rgb[111 - Y0, 500 - X0:502 - X0] = (200, 60, 170)
        np.testing.assert_allclose(megarule.outline_colours(rgb)[300], (200, 60, 170), atol=1)


class Drawing(unittest.TestCase):
    def draw(self, card, canvas=None, printed=False):
        canvas = np.zeros((H, W, 4), np.float32) if canvas is None else canvas
        ring = np.zeros((H, W, 4), np.float32)
        ring[..., 3] = 255
        with mock.patch.object(megarule, "badge", return_value=sprite()), \
                mock.patch.object(megarule.ts, "text_width", side_effect=fake_width), \
                mock.patch.object(megarule.ts, "coverage", side_effect=no_text), \
                mock.patch.object(megarule.stage, "sprite_of", return_value=ring), \
                layers.recording() as drawn:
            lines = megarule.draw(canvas, card, printed_badge=printed)
        return canvas, drawn, lines

    def test_panel_drawn_and_recorded(self):
        canvas, drawn, lines = self.draw({"name": "Mega Alakazam", "stage": 3, "evolves_from": "Alakazam"})
        self.assertGreater(canvas[140, 300, 0], 200)                 # pearl
        self.assertEqual(" ".join(lines), megarule.RULE)
        self.assertEqual(drawn[megarule.KIND][140, 300], 1.0)
        self.assertEqual(drawn[megarule.KIND][117, 600], 1.0)        # a rebuilt badge is drawn too
        self.assertEqual(canvas[200, 300, 0], 0.0)                   # the art below is left alone

    def test_portrait_ring_stays_on_top_of_a_stage_card(self):
        stage_card, _, _ = self.draw({"name": "Mega Alakazam", "stage": 3, "evolves_from": "Alakazam"})
        basic, _, _ = self.draw({"name": "Mega Kangaskhan", "stage": 0})
        self.assertEqual(stage_card[140, 128, 0], 0.0)               # under the ring: the ring's
        self.assertGreater(basic[140, 128, 0], 100)                  # no ring: the panel's

    def test_printed_badge_stays_on_top(self):
        canvas = np.zeros((H, W, 4), np.float32)
        canvas[95:140, 413:717, :3] = (240, 225, 190)                # the print's own badge (a pale gold band)
        canvas[103:114, 470:676, :3] = (200, 160, 20)                # its gold lettering
        canvas, drawn, _ = self.draw({"id": "me01-187", "family": "ex", "name": "Mega Gardevoir", "stage": 2},
                                     canvas, printed=True)
        np.testing.assert_allclose(canvas[130, 430, :3], (240, 225, 190), atol=1)
        self.assertEqual(drawn[megarule.KIND][117, 600], 0.0)        # the print's badge keeps its own foil
        self.assertEqual(drawn[megarule.KIND][125, 430], 0.0)        # the panel under it too
        rim = canvas[np.arange(118, 125), 300, :3]                   # the rim takes the badge's colours (gold)
        self.assertTrue(((rim[:, 0] - rim[:, 2]) > 40).any())


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
        self.assertEqual(edit.changes(card(), card(mega_rule=True)), ["Mega Evolution rule: added"])
        self.assertEqual(edit.changes(card(mega_rule=True), card()), ["Mega Evolution rule: taken off"])

    def test_in_place_the_rule_leaves_the_print_alone(self):
        self.assertEqual(modern.changed(card(), card(mega_rule=True)), ("rule",))
        self.assertEqual(modern.changed(card(), card(mega_rule=True, hp="350")), ("name", "rule"))
        self.assertEqual(modern.changed(card(), card()), ())


if __name__ == "__main__":
    unittest.main()
