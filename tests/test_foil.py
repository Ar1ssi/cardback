"""Foil textures: pure logic (layer recording, texture layout, plate filling). No network, model, fonts or extract.
Run: python -m unittest discover tests"""
import unittest

import numpy as np

from svrestyle import foil, layers, tcg
from svrestyle.imaging import put_sprite
from svrestyle import typeset as ts

W, H = 733, 1024


class Recording(unittest.TestCase):
    def test_nothing_kept_outside_recording(self):
        layers.record(np.ones((4, 4), np.float32))      # must not raise or leak into a later recording
        with layers.recording() as drawn:
            pass
        self.assertEqual(drawn, {})

    def test_innermost_kind_wins(self):
        canvas = np.zeros((H, W, 4), np.float32)
        cov = np.zeros((H, W), np.float32)
        cov[10:20, 10:20] = 1
        with layers.recording() as drawn:
            with layers.kind("footer"):
                ts.stamp(canvas, cov, (255, 255, 255))
                with layers.kind("icon"):
                    sprite = (np.full((5, 5, 3), 200, np.float32), np.ones((5, 5), np.float32), 2, 2, (0, 0))
                    put_sprite(canvas, sprite, 102, 302)
        self.assertEqual(drawn["footer"][15, 15], 1)
        self.assertEqual(drawn["footer"][302, 102], 0)
        self.assertGreater(drawn["icon"][302, 102], 0.9)

    def test_sprite_clipped_at_the_edge(self):
        with layers.recording() as drawn:
            layers.record(np.ones((10, 10), np.float32), W - 3, H - 3)
        self.assertEqual(drawn["other"][H - 1, W - 1], 1)


class Textures(unittest.TestCase):
    def test_front_is_square_with_the_card_at_the_bleed_offset(self):
        canvas = np.zeros((H, W, 4), np.float32)
        canvas[:, 0, :3] = 10
        canvas[:, -1, :3] = 20
        front = foil.front_texture(canvas)
        self.assertEqual(front.shape, (1024, 1024, 3))
        self.assertEqual(front[500, tcg.BLEED_X, 0], 10)
        self.assertEqual(front[500, 0, 0], 10)            # bleed repeats the edge
        self.assertEqual(front[500, 1023, 0], 20)

    def test_plate_bleed_is_no_foil(self):
        plate = foil.plate_texture(np.full((H, W), 255, np.float32))
        self.assertEqual(plate.shape, (1024, 1024))
        self.assertEqual(plate[0, 0], 0)
        self.assertEqual(plate[0, tcg.BLEED_X], 255)
        self.assertEqual(plate[0, tcg.BLEED_X + W], 0)


class Printings(unittest.TestCase):
    def test_non_foil_names_old_and_new(self):
        self.assertFalse(foil.is_foil({"_f": "NonFoil", "_s": "TPCi/Cards3D/Standard/NonFoil"}))
        self.assertFalse(foil.is_foil({"_f": "Standard_NonFoil_J", "_s": "Cards/Standard/Standard_NonFoil_J"}))
        self.assertTrue(foil.is_foil({"_f": "HoloFoil_SWHolo_J", "_s": "Cards/Foil/HoloFoil_SWHolo_J"}))
        self.assertFalse(foil.is_foil(None))


class Plates(unittest.TestCase):
    def test_cut_blacks_out_only_listed_kinds(self):
        drawn = {"icon": np.zeros((H, W), np.float32), "text": np.ones((H, W), np.float32)}
        drawn["icon"][5, 5] = 1
        out = foil.cut(np.full((H, W), 200, np.float32), drawn, ("icon",))
        self.assertEqual(out[5, 5], 0)
        self.assertEqual(out[6, 6], 200)

    def test_fill_rows_follows_horizontal_lines(self):
        img = np.full((H, W), 255, np.float32)
        img[100:104] = 20                       # a dark line across
        img[95:110, 300:310] = 0                # an icon on it
        where = np.zeros((H, W), bool)
        where[95:110, 300:310] = True
        out = foil.fill_rows(img, where)
        self.assertEqual(out[102, 305], 20)     # the line goes on through the icon
        self.assertEqual(out[97, 305], 255)


class OwnFoil(unittest.TestCase):
    """A card's own plate, edited: its foil field and how its plate prints each kind of ink."""
    @staticmethod
    def pattern():
        yy, xx = np.mgrid[:H, :W]
        return (150 + 60 * np.sin(xx / 9.0) * np.cos(yy / 13.0)).astype(np.float32)

    @staticmethod
    def word(y0=60, x0=140):
        cover = np.zeros((H, W), np.float32)
        for x in range(x0, x0 + 200, 9):           # letter strokes
            cover[y0:y0 + 30, x:x + 4] = 1
        return cover

    def test_ink_the_plate_runs_through_is_not_shown(self):
        cover = self.word()
        self.assertIsNone(foil.ink_value(self.pattern(), cover, cover > 0.05))

    def test_ink_the_plate_cuts_out_is_shown(self):
        cover = self.word()
        img = self.pattern() * (1 - cover)
        value, ground = foil.ink_value(img, cover, cover > 0.05)
        self.assertLess(value, 5)
        self.assertGreater(ground, 100)

    def test_lit_letters_on_a_dark_bar(self):
        cover = self.word()
        img = np.full((H, W), 40, np.float32) * (1 - cover) + 210 * cover
        value, ground = foil.ink_value(img, cover, cover > 0.05)
        self.assertAlmostEqual(value, 210, delta=2)
        self.assertAlmostEqual(ground, 40, delta=2)

    def test_transfer_fills_a_gap_from_a_peer_toned_to_the_card(self):
        truth = self.pattern()
        peer = 0.5 * truth + 20                     # same shapes, another card's tone
        gap = self.word(300, 200) > 0
        guess, have = foil.transfer(truth, ~gap, [(peer, np.ones((H, W), bool))], 16)
        self.assertTrue(have[gap].all())
        self.assertLess(np.abs(guess - truth)[gap].mean(), 2)

    def test_bridge_has_no_streaks(self):
        img = np.full((H, W), 200, np.float32)
        img[:, :W // 2] = 100
        gap = np.zeros((H, W), bool)
        gap[200:600, 100:600] = True
        out = foil.bridge(img, gap)
        row = out[400, 100:600]
        self.assertTrue((np.diff(row) >= -0.01).all())    # dark to light, smoothly, left to right
        self.assertTrue(((out >= 99.9) & (out <= 200.1)).all())

    def test_restyled_reverse_takes_the_cards_own_foil_where_it_has_foil(self):
        from unittest import mock
        template = np.full((H, W), 255, np.float32)
        template[100:500, 40:690] = 0               # the new art window
        own_foil = np.full((H, W), 120, np.float32)
        own = np.zeros((H, W), np.float32)
        own[:, 300:] = 1                            # the old card has foil on its right half only
        with mock.patch.object(foil, "me_reverse_template", return_value=template),                 mock.patch.object(foil, "printings", return_value={"reverse": {}}),                 mock.patch.object(foil, "field", return_value=(own_foil, own)):
            out = foil.me_reverse_plate("x", {"stage": 0}, {}, np.full((H, W), 255, np.float32))
        self.assertEqual(out[700, 500], 120)         # own foil
        self.assertEqual(out[700, 100], 255)         # none there: the template's
        self.assertEqual(out[300, 500], 0)           # art window


class InPlace(unittest.TestCase):
    def test_moves_and_weakness_row_keep_the_printed_plate(self):
        from unittest import mock
        card = {"id": "me02-013", "family": "ex", "type": "Fire", "prefix": "", "name": "Mega Charizard X", "hp": "250",
                "stage": 2, "evolves_from": "Charmeleon", "portrait": "", "abilities": [], "retreat": 2,
                "attacks": [{"cost": ["Fire"], "name": "Inferno X", "damage": "90×", "effect": ""}],
                "weakness": {"type": "Water", "value": "×2"}, "resistance": None}
        edited = dict(card, retreat=1, attacks=[dict(card["attacks"][0], damage="100×")])
        printed = np.full((H, W), 180, np.float32)
        drawn = {k: np.ones((H, W), np.float32) for k in ("text", "icon", "row_icon", "row_text")}
        canvas = np.full((H, W, 4), 255, np.float32)
        with mock.patch.object(foil, "printings", return_value={"base": {"_f": "SvUltra", "_s": "x"}, "reverse": None}),                 mock.patch.object(foil, "plate", return_value=printed),                 mock.patch.object(foil, "modern_front", return_value=None),                 mock.patch.object(foil, "texture", return_value=None):
            out = foil.build_modern("me02-013", edited, canvas, drawn, card)
        np.testing.assert_array_equal(foil.card_space(out["base"][1]), printed)


if __name__ == "__main__":
    unittest.main()
