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

    def test_restyled_reverse_is_the_plain_mega_evolution_plate(self):
        """A restyled card's reverse holo is the Mega Evolution template, whatever its old reverse repeated (xy6-43
        Togepi's Fairy symbols, swsh6-96 Galarian Weezing's chevrons) and whatever type it is made."""
        from unittest import mock
        template = np.full((H, W), 200, np.float32)
        template[100:500, 40:690] = 0               # the new art window
        alpha = np.full((H, W), 255, np.float32)
        with mock.patch.object(foil, "me_reverse_template", return_value=template),                 mock.patch.object(foil, "field", side_effect=AssertionError("the old reverse is not read")):
            psychic = foil.me_reverse_plate({"stage": 0, "type": "Psychic"}, {}, alpha)
            fairy = foil.me_reverse_plate({"stage": 0, "type": "Fairy"}, {}, alpha)
        np.testing.assert_array_equal(psychic, template)
        np.testing.assert_array_equal(fairy, template)


class Gen8Reverse(unittest.TestCase):
    """The Gen 8 reverse holo: the type's chevrons where the card's own reverse plate shows foil."""
    @staticmethod
    def own():
        img = np.full((H, W), 25, np.float32)       # a Scarlet & Violet plate's dark ground
        img[30:1000, 30:700] = 70                   # its pattern's darkest
        img[600:900, 300:700] = 235
        img[100:500, 40:690] = 0                    # the art window
        img[650:670, 60:80] = 0                     # an energy icon
        return img

    def test_the_pattern_shows_where_the_own_plate_shows_foil(self):
        from unittest import mock
        pattern = np.full((H, W), 180, np.float32)
        with mock.patch.object(foil, "gen8_pattern", return_value=pattern) as made:
            out = foil.gen8_plate("Grass", self.own())
        made.assert_called_once_with("Grass")
        self.assertEqual(out[10, 10], 0)            # the ground: no foil
        self.assertEqual(out[300, 300], 0)          # the art window
        self.assertEqual(out[660, 70], 0)           # the icon
        self.assertEqual(out[700, 200], 180)        # the pattern's darkest still all foil: its own pattern goes
        self.assertEqual(out[700, 400], 180)

    def test_only_the_generic_reverse_is_replaced(self):
        self.assertTrue(foil.generic_reverse({"_s": "TPCi/Cards3D/HoloFoil/FlatSilver", "_f": "FlatSilver"}))
        self.assertTrue(foil.generic_reverse({"_s": "Cards/Foil/HoloFoil_FlatSilver_J"}))
        self.assertFalse(foil.generic_reverse({"_s": "TPCi/Cards3D/HoloFoil/SvHolo"}))
        self.assertFalse(foil.generic_reverse({"_s": "TPCi/Cards3D/HoloFoil/FlatSilver", "_e": "sv3_e_085"}))

    def test_a_restyled_card_gets_it_whether_its_old_print_had_a_reverse_or_not(self):
        from unittest import mock
        canvas = np.full((H, W, 4), 255, np.float32)
        flat = np.full((H, W), 235, np.float32)
        flat[100:500, 40:690] = 0
        pattern = np.full((H, W), 140, np.float32)
        with mock.patch.object(foil, "printings", return_value={"base": None, "reverse": None}), \
                mock.patch.object(foil, "me_reverse_plate", return_value=flat), \
                mock.patch.object(foil, "gen8_pattern", return_value=pattern):
            out = foil.build("sm1-1", "me", canvas, {}, {"stage": 0, "type": "Grass"})
        self.assertIsNone(out["base"])
        spec, texture = out["reverse"]
        self.assertEqual(spec, foil.ME_REVERSE)
        card = foil.card_space(texture)
        self.assertEqual(card[700, 300], 140)
        self.assertEqual(card[300, 300], 0)

    def test_a_trainer_takes_the_chevrons_on_its_gen9_field(self):
        from unittest import mock
        gen9 = np.full((H, W), 20, np.float32)
        gen9[40:70, 50:150] = 230                   # the header's lettering
        gen9[140:1000, 30:700] = 120                # the pattern's field...
        gen9[140:530, 55:680] = 20                  # ...round the art window
        gen9[890:990, 240:705] = 20                 # the rule pill's dark rim
        gen9[895:985, 245:700] = 170                # the pill
        pattern = np.full((H, W), 200, np.float32)
        with mock.patch.object(foil, "printings", return_value={"reverse": {}}), \
                mock.patch.object(foil, "plate", return_value=gen9), \
                mock.patch.object(foil, "gen8_pattern", return_value=pattern):
            out = foil.gen8_trainer_plate("Item")
        self.assertEqual(out[50, 100], 230)         # lettering as Gen 9 prints it
        self.assertEqual(out[300, 300], 20)         # the art window dark
        self.assertEqual(out[700, 300], 200)        # the chevrons
        self.assertEqual(out[940, 400], 170)        # the pill as Gen 9 prints it


class InPlace(unittest.TestCase):
    @staticmethod
    def disc(cy, cx, r=12):
        yy, xx = np.mgrid[:H, :W]
        return (np.hypot(yy - cy, xx - cx) < r).astype(np.float32)

    def test_a_removed_cost_leaves_no_cut_out(self):
        """An attack that drops an energy (me02.5-075 Clefable's Magical Shot, Psychic Colorless Colorless ->
        Psychic Colorless, as reported): its third circle was cut out of the reverse plate and must go with it."""
        from unittest import mock
        card = {"id": "me02.5-075", "family": "regular", "type": "Psychic", "prefix": "", "name": "Clefable",
                "hp": "120", "stage": 1, "evolves_from": "Clefairy", "portrait": "", "abilities": [], "retreat": 2,
                "attacks": [{"cost": ["Psychic", "Colorless", "Colorless"], "name": "Magical Shot", "damage": "100",
                             "effect": ""}], "weakness": {"type": "Metal", "value": "×2"}, "resistance": None}
        edited = dict(card, attacks=[dict(card["attacks"][0], cost=["Psychic", "Colorless"])])
        old_icons = self.disc(766, 70) + self.disc(766, 115) + self.disc(766, 160)
        printed = np.full((H, W), 229, np.float32) * (1 - old_icons)          # the cut-out circles
        text = np.zeros((H, W), np.float32)
        text[760:772, 240:420] = 1                                            # foil runs through the text
        drawn = {"icon": self.disc(766, 70) + self.disc(766, 115), "text": text}
        off = np.zeros((H, W), bool)
        off[740:792, 40:440] = True
        with mock.patch.object(foil, "printings", return_value={"base": None, "reverse": {"_f": "FlatSilver", "_s": "x"}}), \
                mock.patch.object(foil, "plate", return_value=printed), \
                mock.patch.object(foil, "field", return_value=(np.full((H, W), 229, np.float32), np.ones((H, W)))), \
                mock.patch.object(foil, "printed_blocks", return_value={"icon": old_icons, "text": text}), \
                mock.patch("svrestyle.modern.taken_off", return_value=(off, np.zeros((H, W), bool))), \
                mock.patch.object(foil, "modern_front", return_value=None), \
                mock.patch.object(foil, "texture", return_value=None):
            out = foil.build_modern("me02.5-075", edited, np.full((H, W, 4), 255, np.float32), drawn, card)
        plate = foil.card_space(out["reverse"][1]).astype(np.float32)
        self.assertGreater(plate[766, 160], 220)          # the third circle: foil again
        self.assertLess(plate[766, 70], 10)               # the two left: cut, at the printed value
        self.assertLess(plate[766, 115], 10)
        self.assertEqual(plate[766, 300], 229)            # text the foil runs through: as printed

    def test_untouched_blocks_keep_the_printed_plate(self):
        from unittest import mock
        card = {"id": "me02-013", "family": "ex", "type": "Fire", "prefix": "", "name": "Mega Charizard X", "hp": "250",
                "stage": 2, "evolves_from": "Charmeleon", "portrait": "", "abilities": [], "retreat": 2,
                "attacks": [{"cost": ["Fire"], "name": "Inferno X", "damage": "90×", "effect": ""}],
                "weakness": {"type": "Water", "value": "×2"}, "resistance": None}
        printed = np.full((H, W), 180, np.float32)
        with mock.patch.object(foil, "printings", return_value={"base": {"_f": "SvUltra", "_s": "x"}, "reverse": None}), \
                mock.patch.object(foil, "plate", return_value=printed), \
                mock.patch.object(foil, "modern_front", return_value=None), \
                mock.patch.object(foil, "texture", return_value=None):
            out = foil.build_modern("me02-013", card, np.full((H, W, 4), 255, np.float32), {}, card)
        np.testing.assert_array_equal(foil.card_space(out["base"][1]), printed)


class AceSpecFoil(unittest.TestCase):
    """build_ace: an ACE SPEC Trainer's own AceFoil plate, the edited text's cut-outs moved."""
    CARD = {"id": "sv05-154", "kind": "Tool", "name": "Maximum Belt", "effect": "Old text.", "illustrator": "A",
            "character": "", "hp": "", "abilities": [], "attacks": [], "ace": True}

    def build(self, edited, plate, scan, drawn):
        from unittest import mock
        record = {"_f": "AceFoil", "_s": "TPCi/Cards3D/HoloFoil/AceFoil"}
        field = np.full((H, W), 110, np.float32)
        with mock.patch.object(foil, "printings", return_value={"base": record, "reverse": None}),                 mock.patch.object(foil, "plate", return_value=plate),                 mock.patch.object(foil, "field", return_value=(field, np.ones((H, W)))),                 mock.patch.object(foil.tcg, "scan", return_value=scan),                 mock.patch.object(foil, "modern_front", return_value=None),                 mock.patch.object(foil, "texture", return_value=None):
            return foil.build_ace("sv05-154", edited, np.full((H, W, 4), 255, np.float32), drawn, self.CARD)

    def test_the_old_text_is_filled_and_the_new_cut_out(self):
        scan = np.zeros((H, W, 4), np.float32)
        scan[..., :3] = (236, 0, 140)
        old = np.zeros((H, W), bool)
        for x in range(100, 400, 12):                       # the printed text: white strokes on the pink
            old[700:716, x:x + 5] = True
        scan[old, :3] = 255
        plate = np.full((H, W), 110, np.float32)
        plate[700:716, 100:400][old[700:716, 100:400]] = 27                       # cut out of the foil
        new = np.zeros((H, W), np.float32)
        new[760:776, 100:400] = 1                           # the new text, a block of cover
        out = self.build(dict(self.CARD, effect="New text."), plate, scan, {"text": new})
        got = foil.card_space(out["base"][1]).astype(np.float32)
        self.assertEqual(got[708, 102], 110)                # the old cut-out: foil again
        self.assertLess(got[768, 200], 40)                  # the new text: cut at the printed value
        self.assertEqual(got[600, 300], 110)                # elsewhere as printed
        self.assertEqual(out["base"][0], ("TPCi/Cards3D/HoloFoil/AceFoil", "MAT_Cards3D_AceFoil", None))

    def test_an_unedited_card_keeps_its_plate(self):
        plate = np.random.default_rng(1).uniform(0, 255, (H, W)).astype(np.float32)
        out = self.build(dict(self.CARD), plate, np.zeros((H, W, 4), np.float32), {})
        np.testing.assert_array_equal(foil.card_space(out["base"][1]), plate.astype(np.uint8))


if __name__ == "__main__":
    unittest.main()
