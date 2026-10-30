"""Type changes for regular Pokemon: the editor's field and checks, and recolouring a print from one type's frame to
another's on synthetic frames. No network, model, fonts or extract. Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

from svrestyle import compose, edit, foil, frame, layers, meframe, modern, retype

W, H = 733, 1024
ORANGE, TEAL, SILVER = (247, 174, 120), (0, 89, 104), (200, 200, 200)
RED = (225, 24, 33)


def card(**over):
    c = {"id": "me01-019", "family": "regular", "type": "Fire", "prefix": "", "name": "Vulpix", "hp": "70",
         "stage": 0, "abilities": [], "retreat": 0, "attacks": [], "weakness": None, "resistance": None}
    c.update(over)
    return c


def frame_of(colour, phase):
    """A type's frame: its colour with a faint texture of its own, a silver Pokedex strip, a grey art window."""
    yy, xx = np.mgrid[:H, :W]
    img = np.zeros((H, W, 4), np.float32)
    img[..., :3] = np.array(colour, np.float32) + 4 * np.sin((xx + phase) / 7.0)[..., None]
    img[484:510, 90:660, :3] = SILVER
    x0, y0, x1, y1 = meframe.ART
    img[y0:y1, x0:x1, :3] = 128
    img[..., 3] = 255
    return img


class Recolour(unittest.TestCase):
    def setUp(self):
        self.frames = {"Fire": frame_of(ORANGE, 0), "Darkness": frame_of(TEAL, 3), "Water": frame_of((90, 180, 240), 5)}
        for patch in (mock.patch.object(meframe, "template", side_effect=lambda t: self.frames[t].copy()),
                      mock.patch.object(retype, "_footer_boxes", return_value=np.zeros((H, W), bool))):
            patch.start()
            self.addCleanup(patch.stop)
        retype._dark_panel.cache_clear()
        self.addCleanup(retype._dark_panel.cache_clear)

    def printed(self):
        """A Fire card: a dark word, a red badge, a dex line on the strip, art in the window."""
        img = self.frames["Fire"].copy()
        img[600:607, 200:300, :3] = frame.DARK_INK            # a word's stroke
        img[700:720, 60:110, :3] = RED                        # the Ability badge
        img[494:500, 200:260, :3] = frame.DARK_INK            # Pokedex line on the silver strip
        img[300:310, 300:310, :3] = (10, 200, 30)              # art
        return img

    def test_ink_follows_the_panel(self):
        np.testing.assert_array_equal(retype.ink("Fire"), frame.DARK_INK)
        np.testing.assert_array_equal(retype.ink("Darkness"), compose.WHITE)

    def test_frame_swapped_print_kept(self):
        out = retype.recolour(self.printed(), card(), "Darkness")[..., :3]
        new = self.frames["Darkness"][..., :3]
        np.testing.assert_allclose(out[800:850, 300:600], new[800:850, 300:600], atol=0.5)   # the new panel
        np.testing.assert_allclose(out[602, 220:280], np.tile(compose.WHITE, (60, 1)), atol=1)  # Darkness words: white
        np.testing.assert_allclose(out[710, 75:95], np.tile(RED, (20, 1)), atol=1)            # the badge as printed
        np.testing.assert_allclose(out[496, 210:250], np.tile(frame.DARK_INK, (40, 1)), atol=1)  # silver: as printed
        np.testing.assert_array_equal(out[300:310, 300:310], np.tile((10, 200, 30), (10, 10, 1)))  # art untouched

    def test_retyped_words_recorded_for_the_foil(self):
        with layers.recording() as drawn:
            retype.recolour(self.printed(), card(), "Darkness")
        words = drawn[retype.WORDS]
        self.assertGreater(words[602, 220:280].min(), 0.9)       # the word set again in the new ink
        self.assertEqual(words[710, 75:95].max(), 0)              # the badge keeps its colours: no word
        self.assertEqual(words[494:500, 200:260].max(), 0)        # the silver strip stays as printed

    def test_words_keep_their_ink_between_light_types(self):
        out = retype.recolour(self.printed(), card(), "Water")[..., :3]
        np.testing.assert_allclose(out[602, 220:280], np.tile(frame.DARK_INK, (60, 1)), atol=1)
        np.testing.assert_allclose(out[800:850, 300:600], self.frames["Water"][800:850, 300:600, :3], atol=0.5)

    def test_back_and_forth(self):
        there = retype.recolour(self.printed(), card(), "Darkness")
        back = retype.recolour(there, card(type="Darkness"), "Fire")
        np.testing.assert_allclose(back[602, 220:280, :3], np.tile(frame.DARK_INK, (60, 1)), atol=2)
        np.testing.assert_allclose(back[800:850, 300:600, :3], self.frames["Fire"][800:850, 300:600, :3], atol=0.5)


class RedNames(Recolour):
    """Red Ability names keep their red; dark panels print them in a white keyline, light ones without (mecompose)."""
    NAME = (slice(650, 662), slice(300, 360))

    def named(self, frame_name, keyline):
        img = self.frames[frame_name].copy()
        if keyline:
            img[648:664, 298:362, :3] = compose.WHITE
        img[self.NAME[0], self.NAME[1], :3] = RED
        return img

    def test_light_to_dark_adds_a_keyline(self):
        out = retype.recolour(self.named("Fire", False), card(), "Darkness")[..., :3]
        np.testing.assert_allclose(out[656, 310:350], np.tile(RED, (40, 1)), atol=1)
        np.testing.assert_allclose(out[649, 310:350], np.tile(compose.WHITE, (40, 1)), atol=8)   # a 1 px line, anti-aliased
        np.testing.assert_allclose(out[640, 310:350], self.frames["Darkness"][640, 310:350, :3], atol=0.5)

    def test_dark_to_light_drops_it(self):
        out = retype.recolour(self.named("Darkness", True), card(type="Darkness"), "Water")[..., :3]
        np.testing.assert_allclose(out[656, 310:350], np.tile(RED, (40, 1)), atol=1)
        np.testing.assert_allclose(out[649, 310:350], self.frames["Water"][649, 310:350, :3], atol=1)


class Shifted(Recolour):
    """A Scarlet & Violet print sits a pixel or so off its Mega Evolution template: it is moved onto it first."""
    def textured(self, colour, seed):
        img = self.frames["Fire"].copy()
        img[..., :3] = np.array(colour, np.float32) + np.random.default_rng(seed).normal(0, 6, (H, W, 1))
        img[484:510, 90:660, :3] = SILVER
        return img

    def test_offset_found(self):
        old = self.textured(ORANGE, 1)[..., :3]
        for shift in ((1, 1), (-2, 0), (0, 0)):
            self.assertEqual(retype.offset(retype._shift(old, *shift), old), shift)

    def test_shifted_print_recoloured(self):
        self.frames["Fire"], self.frames["Darkness"] = self.textured(ORANGE, 1), self.textured(TEAL, 2)
        out = retype.recolour(retype._shift(self.printed(), 1, 1), card(), "Darkness")[..., :3]
        new = retype._shift(self.frames["Darkness"][..., :3], 1, 1)
        np.testing.assert_allclose(out[800:850, 300:600], new[800:850, 300:600], atol=0.5)   # the grain lines up
        np.testing.assert_allclose(out[603, 222:281], np.tile(compose.WHITE, (59, 1)), atol=1)


class Icons(unittest.TestCase):
    def test_rim_found_on_the_print(self):
        """An energy icon a pixel off the composers' place, with a figure inside edged only in part."""
        yy, xx = np.mgrid[:H, :W]
        img = np.full((H, W, 3), (90, 180, 240), np.float32)
        icon = np.clip(12.6 + 0.5 - np.hypot(xx - 126.5, yy - 896.5), 0, 1)[..., None]    # anti-aliased, as printed
        img = icon * np.array((247, 174, 8), np.float32) + (1 - icon) * img
        img[890:902, 124:128] = 0                                     # the bolt
        cx, cy, r = retype._on_rim(img, 127.0, 895.8, 13.1)
        self.assertLess(abs(cx - 126.5) + abs(cy - 896.5), 0.6)
        self.assertLess(abs(r - 12.6), 0.6)

    def test_disc_is_whole_inside(self):
        cover = retype._disc(100, 100, 10)
        self.assertEqual(cover[100, 91], 1.0)                         # white on a white panel still counts
        self.assertEqual(cover[100, 112], 0.0)
        self.assertTrue(0 < cover[100, 110] < 1)                       # the anti-aliased rim


class FooterBoxes(unittest.TestCase):
    def test_white_box_on_a_light_frame(self):
        """A white box differs from a light frame only by its 1 px dark outline: the box is found whole."""
        img = np.full((H, W, 3), 225, np.float32)
        img[960:988, 44:63] = frame.DARK_INK
        img[961:987, 45:62] = 255
        drawn = np.zeros((H, W), np.float32)
        drawn[960:988, 44:63] = 1
        with mock.patch.object(retype.tcg, "card", return_value={"set": {"id": "x"}, "regulationMark": "G"}), \
                mock.patch.object(retype.tcg, "set_info", return_value={}), \
                mock.patch.object(retype.mecompose, "set_code", return_value=""), \
                mock.patch.object(retype.compose, "footer_boxes", return_value=drawn):
            boxes = retype._footer_boxes("x", img)
        self.assertTrue(boxes[960:988, 44:63].all())
        self.assertFalse(boxes[950:958].any() or boxes[:, 66:].any())


class TemplateRows(unittest.TestCase):
    def test_row_values_take_the_rows_grain(self):
        """A weakness-row value box no card leaves open is filled with the row's own grain, not a blur."""
        rng = np.random.default_rng(0)
        clean = np.full((H, W, 4), 255, np.float32)
        clean[..., :3] = 150 + rng.normal(0, 8, (H, W, 1))
        need = np.zeros((H, W), bool)
        x0, y0, x1, y1 = meframe.ROW_VALUES[0]
        need[y0 + 2:y1 - 2, x0:x1] = True
        before = clean.copy()
        filled = meframe._row_grain(clean, need, np.zeros((H, W), bool))
        self.assertTrue(filled[need].all())
        hole = clean[y0 + 2:y1 - 2, x0:x1, 0]
        self.assertGreater(hole.std(), 6)                             # grain kept (a blur would be ~0)
        self.assertLess(abs(hole.mean() - 150), 3)
        self.assertFalse((clean[~need] != before[~need]).any())

    def test_dex_text_is_letters_only(self):
        """The strip's own lines run along it: only what differs along the row is the dex line."""
        img = np.full((H, W, 3), 200, np.float32)
        img[495:497, :] = 120                                         # a line of the strip's shading
        img[490:500, 150:156] = 20                                    # a letter
        text = meframe._dex_text(img)
        self.assertTrue(text[492, 152])
        self.assertFalse(text[495, 400])


    def test_median_of_prints(self):
        prints = {f"c{i}": np.full((H, W, 4), 100.0, np.float32) for i in range(3)}
        prints["c1"][911:914, 200:260, :3] = 0                       # one card's print over the rule
        with mock.patch.object(meframe.tcg, "scan", side_effect=lambda i: prints[i]):
            rows = meframe._median_rows(list(prints), meframe.LOWER_RULE)
        self.assertEqual(rows.shape, (meframe.LOWER_RULE[1] - meframe.LOWER_RULE[0], W, 3))
        np.testing.assert_array_equal(rows, 100)


class Editor(unittest.TestCase):
    def test_only_regular_cards_show_a_type(self):
        self.assertEqual(edit.editable(card())["type"], "Fire")
        self.assertNotIn("type", edit.editable(card(family="ex", name="Gengar")))
        self.assertNotIn("type", edit.editable({k: v for k, v in card().items() if k != "family"}))

    def test_readers_can_allow_a_type(self):
        # a Pokemon V restyled as ex, a Mega ex print megaretype recolours
        self.assertEqual(edit.editable({**{k: v for k, v in card().items() if k != "family"}, "retype": True})["type"],
                         "Fire")
        mega = card(family="ex", name="Mega Charizard X", retype=True)
        self.assertEqual(edit.apply(mega, {"type": "Dragon"})["type"], "Dragon")
        with self.assertRaisesRegex(ValueError, "keeps its type"):
            edit.apply(card(family="ex", name="Mega Gengar", retype=False), {"type": "Water"})

    def test_apply(self):
        self.assertEqual(edit.apply(card(), {"type": "Water"})["type"], "Water")
        with self.assertRaisesRegex(ValueError, "unknown energy type"):
            edit.apply(card(), {"type": "Light"})
        self.assertEqual(edit.apply(card(), {"type": "Fairy"})["type"], "Fairy")      # its frame: fairy.py
        with self.assertRaisesRegex(ValueError, "keeps its type"):
            edit.apply(card(family="ex"), {"type": "Water"})
        self.assertEqual(edit.apply(card(family="ex"), {"type": "Fire"})["type"], "Fire")     # unchanged: fine

    def test_changes_and_groups(self):
        after = edit.apply(card(), {"type": "Psychic"})
        self.assertIn("Type: Fire → Psychic", edit.changes(card(), after))
        self.assertEqual(modern.changed(card(), after), ("type",))

    def test_still_printed(self):
        printed = card(stage=1, evolves_from="Vulpix", retreat=2, weakness={"type": "Water", "value": "×2"},
                       attacks=[{"cost": ["Fire"], "name": "Ember", "damage": "30", "effect": ""}])
        edited = dict(printed, type="Water", retreat=1, weakness=None, stage=0, attacks=[])
        shown = modern.still_printed(printed, edited, ("moves", "row", "stage", "type"))
        self.assertEqual((shown["attacks"], shown["weakness"], shown["retreat"], shown["stage"]), ([], None, 1, 0))
        self.assertEqual(shown["type"], "Fire")                  # recoloured from the printed type
        kept = modern.still_printed(printed, dict(printed, type="Water"), ("type",))
        self.assertEqual(kept, printed)


if __name__ == "__main__":
    unittest.main()


class LetteringPlate(unittest.TestCase):
    """A type change between white and dark lettering: reverse plates cut white lettering out (Darkness) and run
    the foil through dark lettering (every other type, me1-me2 plates)."""
    def plate(self):
        img = np.full((H, W), 229, np.float32)
        img[0:8, :] = 0                       # the border
        img[40:80, 140:150] = 0               # a letter touching nothing
        img[8:30, 300:304] = 0                # a letter joined to the border
        img[600:620, 60:80] = 0               # an energy symbol beside the text
        img[640:650, 200:300] = 0             # a line of text
        img[640:643, 305:308] = 0             # an apostrophe the word finder missed
        return img

    def letters(self):
        out = np.zeros((H, W), np.float32)
        out[40:80, 140:150] = 1
        out[10:30, 300:304] = 1
        out[640:650, 200:300] = 1
        return out

    def test_cutouts_are_the_lettering_only(self):
        gone = foil.lettering_cutouts(self.plate(), self.letters())
        self.assertTrue(gone[40:80, 140:150].all() and gone[640:650, 200:300].all())
        self.assertTrue(gone[640:643, 305:308].all())                # small marks beside the words go too
        self.assertTrue(gone[34:37, 140:150].all())                   # with the rim the plate shows round them
        self.assertFalse(gone[600:620, 60:80].any())                  # the energy symbol stays cut
        self.assertFalse(gone[0:5, 200:400].any())                    # the border stays, though a letter joins it

    def test_fill_takes_the_foil_round_it(self):
        img = self.plate()
        gone = foil.lettering_cutouts(img, self.letters())
        filled = foil.fill_from(img, ~foil.dilate(gone | (img < foil.MARK_DARK), foil.MARK_GROW))
        np.testing.assert_allclose(filled[gone & (img < foil.MARK_DARK)], 229, atol=1)
