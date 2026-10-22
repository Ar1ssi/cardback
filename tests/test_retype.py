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
                      mock.patch.object(retype, "_footer_boxes", return_value=np.zeros((H, W), np.float32))):
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


class TemplateRows(unittest.TestCase):
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
        self.assertNotIn("type", edit.editable({k: v for k, v in card().items() if k != "family"}))   # V -> ex

    def test_apply(self):
        self.assertEqual(edit.apply(card(), {"type": "Water"})["type"], "Water")
        with self.assertRaisesRegex(ValueError, "unknown energy type"):
            edit.apply(card(), {"type": "Fairy"})
        with self.assertRaisesRegex(ValueError, "only regular"):
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
