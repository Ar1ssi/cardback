"""Tests for the logic that needs no network, model or fonts. Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from svrestyle import compose, swsh, tcg
from svrestyle import typeset as ts


class PokemonTcgId(unittest.TestCase):
    def test_sword_and_shield(self):
        self.assertEqual(tcg.pokemontcg_id("swsh6-99"), "swsh6/99")
        self.assertEqual(tcg.pokemontcg_id("swsh9-040"), "swsh9/40")

    def test_half_sets(self):
        self.assertEqual(tcg.pokemontcg_id("sv03.5-065"), "sv3pt5/65")
        self.assertEqual(tcg.pokemontcg_id("swsh12.5-013"), "swsh12pt5/13")

    def test_paired_sets(self):
        # TCGdex sv10.5b-001 / sv10.5w-001 are Snivy (Black Bolt) / Sewaddle (White Flare): pokemontcg.io
        # zsv10pt5-1 / rsv10pt5-1
        self.assertEqual(tcg.pokemontcg_id("sv10.5b-001"), "zsv10pt5/1")
        self.assertEqual(tcg.pokemontcg_id("sv10.5w-001"), "rsv10pt5/1")
        with self.assertRaises(ValueError):
            tcg.pokemontcg_id("sv10.5x-001")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            tcg.pokemontcg_id("not a card")


class TcgLiveId(unittest.TestCase):
    def test_sets_by_rule(self):
        self.assertEqual(tcg.tcglive_id("swsh6-099"), "swsh6_99")
        self.assertEqual(tcg.tcglive_id("sv03.5-065"), "sv3-5_65")
        self.assertEqual(tcg.tcglive_id("me01-054"), "me1_54")
        self.assertEqual(tcg.tcglive_id("swsh12-089"), "swsh12_89")

    def test_promos_and_subsets(self):
        self.assertEqual(tcg.tcglive_id("swshp-SWSH055"), "swshbsp_55")
        self.assertEqual(tcg.tcglive_id("sma-SV33"), "sm11-5_102")
        self.assertEqual(tcg.tcglive_id("sv10.5b-086"), "zsv10-5_86")      # bundles zsv10-5_en_086
        self.assertEqual(tcg.tcglive_id("sv10.5w-172"), "rsv10-5_172")

    def test_no_extract_means_no_front(self):
        with mock.patch.object(tcg, "TCGLIVE", None):
            self.assertIsNone(tcg.tcglive_front("swsh6-99"))


class ReminderTextItalics(unittest.TestCase):
    def test_parentheses_are_italic(self):
        out = ts.words("Heal 30. (Don't apply Weakness.) Then draw.")
        faces = {w: f for w, f in out}
        self.assertEqual(faces["Heal"], ts.MEDIUM)
        self.assertEqual(faces["(Don't"], ts.MEDIUM_ITALIC)
        self.assertEqual(faces["Weakness.)"], ts.MEDIUM_ITALIC)
        self.assertEqual(faces["Then"], ts.MEDIUM)

    def test_runs_merge_same_face(self):
        runs = ts.runs_of([("a", ts.MEDIUM), ("b", ts.MEDIUM), ("(c)", ts.MEDIUM_ITALIC)])
        self.assertEqual(runs, [("a b ", ts.MEDIUM), ("(c)", ts.MEDIUM_ITALIC)])


def fake_card(**over):
    base = {"id": "swsh6-99", "name": "Galarian Slowking V", "category": "Pokemon", "stage": "Basic",
            "types": ["Darkness"], "hp": 220, "localId": "099", "set": {"id": "swsh6"},
            "attacks": [{"name": "Word of Ruin", "effect": "The opponent's Pokémon is Knocked Out.", "cost": ["Darkness"]}],
            "retreat": 3, "illustrator": "Mitsuhiro Arita", "regulationMark": "E"}
    base.update(over)
    return base


SET = {"cardCount": {"official": 198}, "abbreviation": {"official": "CRE"}, "releaseDate": "2021-06-18"}


class ReadCard(unittest.TestCase):
    def read(self, **over):
        with mock.patch.object(tcg, "card", return_value=fake_card(**over)), \
             mock.patch.object(tcg, "set_info", return_value=SET):
            return compose.read_card("swsh6-99")

    def test_fields(self):
        c = self.read()
        self.assertEqual((c["prefix"], c["name"], c["number"], c["set_code"], c["year"]),
                         ("Galarian", "Slowking", "099/198", "CRE", "2021"))
        self.assertIn("’", c["attacks"][0]["effect"])      # printed apostrophe

    def test_rejects_vmax(self):
        with self.assertRaises(SystemExit):
            self.read(name="Galarian Slowking VMAX", stage="VMAX")

    def test_rejects_type_without_sv_frame(self):
        with self.assertRaises(SystemExit):
            self.read(types=["Fairy"])


class TextDetection(unittest.TestCase):
    def test_finds_keylined_white_text_and_not_plain_art(self):
        rng = np.random.default_rng(0)
        img = Image.fromarray((rng.uniform(70, 160, (1024, 733, 3))).astype(np.uint8))
        d = ImageDraw.Draw(img)
        f = ImageFont.load_default(size=26)
        d.text((60, 700), "Discard a card from your hand. Then draw.", font=f, fill=(255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0))
        d.ellipse([400, 560, 470, 630], fill=(240, 240, 240))       # bright art blob with no keyline
        bands, badges, glyphs = swsh.find_text(np.asarray(img).astype(np.float32))
        self.assertTrue(any(b.y0 <= 712 <= b.y1 for b in bands))
        self.assertFalse(any(b.y0 <= 595 <= b.y1 for b in bands))
        self.assertFalse(badges)
        self.assertTrue(glyphs[705:725, 80:300].any())


if __name__ == "__main__":
    unittest.main()
