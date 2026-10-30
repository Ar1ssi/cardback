"""Fairy in the Scarlet & Violet / Mega Evolution look (fairy.py): the type in the editor and its rules, the Psychic ->
Fairy colour map (fitted here on synthetic coins), the symbol in effect text, and, with a TCG Live extract
(SVRESTYLE_TCGLIVE) and Gill Sans (SVRESTYLE_FONTS), its icons, frames and Basic Energy.
Run: python -m unittest discover tests"""
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from svrestyle import edit, fairy, follow, meframe, typeset as ts
from svrestyle.imaging import dilate, fill_holes


def turned(rgb, degrees, gain=1.0):
    """rgb with its Oklab hue turned and its chroma scaled."""
    lab = fairy.oklab(rgb)
    chroma, hue = np.hypot(lab[..., 1], lab[..., 2]) * gain, np.arctan2(lab[..., 2], lab[..., 1]) + np.radians(degrees)
    return fairy.from_oklab(np.stack([lab[..., 0], chroma * np.cos(hue), chroma * np.sin(hue)], -1))


def coins():
    """A Psychic coin of purples from dark to light, and the Fairy coin as the same turned 30 degrees, chroma x1.2."""
    light = np.linspace(0.42, 0.68, 512)[:, None].repeat(512, 1)
    lab = np.stack([light, np.full_like(light, 0.10), np.full_like(light, -0.12)], -1)
    psychic = fairy.from_oklab(lab).astype(np.float32)
    return {"psychic": psychic, "fairy": turned(psychic, 30, 1.2).astype(np.float32)}


class Types(unittest.TestCase):
    def test_fairy_is_an_energy_type(self):
        self.assertIn("Fairy", edit.ENERGY)
        self.assertEqual(edit.LETTER["Fairy"], "Y")
        self.assertEqual(ts.energy_text("Attach a {Y} Energy")[1], ["Fairy"])
        self.assertIn("Fairy", meframe.TYPES)
        card = {"id": "sm5-94", "family": "regular", "type": "Fairy", "prefix": "", "name": "Tapu Lele", "hp": "110",
                "stage": 0, "abilities": [], "attacks": [], "weakness": None, "resistance": None, "retreat": 1}
        out = edit.apply(card, {"weakness": {"type": "Fairy", "value": "×2"},
                                "attacks": [{"cost": ["Fairy", "Colorless"], "name": "Wonder Shine", "damage": "70"}]})
        self.assertEqual((out["weakness"]["type"], out["attacks"][0]["cost"]), ("Fairy", ["Fairy", "Colorless"]))

    def test_follow_takes_the_sun_and_moon_rule(self):
        # weak to Metal, resisting Darkness, at the new era's -30 (follow.FAIRY)
        with mock.patch.object(follow.tcg, "card", return_value={"weaknesses": [], "resistances": []}):
            follow.rules.cache_clear()
            try:
                self.assertEqual(follow.rules(False)["Fairy"], ("Metal", ("Darkness", "-30")))
            finally:
                follow.rules.cache_clear()


class ColourMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        fake = coins()
        self.patches = [mock.patch.object(fairy, "coin", side_effect=lambda t: fake[t]),
                        mock.patch.object(fairy, "_asset", side_effect=lambda name: Path(self.tmp.name, name))]
        for p in self.patches:
            p.start()
        fairy.colour_map.cache_clear()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        fairy.colour_map.cache_clear()
        self.tmp.cleanup()

    def test_greys_stay(self):
        greys = np.array([[0, 0, 0], [255, 255, 255], [200, 200, 202], [40, 40, 40]], np.float32)
        np.testing.assert_allclose(fairy.colour(greys), greys, atol=1.0)

    def test_mid_tones_turn_as_the_coins_do(self):
        lab = np.array([[0.55, 0.10, -0.12]])
        purple = fairy.from_oklab(lab)
        np.testing.assert_allclose(fairy.colour(purple), turned(purple, 30, 1.2), atol=3.0)

    def test_pastels_turn_their_hue_alone(self):
        pastel = fairy.from_oklab(np.array([[0.9, 0.04, -0.05]]))
        got = fairy.oklab(fairy.colour(pastel))[0]
        want = fairy.oklab(turned(pastel, 30))[0]           # past LIGHT_TAPER: lightness and chroma kept
        np.testing.assert_allclose(got, want, atol=0.01)

    def test_colours_stay_in_srgb(self):
        violet = np.array([[60, 0, 120], [30, 0, 70]], np.float32)       # chroma gain would take these out
        out = fairy.colour(violet)
        self.assertTrue(((out >= 0) & (out <= 255)).all())
        self.assertTrue((out.max(-1) > 20).all())                         # not clipped to black


class Glyph(unittest.TestCase):
    def test_the_symbol_sits_where_the_lifted_ones_do(self):
        yy, xx = np.mgrid[:512, :512]
        disc = (np.hypot(xx - fairy.COIN_C[0], yy - fairy.COIN_C[1]) < 100).astype(np.float32)
        with mock.patch.object(fairy, "coin_symbol", return_value=disc):
            shape, x, y = fairy.glyph(4)
            ys, xs = np.nonzero(shape > 0.5)
            cx, cy = x + (xs.min() + xs.max()) / 2, y + (ys.min() + ys.max()) / 2
            self.assertAlmostEqual(cx / 4, fairy.GLYPH_AT[0], delta=0.3)
            self.assertAlmostEqual(cy / 4, fairy.GLYPH_AT[1], delta=0.3)
            # the art radius: a disc of 100 of the coin's 164 px
            self.assertAlmostEqual((xs.max() - xs.min() + 1) / 4 / 2, fairy.GLYPH_R * 100 / fairy.COIN_ART_R, delta=0.4)
            half, _, _ = fairy.glyph(4, scale=0.5)
            self.assertLess(half.shape[0], shape.shape[0])


@unittest.skipUnless(os.environ.get("SVRESTYLE_FONTS") and os.environ.get("SVRESTYLE_TCGLIVE"),
                     "needs Gill Sans (SVRESTYLE_FONTS) and a TCG Live extract (SVRESTYLE_TCGLIVE)")
class Composing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ts.configure(os.environ["SVRESTYLE_FONTS"])

    @staticmethod
    def hue(rgb):
        lab = fairy.oklab(rgb)
        return np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))

    def test_icons_are_pink_with_the_symbol(self):
        for kind in ("name", "cost"):
            (crop, alpha, scx, scy, (ox, oy)), r = meframe.icon("Fairy", kind)
            yy, xx = np.mgrid[:crop.shape[0], :crop.shape[1]]
            rr = np.hypot(xx - (scx - ox), yy - (scy - oy))
            disc = crop[(rr > 0.8 * r) & (rr < r - 1.5)]
            symbol = fill_holes((crop.max(-1) < 60) & (rr < 0.9 * r))
            ring = dilate(symbol, 2) & ~symbol & (rr < 0.85 * r)
            away = dilate(symbol, 7) & ~dilate(symbol, 4) & (rr < 0.85 * r)
            light = lambda px: np.median(fairy.oklab(px)[:, 0])
            with self.subTest(kind=kind):
                self.assertTrue(np.median(disc[:, 0]) > np.median(disc[:, 2]) > np.median(disc[:, 1]))   # pink
                self.assertGreater((crop[rr < 0.5 * r].max(-1) < 60).mean(), 0.2)                         # the symbol
                self.assertGreater(light(crop[ring]) - light(crop[away]), 0.03)      # a light ring round it, as Psychic's

    def test_frame_is_psychics_in_fairy_colours(self):
        psychic, fairy_ = meframe.template("Psychic"), meframe.template("Fairy")
        x0, y0, x1, y1 = meframe.PANEL
        turn = np.median(self.hue(fairy_[y0:y1, x0:x1, :3]) - self.hue(psychic[y0:y1, x0:x1, :3]))
        self.assertTrue(15 < turn < 50, turn)
        border = np.s_[300:700, 2:20, :3]                 # the silver border stays
        self.assertLess(np.abs(fairy_[border] - psychic[border]).max(), 3)

    def test_a_sun_and_moon_fairy_card_is_restyled(self):
        from svrestyle import mecompose
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img, report = mecompose.compose_card("sm5-94")
        self.assertEqual(report["card"]["type"], "Fairy")
        x0, y0, x1, y1 = meframe.PANEL
        self.assertGreater(np.median(img[y0:y1, x0:x1, 0]) - np.median(img[y0:y1, x0:x1, 1]), 40)     # pink panel

    def test_the_symbol_in_text(self):
        from svrestyle import compose
        cover = compose.energy_symbol("Fairy", 300, 700, compose.BODY_SIZE)
        ys, xs = np.nonzero(cover > 0.5)
        self.assertTrue(len(ys))
        self.assertLess(abs((xs.min() + xs.max()) / 2 - 300), 4)              # in its gap
        self.assertTrue(680 < ys.min() < ys.max() < 702)                      # on the line, a little over the caps

    def test_basic_energy(self):
        from svrestyle import energy, tcg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img, report = energy.compose_card("sm1-172")
        self.assertEqual(report["card"]["type"], "Fairy")
        psychic = tcg.scan("mee-005")
        ball = np.s_[440:640, 260:470, :3]                   # the ball's middle: Fairy's symbol, not Psychic's eye
        self.assertGreater(np.abs(img[ball] - psychic[ball]).mean(), 20)
        sky = np.s_[200:260, 60:120, :3]                    # the background: pink
        self.assertGreater(np.median(img[sky][..., 0]) - np.median(img[sky][..., 1]), 40)
