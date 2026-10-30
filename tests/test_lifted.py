"""HP, damage, weakness-row values and effect-text energy symbols in glyphs lifted off the print: where glyphs go,
how they are fitted, how they are cut off a scan. Pure logic on made-up scans: no network, fonts or extract.
Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

from svrestyle import compose, lifted

W, H = 733, 1024
U = lifted.UP


def glyphs(**over):
    g = dict(shapes={c: np.ones((8 * U, 4 * U), np.float32) for c in ("0", "1", "HP", "+")},
             x={"0": 600.0 * U, "1": 600.0 * U, "HP": 590.0 * U, "+": 610.0 * U},
             y={"0": 50.0 * U, "1": 50.0 * U, "HP": 54.0 * U, "+": -20.0 * U},
             advance={"0": 20.0 * U, "1": 12.0 * U, "HP": 0.0, "+": 0.0},
             kern={("1", "0"): 2.0 * U})
    g.update(over)
    return lifted.Glyphs(**g)


def panel(value=200.0):
    """A made-up card: a panel shading across, with a little grain."""
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:H, :W]
    lum = value - 0.05 * xx + rng.normal(0, 1.5, (H, W))
    return np.repeat(np.clip(lum, 0, 255)[..., None], 4, axis=2).astype(np.float32)


def ink(scan, x, y, w, h, level=30.0):
    scan[y:y + h, x:x + w, :3] = level


class Placing(unittest.TestCase):
    def test_right_aligned_by_the_advances_and_kerns_right_of_each_glyph(self):
        placed = lifted.place(glyphs(), ["HP", "1", "0"])
        self.assertEqual([p[0] for p in placed], ["HP", "1", "0"])
        self.assertAlmostEqual(placed[2][1], 600 * U)                   # nothing right of the last figure
        self.assertAlmostEqual(placed[1][1], (600 - 20 - 2) * U)        # the 0's advance and the 1-0 kern
        self.assertAlmostEqual(placed[0][1], (590 - 12 - 20 - 2) * U)   # the label moves with both figures

    def test_coverage_sets_the_label_and_figures_or_gives_none(self):
        with mock.patch.object(lifted, "load", return_value=glyphs()):
            cov = lifted.coverage("hp", "10")
            self.assertIsNone(lifted.coverage("hp", "20"))            # no 2: the caller sets it in Gill Sans
            self.assertIsNone(lifted.coverage("hp", ""))
        self.assertEqual(cov[54, 601], 1.0)                             # the 0 at x 600
        self.assertEqual(cov[54, 579], 1.0)                             # the 1 at 600 - 20 - 2
        self.assertEqual(cov[58, 557], 1.0)                             # the label at 590 - 12 - 20 - 2, row 54
        self.assertEqual(cov[54, 590], 0.0)

    def test_damage_hangs_from_the_attack_name_baseline(self):
        with mock.patch.object(lifted, "load", return_value=glyphs()):
            cov = lifted.coverage("damage", "10+", base=300)
        ys, xs = np.nonzero(cov[:, 609:615] > 0.5)
        self.assertEqual(ys.min(), 280)                                 # the + 20 px above the baseline

    def test_no_glyphs_no_coverage(self):
        with mock.patch.object(lifted, "load", return_value=None):
            self.assertIsNone(lifted.coverage("damage", "30"))


class Fitting(unittest.TestCase):
    def test_fit_finds_the_advances_kerns_and_rows_it_was_made_with(self):
        x, adv, kern, row = {"0": 2400.0, "1": 2410.0, "2": 2404.0}, {"0": 128.0, "1": 96.0, "2": 120.0}, 8.0, 160.0
        texts = ["10", "20", "100", "110", "120", "200", "210", "220", "1100", "2010"] * 4
        frames = {c: [] for c in x}
        for text in texts:
            for j, ch in enumerate(text):
                right = text[j + 1:]
                corner = x[ch] - sum(adv[c] for c in right) - sum(kern for p in zip(text[j:], right) if p == ("1", "0"))
                frames[ch].append((tuple(text), j, (corner, row)))
        averaged = {c: (np.ones((4, 4), np.float32), f) for c, f in frames.items()}
        fitted = lifted.fit(averaged)
        self.assertEqual(set(fitted.y.values()), {row})
        for text in ("10", "210", "1100", "2110", "1220"):               # seen, and never seen, land where made
            for (ch, got, _), j in zip(lifted.place(fitted, list(text)), range(len(text))):
                right = text[j + 1:]
                want = x[ch] - sum(adv[c] for c in right) - sum(kern for p in zip(text[j:], right) if p == ("1", "0"))
                self.assertAlmostEqual(got, want, delta=2.0)

    def test_average_aligns_shifted_samples(self):
        shape = np.zeros((12 * U, 10 * U), np.float32)
        shape[4 * U:8 * U, 3 * U:6 * U] = 1
        moved = np.roll(shape, (U, 2 * U), (0, 1))
        samples = [(("0",), 0, (shape, (1000, 400))), (("0",), 0, (moved, (1000, 400)))]
        mean, frames = lifted.average(samples)["0"]
        self.assertGreater(mean.max(), 0.99)                            # laid one on the other, not side by side
        (_, _, a), (_, _, b) = frames
        self.assertEqual((b[0] - a[0], b[1] - a[1]), (2 * U, U))        # the moved one's glyph sits that much on

    def test_two_samples_unlike_each_other_give_no_glyph(self):
        bar = np.zeros((12 * U, 12 * U), np.float32)
        bar[1 * U:11 * U, 6 * U:7 * U] = 1                              # the symbol
        letter = bar.T.copy()                                           # what a gap matched to the wrong words holds
        unlike = [(("Dragon",), 0, (bar, (0, 0))), (("Dragon",), 0, (letter, (0, 0)))]
        alike = [(("Dragon",), 0, (bar, (0, 0))), (("Dragon",), 0, (bar.copy(), (0, 0)))]
        self.assertNotIn("Dragon", lifted.average(unlike))              # the caller sets the disc instead
        self.assertIn("Dragon", lifted.average(alike))
        three = unlike + alike[:1]                                      # a majority: the odd one is dropped
        self.assertEqual(len(lifted.average(three)["Dragon"][1]), 2)


class Lifting(unittest.TestCase):
    def test_hp_figures_and_label_come_off_a_shaded_panel(self):
        scan = panel()
        for x in (560, 592):                                            # "70": two tall figures
            ink(scan, x, 42, 22, 42)
        for x in (520, 534):                                            # the label's H and P
            ink(scan, x, 70, 11, 14)
        samples = lifted.hp_samples(scan, "70")
        self.assertEqual(sorted(j for _, j, _ in samples), [0, 1, 2])
        self.assertTrue(all(seq == ("HP", "7", "0") for seq, _, _ in samples))
        figure = next(piece for _, j, piece in samples if j == 2)
        shape, (cx, cy) = figure
        self.assertEqual((cx, cy), ((592 - lifted.PAD) * U, (42 - lifted.PAD) * U))
        self.assertGreater(shape[shape.shape[0] // 2, shape.shape[1] // 2], 0.95)

    def test_a_card_whose_figures_do_not_add_up_is_skipped(self):
        scan = panel()
        ink(scan, 592, 42, 22, 42)
        self.assertEqual(lifted.hp_samples(scan, "70"), [])
        light = panel(80)
        for x in (560, 592):
            ink(light, x, 42, 22, 42, level=250)
        self.assertEqual(lifted.hp_samples(light, "70"), [])          # white figures (full art): not lifted

    def test_damage_with_its_suffix_and_the_name_baseline(self):
        scan = panel()
        for x in (612, 636):                                            # "30+"
            ink(scan, x, 600, 18, 28)
        ink(scan, 660, 608, 12, 12)
        for x in range(260, 380, 20):                                   # the attack name, baseline 628
            ink(scan, x, 602, 14, 26)
        samples = lifted.damage_samples(scan, ["30+"])
        self.assertEqual([seq[j] for seq, j, _, _ in samples], ["3", "0", "+"])
        self.assertTrue(all(base == 628 for *_, base in samples))
        self.assertEqual(lifted.damage_samples(scan, ["30", "50"]), [])   # one printed damage for two attacks

    def test_row_value(self):
        scan = panel()
        ink(scan, 146, 896, 10, 10)                                     # ×
        ink(scan, 162, 890, 14, 18)                                     # 2
        samples = lifted.row_samples(scan, "×2", lifted.ROW_BOXES["weakness"])
        self.assertEqual([seq[j] for seq, j, _ in samples], ["×", "2"])


class Setting(unittest.TestCase):
    def test_hp_and_damage_use_the_lifted_figures(self):
        figures = np.zeros((H, W), np.float32)
        figures[50:60, 600:620] = 1
        with mock.patch.object(lifted, "coverage", return_value=figures) as cov:
            self.assertIs(compose.hp_coverage("120"), figures)
            cov.assert_called_once_with("hp", "120")

    def test_weakness_row_values_use_the_lifted_figures(self):
        figures = np.zeros((H, W), np.float32)
        figures[890:905, 150:170] = 1
        canvas = np.zeros((H, W, 4), np.float32)
        card = {"weakness": {"type": "Fire", "value": "×2"}, "resistance": {"type": "Water", "value": "-30"},
                "retreat": 0}
        with mock.patch.object(lifted, "coverage", return_value=figures) as cov, \
                mock.patch.object(compose, "energy"), mock.patch.object(compose.ts, "text_width", return_value=69.0):
            compose.weakness_row(canvas, card, compose.WHITE, None, labels=False)
        self.assertEqual(cov.call_args_list, [mock.call("weakness", "×2"), mock.call("resistance", "-30")])
        self.assertEqual(canvas[895, 160, 0], 255)


def words(img, x, y, dx=0):
    """Made-up letters (strokes of a few heights) between x and x + 36 on the line whose middle is y."""
    for i, h in enumerate((14, 9, 17, 9, 12, 16)):
        img[y + 8 - h:y + 8, x + dx + 6 * i:x + dx + 6 * i + 3] = 1.0


class Symbols(unittest.TestCase):
    def test_a_symbol_hangs_from_its_gap_centre_and_baseline_and_scales_with_the_text(self):
        energy = glyphs(shapes={"Fire": np.ones((20 * U, 16 * U), np.float32)}, x={"Fire": -8.0 * U},
                        y={"Fire": -22.0 * U}, advance={}, kern={})
        with mock.patch.object(lifted, "load", return_value=energy):
            full = lifted.symbol("Fire", 300, 700)
            half = lifted.symbol("Fire", 300, 700, scale=0.5)
            self.assertIsNone(lifted.symbol("Water", 300, 700))
        ys, xs = np.nonzero(full > 0.5)
        self.assertEqual((xs.min(), xs.max(), ys.min(), ys.max()), (292, 307, 678, 697))
        ys, xs = np.nonzero(half > 0.5)
        self.assertEqual((xs.min(), xs.max(), ys.min(), ys.max()), (296, 303, 689, 698))

    def test_render_keeps_the_most_where_shapes_overlap_and_clips_at_the_card(self):
        a, b = np.full((8, 8), 0.5, np.float32), np.ones((8, 8), np.float32)
        out = lifted.render([(a, 0, 0), (b, 4, 4), (b, -8, 1020 * U)])
        self.assertEqual(out[0, 0], 0.5)
        self.assertEqual(out[1, 1], 1.0)
        self.assertEqual(out.shape, (H, W))

    def test_a_gap_is_found_where_the_words_either_side_land_on_the_print(self):
        ours, printed = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
        words(ours, 252, 600), words(ours, 312, 600)                   # our gap centred on 300
        words(printed, 252, 605, dx=-2), words(printed, 312, 605, dx=4)  # the print's, wider and 5 px lower
        self.assertEqual(lifted.find_gap(ours, printed, 300, 600), (301.0, 5.0))
        self.assertIsNone(lifted.find_gap(ours, np.zeros((H, W), np.float32), 300, 600))

    def test_effect_text_takes_the_printed_symbol_else_the_disc(self):
        printed = np.ones((H, W), np.float32)
        with mock.patch.object(lifted, "symbol", return_value=printed) as symbol, \
                mock.patch.object(compose, "inline_energy") as disc:
            self.assertIs(compose.energy_symbol("Fire", 300, 700, compose.BODY_SIZE / 2), printed)
            symbol.assert_called_once_with("Fire", 300, 700, 0.5)
            disc.assert_not_called()
        with mock.patch.object(lifted, "symbol", return_value=None), \
                mock.patch.object(compose, "inline_energy", return_value="disc") as disc, \
                mock.patch.object(compose.ts, "cap_height", return_value=17.0):
            self.assertEqual(compose.energy_symbol("Fire", 300, 700, compose.BODY_SIZE), "disc")
            disc.assert_called_once_with("Fire", 300, 700 - 8.5, compose.BODY_SIZE * compose.INLINE_R)


if __name__ == "__main__":
    unittest.main()
