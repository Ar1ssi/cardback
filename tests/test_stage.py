"""Stage editing: Basic / Stage 1 / Stage 2, "Evolves from" and the evolution portrait. Pure logic: no network,
model, fonts or extract (TCGdex and scans are mocked). Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np
from PIL import ImageFont

from svrestyle import edit, foil, modern, stage

W, H = 733, 1024


def card(**over):
    c = {"id": "sv03.5-064", "family": "regular", "type": "Psychic", "prefix": "", "name": "Kadabra", "hp": "80",
         "stage": 1, "evolves_from": "Abra", "portrait": "", "portrait_of": "Abra", "abilities": [], "retreat": 1,
         "attacks": [{"cost": ["Psychic"], "name": "Psyshot", "damage": "30", "effect": ""}],
         "weakness": {"type": "Darkness", "value": "×2"}, "resistance": {"type": "Fighting", "value": "-30"}}
    c.update(over)
    return c


class Editing(unittest.TestCase):
    def test_round_trip_keeps_stage(self):
        c = card()
        out = edit.apply(c, edit.editable(c))
        self.assertEqual((out["stage"], out["evolves_from"], out["portrait"]), (1, "Abra", ""))

    def test_stage_and_portrait(self):
        out = edit.apply(card(), {"stage": 2, "evolves_from": "Kadabra", "portrait": " sv03.5-064 "})
        self.assertEqual((out["stage"], out["evolves_from"], out["portrait"]), (2, "Kadabra", "sv03.5-064"))
        self.assertEqual(edit.apply(card(), {"stage": 0, "evolves_from": ""})["stage"], 0)

    def test_mega_stage(self):
        out = edit.apply(card(), {"stage": 3, "evolves_from": "Kadabra"})
        self.assertEqual(out["stage"], stage.MEGA)
        self.assertEqual(edit.changes(card(), out), ["Stage: Stage 1 → Mega", "Evolves from: Abra → Kadabra"])
        self.assertEqual(modern.stage_parts(card(), card(stage=3)), ("tab",))
        with self.assertRaises(ValueError):
            edit.apply(card(), {"stage": 3, "evolves_from": ""})
        with mock.patch.object(stage, "mega_tab", return_value="mega") as tab:
            self.assertEqual(stage.sprite_of(stage.MEGA), "mega")
        tab.assert_called_once_with()

    def test_rejections(self):
        for edits in ({"stage": 4}, {"stage": True}, {"stage": "1"}, {"evolves_from": ""},
                      {"stage": 2, "evolves_from": "  "}, {"portrait": "../x"}, {"portrait": 5}):
            with self.subTest(edits), self.assertRaises(ValueError):
                edit.apply(card(), edits)

    def test_stage_is_its_own_group(self):
        self.assertEqual(modern.changed(card(), card(stage=2)), ("stage",))
        self.assertEqual(modern.changed(card(), card(evolves_from="Ralts")), ("stage",))
        self.assertEqual(modern.changed(card(), card(portrait="me01-055")), ("stage",))


class Parts(unittest.TestCase):
    def test_what_is_drawn(self):
        basic = card(stage=0, evolves_from="", portrait_of=None)
        self.assertEqual(modern.stage_parts(basic, card()), stage.REGULAR)
        self.assertEqual(modern.stage_parts(card(family="ex"), card(family="ex", stage=0)), ("tab",))
        self.assertEqual(modern.stage_parts(basic, card(family="ex")), stage.EX)
        self.assertEqual(modern.stage_parts(card(), card(stage=2)), ("tab",))
        self.assertEqual(modern.stage_parts(card(), card(evolves_from="Ralts")), ("strip", "disc"))
        self.assertEqual(modern.stage_parts(card(), card(portrait="me01-055")), ("disc",))
        self.assertEqual(modern.stage_parts(basic, basic), ())

    def test_mega_ex_strip_ends_at_its_badge(self):
        self.assertEqual(stage.strip_end(card(family="ex", name="Mega Clefable")), stage.MEGA_STRIP_END)
        self.assertEqual(stage.strip_end(card(family="ex", name="Alakazam")), stage.STRIP[2])
        self.assertEqual(stage.strip_end(card(name="Mega Man")), stage.STRIP[2])     # regular: no badge

    def test_evolves_face_falls_back_to_gill_sans(self):
        stage.evolves_style.cache_clear()
        try:
            with mock.patch.object(stage.Path, "exists", return_value=False), self.assertWarns(UserWarning):
                self.assertEqual(stage.evolves_style(), stage.EVOLVES_FALLBACK)
        finally:
            stage.evolves_style.cache_clear()

    def test_tab_letters_come_off_a_shifted_scan_without_a_trace(self):
        # three made-up tabs: one silver lozenge shading down and across, each with its own dark, white-rimmed letters
        yy, xx = np.mgrid[:H, :W]
        silver = np.clip(215 - (yy - 20) * 1.8 - np.abs(xx - 70) * 0.3, 0, 255).astype(np.float32)
        lozenge = (yy >= 20) & (yy < 62) & (xx >= 8) & (xx < 130)

        def tab(blocks, shift=(0, 0)):
            rgba = np.zeros((H, W, 4), np.float32)
            rgba[..., :3] = silver[..., None]
            rgba[..., 3] = lozenge * 255
            for x0, x1 in blocks:
                rgba[34:50, x0 - 2:x1 + 2, :3] = 250
                rgba[36:48, x0:x1, :3] = 60
            return np.roll(rgba, shift, (0, 1))

        basic, one, two = tab([(30, 40), (52, 60)]), tab([(26, 34), (70, 80)]), tab([(40, 48), (90, 100)])
        scan = tab([(40, 48), (90, 100)], shift=(1, 2))
        scan[..., 3] = 255
        stage._lozenge.cache_clear()
        try:
            with mock.patch.object(stage, "basic_tab", return_value=basic), \
                    mock.patch.object(stage.meframe, "stage_sprite", side_effect=lambda n: {1: one, 2: two}[n]):
                lettered = stage._lettering(scan)
                out = stage.blank_tab(scan)
                inner = stage._lozenge()
        finally:
            stage._lozenge.cache_clear()
        clean = np.roll(silver, (1, 2), (0, 1))
        self.assertTrue(lettered[42, 95] and lettered[42, 44] and not lettered[42, 34])   # STAGE 2's, where it sits
        self.assertLess(float(np.abs(out[..., 0] - clean)[inner].max()), 6)

    def test_tab_letters_lean_forward(self):
        # the printed BASIC / STAGE letters lean right: an I's stem sits further right at its top than its foot
        font = ImageFont.load_default(size=40 * stage.WORD_SS)
        stem = stage._glyph(font, "I", 1.0, stage.MEGA_SHEAR, (0,))[0] > 0.5
        rows = np.nonzero(stem.any(axis=1))[0]
        middle = [np.nonzero(stem[y])[0].mean() for y in (rows[1], rows[-2])]
        self.assertGreater(middle[0] - middle[1], stage.MEGA_SHEAR * (rows[-2] - rows[1]) * 0.8)

    def test_tab_word_is_centred_as_the_printed_ones(self):
        ink = np.zeros((40, 60), np.float32)
        ink[20:31, 5:25] = 1                                  # ink box: columns 5-24, last row 30
        stage._tab_word.cache_clear()
        try:
            with mock.patch.object(stage, "_mega_font", return_value="face.ttf"), \
                    mock.patch.object(stage, "_word", return_value=([ink] * 3, (5, 24, 30))):
                fill = stage._tab_word("MEGA")[-1][0] > 0.5
        finally:
            stage._tab_word.cache_clear()
        ys, xs = np.nonzero(fill)
        self.assertAlmostEqual((xs.min() + xs.max()) / 2, stage.MEGA_CENTRE, delta=0.5)
        self.assertEqual(ys.max(), stage.MEGA_BOTTOM)

    def test_ex_keeps_its_own_swoosh(self):
        self.assertNotIn("swoosh", stage.EX)
        self.assertIn("swoosh", stage.REGULAR)


class Portrait(unittest.TestCase):
    def test_rank_prefers_main_set_regular_newest(self):
        briefs = [{"id": "sv03.5-064", "localId": "064", "name": "Kadabra"},
                  {"id": "sv03.5-185", "localId": "185", "name": "Kadabra"},          # secret rare
                  {"id": "base1-32", "localId": "32", "name": "Kadabra"},             # no portrait disc
                  {"id": "me01-055", "localId": "055", "name": "Kadabra"},
                  {"id": "sv10-050", "localId": "050", "name": "Kadabra ex"},
                  {"id": "sv08-060", "localId": "060", "name": "Kadabra"}]
        with mock.patch.object(stage, "_official", return_value=165):
            self.assertEqual(stage.rank(briefs, stage.PORTRAIT_ERAS),
                             ["me01-055", "sv08-060", "sv03.5-064", "sv10-050", "sv03.5-185"])

    def test_which_picture(self):
        own, named, found = object(), object(), object()
        with mock.patch.object(stage, "from_card", return_value=named) as from_card, \
                mock.patch.object(stage, "find", return_value=found) as find:
            self.assertIs(stage.portrait(card(), lambda: own), own)
            self.assertIs(stage.portrait(card(portrait="me01-055"), lambda: own), named)
            from_card.assert_called_once_with("me01-055")
            self.assertIs(stage.portrait(card(evolves_from="Ralts"), lambda: own), found)
            find.assert_called_once_with("Ralts", mega=False)
            self.assertIs(stage.portrait(card(), None), found)

    def test_named_card_stage_gives_disc_else_art(self):
        with mock.patch.object(stage, "disc_sprite", return_value="disc"), \
                mock.patch.object(stage, "art_sprite", return_value="art"):
            for card_id, data, want in (("me01-055", {"stage": "Stage1"}, "disc"), ("me01-054", {"stage": "Basic"}, "art"),
                                        ("swsh3-87", {"stage": "Stage1"}, "art")):
                with self.subTest(card_id), mock.patch.object(stage.tcg, "card", return_value=data):
                    self.assertEqual(stage.from_card(card_id), want)

    def test_find_falls_back_to_art_then_nothing(self):
        def search(field, name):
            return [] if field in ("evolveFrom", "stage") else [{"id": "swsh6-54", "localId": "54", "name": name}]
        with mock.patch.object(stage, "_search", side_effect=search), \
                mock.patch.object(stage, "_official", return_value=200), \
                mock.patch.object(stage, "pokeapi_sprite", return_value=None), \
                mock.patch.object(stage, "art_sprite", return_value="art"), self.assertWarns(UserWarning):
            self.assertEqual(stage.find("Galarian Slowpoke"), "art")
        with mock.patch.object(stage, "_search", return_value=[]), \
                mock.patch.object(stage, "pokeapi_sprite", return_value=None), self.assertWarns(UserWarning):
            self.assertIsNone(stage.find("Nobody"))

    def test_source_order(self):
        calls = []
        def source(name, result):
            def f(*a):
                calls.append(name)
                return result
            return f
        with mock.patch.object(stage, "_search", return_value=[]), \
                mock.patch.object(stage, "find_xy_mega", side_effect=source("xy", "xy")), \
                mock.patch.object(stage, "pokeapi_sprite", side_effect=source("artwork", "artwork")):
            self.assertEqual(stage.find("Latias", mega=True), "xy")          # Mega: the XY Mega card first
            self.assertEqual(calls, ["xy"])
            calls.clear()
            self.assertEqual(stage.find("Latias"), "xy")                     # others: after SV / ME evolutions
        with mock.patch.object(stage, "_search", return_value=[]), \
                mock.patch.object(stage, "find_xy_mega", return_value=None), \
                mock.patch.object(stage, "pokeapi_sprite", return_value="artwork"):
            self.assertEqual(stage.find("Latias", mega=True), "artwork")

    def test_mega_card_names(self):
        for name in ("Latias-EX", "M Latias EX", "Latias ex", "latias", "M Latias-EX"):
            self.assertEqual(stage._base_name(name), "latias", name)
        self.assertEqual(stage._base_name("Mr. Mime"), "mr. mime")
        briefs = [{"id": "xy6-59", "localId": "59", "name": "M Latios EX"},
                  {"id": "xy6-102", "localId": "102", "name": "M Latios EX"},
                  {"id": "xy2-13", "localId": "13", "name": "M Charizard EX"}]
        with mock.patch.object(stage, "_search", return_value=briefs), \
                mock.patch.object(stage, "_official", return_value=98), \
                mock.patch.object(stage, "xy_sprite", side_effect=lambda i: i):
            self.assertEqual(stage.find_xy_mega("Latios-EX"), "xy6-59")     # the main-set card, not the secret rare
            self.assertIsNone(stage.find_xy_mega("Latias"))

    def test_xy_portrait_keys_the_gold_off(self):
        scan = np.zeros((H, W, 4), np.float32)
        cx, cy, r = stage.XY_DISC
        yy, xx = np.mgrid[:H, :W]
        d = np.hypot(xx - cx, yy - cy)
        scan[d < r, :3] = (230, 180, 40)                                     # gold disc
        scan[(np.abs(xx - cx) < 16) & (np.abs(yy - cy) < 16), :3] = 20        # figure outline
        scan[(np.abs(xx - cx) < 13) & (np.abs(yy - cy) < 13), :3] = (60, 120, 220)
        with mock.patch.object(stage.tcg, "scan", return_value=scan):
            rgb, alpha = stage.xy_sprite("xy6-59")
        c = alpha.shape[0] // 2
        self.assertGreater(alpha[c, c], 0.9)                                  # the figure stays
        self.assertLess(alpha[c, c + 30], 0.1)                                # the gold goes

    def test_artwork_names(self):
        urls = []
        def fetch_json(url, dest):
            urls.append(url.rsplit("/", 1)[1])
            return {"sprites": {"other": {"official-artwork": {"front_default": None}}}}
        with mock.patch.object(stage.cache, "fetch_json", side_effect=fetch_json):
            for name in ("Galarian Slowpoke", "Farfetch’d", "Mr. Mime", "Nidoran♀", "Latias-EX"):
                self.assertIsNone(stage.pokeapi_sprite(name))
        self.assertEqual(urls, ["slowpoke-galar", "farfetchd", "mr-mime", "nidoran-f", "latias"])

    def test_portrait_stays_in_the_disc(self):
        canvas = np.zeros((H, W, 4), np.float32)
        stage.put_portrait(canvas, np.full((100, 100, 3), 255, np.float32), np.ones((100, 100), np.float32))
        cx, cy, r = stage.meframe.PORTRAIT_DISC
        self.assertEqual(canvas[int(cy), int(cx), 0], 255)
        self.assertEqual(canvas[int(cy), int(cx + r + 2), 0], 0)


class Plates(unittest.TestCase):
    def test_retab_takes_the_tab_value_around_new_letters(self):
        img = np.full((H, W), 180, np.float32)
        img[40:50, 30:60:6] = 0                       # old letters: strokes
        img[40:50, 31:60:6] = 0
        tab = np.zeros((H, W), np.float32)
        tab[34:56, 24:66] = 1                         # where the new tab was pasted
        out = foil.retab(img, {"stage_tab": tab})
        self.assertEqual(out[45, 30], 180)
        self.assertEqual(out[45, 33], 180)
        self.assertIs(foil.retab(img, {}), img)


if __name__ == "__main__":
    unittest.main()
