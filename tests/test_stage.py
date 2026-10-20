"""Stage editing: Basic / Stage 1 / Stage 2, "Evolves from" and the evolution portrait. Pure logic: no network,
model, fonts or extract (TCGdex and scans are mocked). Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import numpy as np

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
            find.assert_called_once_with("Ralts")
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
            return [] if field == "evolveFrom" else [{"id": "swsh6-54", "localId": "54", "name": name}]
        with mock.patch.object(stage, "_search", side_effect=search), \
                mock.patch.object(stage, "_official", return_value=200), \
                mock.patch.object(stage, "art_sprite", return_value="art"), self.assertWarns(UserWarning):
            self.assertEqual(stage.find("Galarian Slowpoke"), "art")
        with mock.patch.object(stage, "_search", return_value=[]), self.assertWarns(UserWarning):
            self.assertIsNone(stage.find("Nobody"))

    def test_portrait_stays_in_the_disc(self):
        canvas = np.zeros((H, W, 4), np.float32)
        stage.put_portrait(canvas, np.full((100, 100, 3), 255, np.float32), np.ones((100, 100), np.float32))
        cx, cy, r = stage.meframe.PORTRAIT_DISC
        self.assertEqual(canvas[int(cy), int(cx), 0], 255)
        self.assertEqual(canvas[int(cy), int(cx + r + 2), 0], 0)


class Plates(unittest.TestCase):
    def test_retab_takes_the_tab_value_around_new_letters(self):
        img = np.full((H, W), 180, np.float32)
        img[40:50, 30:60] = 0                         # old letters
        tab = np.zeros((H, W), np.float32)
        tab[38:52, 28:62] = 1                         # where the new letters were pasted
        out = foil.retab(img, {"stage_tab": tab})
        self.assertEqual(out[45, 45], 180)
        self.assertIs(foil.retab(img, {}), img)


if __name__ == "__main__":
    unittest.main()
