"""Mega ex type change (megaretype.py), Energy and damage set again where the card prints them (inplace.py), "Follow
the type" (follow.py) and the custom-set check on the simulator's Energy types (export.unsupported_types). The Composing tests
need a TCG Live extract (SVRESTYLE_TCGLIVE) and Gill Sans (SVRESTYLE_FONTS); the first builds the Mega ex assets
(about four minutes, then cached). Run: python -m unittest discover tests"""
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from svrestyle import inplace, edit, export, follow, megaretype

W, H = 733, 1024
INFERNO = {"cost": ["Fire", "Fire"], "name": "Inferno X", "damage": "90×",
           "effect": "Discard any amount of {R} Energy from among your Pokémon."}


def card(**over):
    c = {"id": "me02-013", "family": "ex", "type": "Fire", "prefix": "", "name": "Mega Charizard X", "hp": "360",
         "stage": 2, "evolves_from": "Charmeleon", "portrait": "", "abilities": [], "attacks": [dict(INFERNO)],
         "weakness": {"type": "Water", "value": "×2"}, "resistance": None, "retreat": 2, "retype": True}
    c.update(over)
    return c


def attacks(*costs, effect=INFERNO["effect"]):
    return [{**INFERNO, "cost": list(cost), "effect": effect} for cost in costs]


class InPlace(unittest.TestCase):
    def test_energy_alone(self):
        after = card(attacks=attacks(["Water", "Water"], effect=INFERNO["effect"].replace("{R}", "{W}")))
        self.assertEqual(inplace.changes(card(), after),
                         {"costs": {"Fire": "Water"}, "text": {"Fire": "Water"}, "damage": {}})
        self.assertEqual(inplace.changes(card(), card(attacks=attacks(["Fairy", "Fairy"]))),
                         {"costs": {"Fire": "Fairy"}, "text": {}, "damage": {}})

    def test_damage(self):
        after = card(attacks=[{**INFERNO, "damage": "120×"}])
        self.assertEqual(inplace.changes(card(), after), {"costs": {}, "text": {}, "damage": {0: ("90×", "120×")}})
        both = card(attacks=[{**INFERNO, "cost": ["Water", "Water"], "damage": "30"}])
        self.assertEqual(inplace.changes(card(), both),
                         {"costs": {"Fire": "Water"}, "text": {}, "damage": {0: ("90×", "30")}})

    def test_other_edits_are_not(self):
        for after in (card(attacks=[{**INFERNO, "name": "Blue Inferno"}]),             # a name
                      card(attacks=[{**INFERNO, "damage": ""}]),                        # a damage taken off
                      card(attacks=attacks(["Fire", "Fire", "Fire"])),                  # a cost added
                      card(attacks=attacks(["Water", "Grass"])),                        # one type into two
                      card(attacks=attacks(["Fire", "Fire"], effect="Discard {R} and {R}.")),   # other words
                      card(attacks=[]),
                      card()):                                                          # no change at all
            with self.subTest(after=after["attacks"]):
                self.assertIsNone(inplace.changes(card(), after))
        added = card(attacks=[{**INFERNO, "damage": ""}])
        self.assertIsNone(inplace.changes(added, card()))                              # a damage added

    def test_other_symbols_stay(self):
        before = card(attacks=attacks(["Fire", "Colorless"], effect="Attach {R} or {C} Energy."))
        after = card(attacks=attacks(["Water", "Colorless"], effect="Attach {W} or {C} Energy."))
        self.assertEqual(inplace.changes(before, after)["text"], {"Fire": "Water"})

    def test_damages_are_found_in_order(self):
        # a "30" also matches the end of a "130" below it: the attacks' order down the card tells them apart
        from svrestyle import compose
        figures = {"30": np.zeros((H, W), np.float32), "130": np.zeros((H, W), np.float32)}
        figures["30"][590:600, 640:670] = 1
        figures["130"][590:600, 610:670] = 1
        figures["130"][590:600, 625:628] = 0
        cov = np.zeros((H, W), np.float32)
        cov[600:610, 641:671] = 1                                          # attack 1: "30", 10 down, 1 right
        cov[700:710, 611:671] = 1                                          # attack 2: "130", 110 down
        cov[700:710, 626:629] = 0
        with mock.patch.object(compose, "damage_figures", side_effect=lambda text, base: figures[text]):
            self.assertEqual(inplace.find_damages(cov, (30, 524, 703, 870), ["30", "130"]), [(1, 10), (1, 110)])
            self.assertIsNone(inplace.find_damages(cov, (30, 524, 703, 870), ["30", "130", "30"]))   # one not printed


class Follow(unittest.TestCase):
    RULES = {"Water": ("Lightning", None), "Darkness": ("Grass", None), "Dragon": (None, None),
             "Grass": ("Fire", None), "Psychic": ("Darkness", ("Fighting", "-30")),
             "Fairy": ("Metal", ("Darkness", "-30"))}

    def setUp(self):
        p = mock.patch.object(follow, "rules", return_value=self.RULES)
        p.start()
        self.addCleanup(p.stop)

    def followed(self, original, new_type, **over):
        edited = edit.apply(original, {"type": new_type, **over}) if over else dict(original, type=new_type)
        return edit.apply(edited, follow.follow(original, edited))

    def test_costs_text_and_sides_follow(self):
        out = self.followed(card(), "Psychic")
        self.assertEqual(out["attacks"][0]["cost"], ["Psychic", "Psychic"])
        self.assertIn("{P}", out["attacks"][0]["effect"])
        self.assertEqual(out["weakness"], {"type": "Darkness", "value": "×2"})
        self.assertEqual(out["resistance"], {"type": "Fighting", "value": "-30"})

    def test_dragon(self):
        to = self.followed(card(), "Dragon")                  # a card made Dragon keeps its costs, prints no weakness
        self.assertEqual((to["attacks"][0]["cost"], to["weakness"]), (["Fire", "Fire"], None))
        self.assertIn("{R}", to["attacks"][0]["effect"])
        dragon = card(type="Dragon", attacks=attacks(["Water", "Lightning", "Colorless"]))
        back = self.followed(dragon, "Grass")                 # every coloured Energy becomes the new type's
        self.assertEqual(back["attacks"][0]["cost"], ["Grass", "Grass", "Colorless"])

    def test_colorless_costs_take_any_energy(self):
        colorless = card(type="Colorless", attacks=attacks(["Colorless", "Colorless"], effect="Discard a {C} Energy."))
        out = self.followed(colorless, "Water")                # its costs stay any Energy; its text stays
        self.assertEqual(out["attacks"][0]["cost"], ["Colorless", "Colorless"])
        self.assertIn("{C}", out["attacks"][0]["effect"])
        made = self.followed(card(), "Colorless")               # a card made Colorless pays Colorless for its own
        self.assertEqual(made["attacks"][0]["cost"], ["Colorless", "Colorless"])

    def test_each_follow_starts_from_the_print(self):
        water = self.followed(card(), "Water")
        again = edit.apply(water, {"type": "Grass"})
        again = edit.apply(again, follow.follow(card(), again))
        self.assertEqual(again["attacks"][0]["cost"], ["Grass", "Grass"])
        self.assertIn("{G}", again["attacks"][0]["effect"])

    def test_costs_the_user_changed_are_mapped_as_they_stand(self):
        mine = card(attacks=attacks(["Fire", "Fire"]) + [{"cost": ["Fire", "Colorless"], "name": "Ember",
                                                          "damage": "30", "effect": ""}])
        out = self.followed(card(), "Fairy", attacks=mine["attacks"])
        self.assertEqual([a["cost"] for a in out["attacks"]], [["Fairy", "Fairy"], ["Fairy", "Colorless"]])
        self.assertEqual(out["resistance"], {"type": "Darkness", "value": "-30"})


class Plate(unittest.TestCase):
    def setUp(self):
        self.img = np.full((H, W), 200, np.float32)
        self.words = np.zeros((H, W), np.float32)
        self.words[650:670, 100:300] = 1

    def test_to_darkness_cuts_the_words_out(self):
        out = megaretype.plate(self.img, "Fire", "Darkness", self.words)
        self.assertEqual(out[660, 200], megaretype.CUT)
        self.assertEqual(out[700, 200], 200)

    def test_from_darkness_closes_them(self):
        img = self.img.copy()
        img[650:670, 100:300] = megaretype.CUT                 # Darkness's white words, cut out
        out = megaretype.plate(img, "Darkness", "Water", self.words)
        self.assertEqual(out[660, 200], 200)                   # the panel's value

    def test_same_ink_keeps_the_plate(self):
        np.testing.assert_array_equal(megaretype.plate(self.img, "Fire", "Fairy", self.words), self.img)


class Frames(unittest.TestCase):
    def setUp(self):
        megaretype._frames.clear()
        self.addCleanup(megaretype._frames.clear)

    def test_frame_by_rarity(self):
        for rarity, want in (("Double rare", "dr"), ("Ultra Rare", "full"), ("Special illustration rare", "full"),
                             ("Mega Hyper Rare", "gold"), ("Illustration rare", None)):
            megaretype._frames.clear()
            with self.subTest(rarity=rarity), mock.patch.object(megaretype.tcg, "card", return_value={"rarity": rarity}):
                self.assertEqual(megaretype.frame_of("me02-999", card(id="me02-999")), want)

    def test_mega_and_sv_ex_with_the_extract(self):
        with mock.patch.object(megaretype.tcg, "card", return_value={"rarity": "Double rare"}), \
                mock.patch.object(megaretype, "tera", return_value=False):
            with mock.patch.object(megaretype.tcg, "TCGLIVE", None):
                self.assertFalse(megaretype.supports("me02-013", card()))
            with mock.patch.object(megaretype.tcg, "TCGLIVE", "extract"):
                self.assertTrue(megaretype.supports("me02-013", card()))
                self.assertTrue(megaretype.supports("sv05-104", card(id="sv05-104", name="Gengar")))
                self.assertFalse(megaretype.supports("me01-054", card(family="regular", name="Mega Abra")))
        megaretype._frames.clear()
        with mock.patch.object(megaretype.tcg, "card", return_value={"rarity": "Double rare"}), \
                mock.patch.object(megaretype, "tera", return_value=True), \
                mock.patch.object(megaretype.tcg, "TCGLIVE", "extract"):
            self.assertFalse(megaretype.supports("sv04-058", card(id="sv04-058", name="Mewtwo")))   # a Tera ex

    def test_sources_are_double_rares_of_their_type(self):
        self.assertEqual(set(megaretype.TYPES) - set(megaretype.SOURCES), {"Fairy"})
        self.assertEqual(megaretype.DERIVED, {"Fairy": "Psychic"})
        ids = [c for ids in megaretype.SOURCES.values() for c in ids]
        self.assertEqual(len(ids), len(set(ids)))
        for lay, refs in megaretype.LAYOUT_REFS.items():
            self.assertTrue(set(refs) <= set(ids), lay)


class Band(unittest.TestCase):
    """An SV ex's band below the panel: what its old type's colour explains moves to the new type's colour."""
    COLOURS = {"Lightning": (255.0, 216.0, 49.0), "Darkness": (5.0, 37.0, 33.0), "Psychic": (186.0, 119.0, 178.0)}
    SPREADS = {"Lightning": 20.0, "Darkness": 45.0, "Psychic": 17.0}

    def setUp(self):
        self.zone = megaretype.zone(megaretype.SV.glitter)
        fields = {t: (np.where(self.zone[..., None], np.array(c, np.float32), 0).astype(np.float32), self.SPREADS[t])
                  for t, c in self.COLOURS.items()}
        self.fixed = np.zeros((H, W), bool)
        self.fixed[876:879, 30:700] = True                                   # a rule
        for p in (mock.patch.object(megaretype, "band_field", side_effect=lambda t, fam=None: fields[t]),
                  mock.patch.object(megaretype, "band_fixed", return_value=self.fixed)):
            p.start()
            self.addCleanup(p.stop)
        self.card = card(id="sv03-999", name="Zapdos", type="Lightning", weakness=None, retreat=0)

    def band(self, old_t, new_t, c):
        return megaretype._band(c, c.copy(), self.card, old_t, new_t, megaretype.SV)

    def test_band_takes_the_new_colour_and_keeps_its_grain_and_the_pieces(self):
        c = np.zeros((H, W, 3), np.float32)
        c[:] = self.COLOURS["Lightning"]
        c[940, 400] += (0.0, -10.0, 10.0)                                     # grain
        c[950, 400] = (40.0, 60.0, 160.0)                                     # art reaching over the band
        c[877, 400] = (200.0, 200.0, 205.0)                                   # the rule
        out = self.band("Lightning", "Psychic", c)
        np.testing.assert_allclose(out[930, 400], self.COLOURS["Psychic"], atol=0.01)
        k = self.SPREADS["Psychic"] / self.SPREADS["Lightning"]               # grain scaled to the new type's
        np.testing.assert_allclose(out[940, 400] - out[930, 400], (0.0, -10.0 * k, 10.0 * k), atol=0.01)
        np.testing.assert_array_equal(out[950, 400], c[950, 400])
        np.testing.assert_array_equal(out[877, 400], c[877, 400])
        np.testing.assert_array_equal(out[600, 400], c[600, 400])           # the panel is not the band

    def test_grain_is_never_grown(self):
        c = np.zeros((H, W, 3), np.float32)
        c[:] = self.COLOURS["Psychic"]
        c[940, 400] += (0.0, 10.0, 0.0)
        out = self.band("Psychic", "Darkness", c)
        np.testing.assert_allclose(out[940, 400] - out[930, 400], (0.0, 10.0, 0.0), atol=0.01)

    def test_a_dark_bands_darker_shades_are_band(self):
        c = np.zeros((H, W, 3), np.float32)
        c[:] = self.COLOURS["Darkness"]
        c[940, 400] = (60.0, 0.0, 0.0)                                        # far off its colour, but darker
        c[945, 400] = (250.0, 120.0, 40.0)                                    # a lighter splash of art
        out = self.band("Darkness", "Lightning", c)
        self.assertGreater(out[940, 400, 0], 200)
        np.testing.assert_array_equal(out[945, 400], c[945, 400])

    def test_the_footers_print_stays_on_a_dark_band(self):
        c = np.zeros((H, W, 3), np.float32)
        c[:] = self.COLOURS["Darkness"]
        c[930:946, 150:170] = 250.0                                           # a letter's white keyline
        c[933:943, 153:167] = 20.0                                            # the letter: dark like the band
        out = self.band("Darkness", "Lightning", c)
        np.testing.assert_array_equal(out[938, 160], c[938, 160])
        np.testing.assert_array_equal(out[931, 160], c[931, 160])
        self.assertGreater(out[938, 200, 0], 200)                             # the band beside it moves


class SimulatorTypes(unittest.TestCase):
    SCHEMA = "export const ENERGY_TYPES = Object.freeze([\n  'Grass',\n  'Fire',\n  'Psychic',\n  'Colorless',\n]);\n"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name, "sim", "client", "src", "assets", "custom-sets")
        self.root.mkdir(parents=True)

    def schema(self, text):
        path = Path(self.tmp.name, "sim", *export.SCHEMA)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def write(self, the_card):
        render = mock.Mock(side_effect=AssertionError("nothing is rendered before the check"))
        return export.write_set_dir([{"id": "sm5-94", "edits": {}}], "", "HB", self.root,
                                    read=lambda i: ("me", the_card), render=render, apply=edit.apply,
                                    changelog=lambda *a: "")

    def test_reads_the_schemas_types(self):
        self.assertIsNone(export.schema_types(self.root))
        self.schema(self.SCHEMA)
        self.assertEqual(export.schema_types(self.root), {"Grass", "Fire", "Psychic", "Colorless"})

    def test_a_type_the_simulator_does_not_take_is_refused(self):
        self.schema(self.SCHEMA)
        fairy = card(family="regular", name="Tapu Lele", type="Fairy", retype=None, stage=0,
                     attacks=attacks(["Fairy"]), weakness={"type": "Metal", "value": "×2"})
        with self.assertRaisesRegex(ValueError, "does not take Fairy, Metal"):
            self.write(fairy)

    def test_a_checkout_without_a_schema_is_not_checked(self):
        self.assertEqual(export.unsupported_types([{"id": "x"}], self.root, read=lambda i: ("me", card(type="Fairy")),
                                                  apply=edit.apply), [])


@unittest.skipUnless(os.environ.get("SVRESTYLE_FONTS") and os.environ.get("SVRESTYLE_TCGLIVE"),
                     "needs Gill Sans (SVRESTYLE_FONTS) and a TCG Live extract (SVRESTYLE_TCGLIVE)")
class Composing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from svrestyle import modern, typeset
        typeset.configure(os.environ["SVRESTYLE_FONTS"])
        cls.modern = modern
        cls.printed = modern.read_card("me02-013")

    def recolour(self, rgba, printed, new_type, plate=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return megaretype.recolour(rgba, printed, new_type, plate)

    def panel(self):
        x0, y0, x1, y1 = megaretype.PANEL
        return np.s_[y0 + 40:y1 - 10, x0 + 200:x1 - 20]        # right of the costs, over the faded art

    def test_the_panel_takes_the_types_tint(self):
        from svrestyle import tcg
        scan = tcg.scan("me02-013")
        water = self.recolour(scan, self.printed, "Water")
        ratio = np.median(water[self.panel()][..., :3] / np.maximum(scan[self.panel()][..., :3], 1), axis=(0, 1))
        want = megaretype.params("Water")[0] / megaretype.params("Fire")[0]
        np.testing.assert_allclose(ratio, want, rtol=0.08)

    def test_a_round_trip_comes_back(self):
        from svrestyle import foil, tcg
        scan = tcg.scan("me02-013")
        there = np.round(self.recolour(scan, self.printed, "Water"))
        back = self.recolour(there, dict(self.printed, type="Water"), "Fire",
                             plate=foil.plate(foil.printings("me02-013")["base"]))
        d = np.abs(back[..., :3] - scan[..., :3]).max(-1)
        self.assertLess(np.median(d[self.panel()]), 3)
        self.assertLess(np.median(d[5:95, 150:450]), 8)        # the name bar

    def test_a_full_art_changes_its_icon_alone(self):
        from svrestyle import tcg
        printed = self.modern.read_card("me02-109")
        scan = tcg.scan("me02-109")
        out = self.recolour(scan, printed, "Dragon")
        changed = np.abs(out[..., :3] - scan[..., :3]).max(-1) > 2
        yy, xx = np.mgrid[:H, :W]
        icon = np.hypot(xx - megaretype.ICON_C[0], yy - megaretype.ICON_C[1]) < megaretype.ICON_DISC[2] + 1
        self.assertTrue(changed[icon].any())
        self.assertFalse(changed[~icon].any())

    def test_an_sv_ex_takes_the_type_on_its_band_and_keeps_its_pieces(self):
        from svrestyle import tcg
        fam = megaretype.SV
        printed = self.modern.read_card("sv03.5-065")                       # Alakazam ex, Psychic
        scan = tcg.scan("sv03.5-065")
        fire = self.recolour(scan, printed, "Fire")
        d = np.abs(fire[..., :3] - scan[..., :3]).max(-1)
        fixed = megaretype.band_fixed(fam)
        self.assertLess(np.median(d[fixed]), 1)                              # the rules, the rule box: as printed
        t_old, t_new = megaretype.band_field("Psychic", fam)[0], megaretype.band_field("Fire", fam)[0]
        band = megaretype.zone(fam.glitter) & ~fixed & (np.abs(scan[..., :3] - t_old).max(-1) < 10)
        moved = np.median(fire[band][:, :3] - scan[band][:, :3], axis=0)
        np.testing.assert_allclose(moved, np.median((t_new - t_old)[band], axis=0), atol=12)
        dark = self.recolour(scan, printed, "Darkness")
        x0, y0, x1, y1 = megaretype.ROW_TEXT                                  # "weakness": white on Darkness
        words = np.zeros((H, W), bool)
        words[y0:y1, x0:x0 + 70] = True
        words &= scan[..., :3].max(-1) < 80
        self.assertGreater(np.median(dark[words][:, :3]), 180)
        x0, y0, x1, y1 = megaretype.FOOT_PRINT                                # the footer: as printed on every type
        foot = np.zeros((H, W), bool)
        foot[y0:y1, x0:x1] = True
        foot &= scan[..., :3].max(-1) < 60
        self.assertLess(np.median(np.abs(dark[foot][:, :3] - scan[foot][:, :3]).max(-1)), 3)

    def test_energy_follows_in_place(self):
        edited = edit.apply(self.printed, {"type": "Darkness"})
        edited = edit.apply(edited, follow.follow(self.printed, edited))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img, report = self.modern.compose_card("me02-013", edited, self.printed)
        self.assertEqual(report["edited"][:2], ["type", "moves"])
        recoloured, _ = self.modern.print_of("me02-013", self.printed, edited, record=False)
        words = np.s_[700:760, 230:640]                         # the attack's text: kept as printed, not set again
        self.assertLess(np.abs(img[words][..., :3] - recoloured[words][..., :3]).max(-1).mean(), 4)
        cx, cy = inplace.cost_icons(img[..., :3], self.printed, "Darkness", 1)[0]
        self.assertTrue(cx < 100 and 600 < cy < 720)            # the first cost is Darkness's icon now
