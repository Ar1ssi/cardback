"""Mega Evolution regular layout: pure logic, checked against what real me1 cards print. No network or model; the text breaks need the fonts (SVRESTYLE_FONTS).
Run: python -m unittest discover tests"""
import os
import unittest
from unittest import mock

import numpy as np

from svrestyle import compose, mecompose, tcg
from svrestyle import typeset as ts

PITCH = 29


def laid(*blocks):
    """'A3' = ability with 3 lines, 'E2' = attack with 2 lines of text, 'n' = attack without text."""
    out = []
    for b in blocks:
        kind = "ability" if b[0] == "A" else "attack"
        lines = [None] * (int(b[1:]) if len(b) > 1 else 0)
        out.append((kind, {}, lines))
    return out


def icon_tops(blocks):
    caps, fits = mecompose.me_layout(laid(*blocks), PITCH)
    return [c - (5 if b[0] == "A" else 4) for c, b in zip(caps, blocks)], fits


class Layout(unittest.TestCase):
    # (blocks, icon-row / badge tops measured on the me1 scan, card)
    CASES = [
        (["E1"], [638], "Abra"),
        (["n"], [659], "Magnemite"),
        (["n", "n"], [618, 713], "Vulpix"),
        (["E1", "n"], [618, 742], "Seedot"),
        (["E2", "n"], [597, 751], "Kirlia"),
        (["E3", "E2"], [577, 730], "Exeggutor"),
        (["E3", "E1"], [598, 750], "Inteleon"),
        (["A3", "n"], [555, 740], "Kadabra"),
        (["A3", "E2"], [555, 711], "Alakazam"),
        (["A4", "E1"], [555, 740], "Hariyama"),
    ]

    def test_matches_real_cards(self):
        for blocks, measured, name in self.CASES:
            with self.subTest(name):
                tops, fits = icon_tops(blocks)
                self.assertTrue(fits)
                for got, want in zip(tops, measured):
                    self.assertLessEqual(abs(got - want), 3, f"{name}: {tops} vs {measured}")

    def test_too_much_text_does_not_fit(self):
        self.assertFalse(icon_tops(["A8", "E6"])[1])


class EnergyIcons(unittest.TestCase):
    def test_every_type_has_a_mega_evolution_icon(self):
        from svrestyle import edit, meframe
        self.assertEqual(set(edit.ENERGY), set(meframe.TYPES))
        # cost icons come from real attack costs; only Dragon (never a cost) falls back to its name-bar icon, and
        # Fairy (never printed) is made on Psychic's (fairy.py)
        self.assertEqual(set(meframe.TYPES) - set(meframe.COST_ICONS), {"Dragon", "Fairy"})
        self.assertEqual(meframe.DERIVED, {"Fairy": "Psychic"})
        for t, (card_id, x, y, r) in meframe.COST_ICONS.items():
            self.assertTrue(card_id.startswith("me"), t)
            self.assertAlmostEqual(x, 71, delta=1)          # the first cost slot
            self.assertAlmostEqual(r, 18.5, delta=0.5)


class Footer(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(mecompose.card_number("SV33", 94), "SV33/SV94")
        self.assertEqual(mecompose.card_number("19", 214), "019/214")
        self.assertEqual(mecompose.card_number("088", 189), "088/189")

    def test_promo_and_subset_numbers(self):
        # dpp / np promos print the number alone; bw11's Radiant Collection counts its own 25 (TCGdex set data)
        self.assertEqual(mecompose.card_number("DP05", 56, {"name": "DP Black Star Promos"}), "DP05")
        self.assertEqual(mecompose.card_number("28", 40, {"name": "Nintendo Black Star Promos"}), "28")
        bw11 = {"name": "Legendary Treasures", "cards": [{"localId": str(n)} for n in range(1, 114)] +
                [{"localId": f"RC{n}"} for n in range(1, 26)]}
        self.assertEqual(mecompose.card_number("RC20", 113, bw11), "RC20/RC25")
        self.assertEqual(mecompose.card_number("20", 113, bw11), "020/113")
        # pl1-SH4 and col1-SL1 print the number alone (scans): the SH run spans sets, 3 a set
        pl1 = {"name": "Platinum", "serie": {"id": "pl"}, "cards": [{"localId": f"SH{n}"} for n in (4, 5, 6)]}
        self.assertEqual(mecompose.card_number("SH4", 127, pl1), "SH4")
        self.assertEqual(mecompose.card_number("SL1", 95, {"name": "Call of Legends", "serie": {"id": "col"}}), "SL1")
        self.assertEqual(mecompose.card_number("12", 127, pl1), "012/127")

    def test_rarity(self):
        self.assertEqual(mecompose.rarity_symbol("Common"), "common")
        self.assertEqual(mecompose.rarity_symbol("Uncommon"), "uncommon")
        for r in ("Rare", "Holo Rare", "Shiny rare", None):
            self.assertEqual(mecompose.rarity_symbol(r), "rare")

    def test_dex_line_as_printed(self):
        # me1 Vulpix prints HT: 2' and WT: 21.8 lbs.; Kadabra HT: 4'3" and WT: 124.6 lbs.
        self.assertEqual(mecompose.dex_text(37, "Fox Pokémon", 6, 99),
                         "NO. 0037   Fox Pokémon   HT: 2'   WT: 21.8 lbs.")
        self.assertEqual(mecompose.dex_text(64, "Psi Pokémon", 13, 565),
                         "NO. 0064   Psi Pokémon   HT: 4'3\"   WT: 124.6 lbs.")


SET = {"cardCount": {"official": 189}, "abbreviation": {"official": "DAA"}, "releaseDate": "2020-08-14"}


def fake_card(**over):
    base = {"id": "swsh3-88", "name": "Tyranitar", "category": "Pokemon", "stage": "Stage2", "types": ["Fighting"],
            "hp": 180, "localId": "88", "set": {"id": "swsh3"}, "evolveFrom": "Pupitar", "rarity": "Holo Rare",
            "attacks": [{"name": "Bedrock Breaker", "effect": "Discard a Stadium in play.", "damage": 120}],
            "weaknesses": [{"type": "Grass", "value": "×2"}], "retreat": 4, "description": "Its body can't be harmed."}
    base.update(over)
    return base


class ReadCard(unittest.TestCase):
    def read(self, **over):
        with mock.patch.object(tcg, "card", return_value=fake_card(**over)), \
             mock.patch.object(tcg, "set_info", return_value=SET), \
             mock.patch.object(mecompose, "species_line", return_value=""):
            return mecompose.read_card("swsh3-88")

    def test_fields(self):
        c = self.read()
        self.assertEqual((c["stage"], c["evolves_from"], c["number"], c["rarity"], c["set_code"], c["old_stage"]),
                         (2, "Pupitar", "088/189", "rare", "DAA", True))
        self.assertIn("’", c["flavor"])

    def test_rejects_v_and_ex(self):
        for name in ("Galarian Moltres V", "Charizard ex", "Pikachu VMAX"):
            with self.subTest(name), self.assertRaises(SystemExit):
                self.read(name=name, stage="Basic")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.environ.get("SVRESTYLE_FONTS"), "needs the Gill Sans faces (SVRESTYLE_FONTS)")
class TextBreaks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ts.configure(os.environ["SVRESTYLE_FONTS"])

    def test_number_goes_with_the_next_word(self):
        # me01-005 Exeggutor, Guard Press (TCGdex text): the print breaks "...this Pokémon takes / 30 less damage"
        effect = ("During your opponent’s next turn, this Pokémon takes 30 less damage from attacks "
                  "(after applying Weakness and Resistance).")
        card = {"id": "me01-005", "abilities": [], "attacks": [{"name": "Guard Press", "cost": ["Grass"],
                                                                 "damage": "30", "effect": effect}]}
        seen = []

        def layout(laid, pitch):
            seen.append(laid)
            return [600], True

        canvas = np.zeros((1024, 733, 4), np.float32)
        compose.text_block(canvas, card, np.zeros(3, np.float32), None, layout=layout,
                           fit_tracking=mecompose.BODY_FIT_TRACK, min_tracking=mecompose.BODY_MIN_TRACK)
        first = "".join(t for t, _ in seen[0][0][2][0][0])
        self.assertTrue(first.endswith("takes"), first)
