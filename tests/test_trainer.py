"""Trainer conversion: card text, kinds, art windows and the editor's Trainer fields. No network or scans: TCGdex
answers are mocked with the cards' own TCGdex text (ids named per test). The Gill Sans checks run when SVRESTYLE_FONTS
is set.
Run: python -m unittest discover tests"""
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from svrestyle import badges, cli, edit, gui, oldwindow, trainer


def tcgdex(**over):
    """A TCGdex Trainer record, ex1-83 Lady Outing by default."""
    base = {"id": "ex1-83", "localId": "83", "category": "Trainer", "name": "Lady Outing", "trainerType": "Supporter",
            "rarity": "Uncommon", "illustrator": "Ken Sugimori", "set": {"id": "ex1"},
            "effect": "You can play only one Supporter card each turn. When you play this card, put it next to your "
                      "Active Pokémon. When your turn ends, discard this card. Search your deck for up to 3 different "
                      "types of basic Energy cards, show them to your opponent, and put them into your hand. Shuffle "
                      "your deck afterward."}
    base.update(over)
    return base


SET = {"cardCount": {"official": 109}, "releaseDate": "2003-06-18", "abbreviation": {"official": "RS"}}


def read(record, s=SET):
    with mock.patch.object(trainer.tcg, "card", return_value=record), \
            mock.patch.object(trainer.tcg, "set_info", return_value=s):
        return trainer.read_card(record["id"])


class Symbols(unittest.TestCase):
    def test_types_become_symbols_where_newer_cards_print_them(self):
        # swsh1-156 Air Balloon: "...is ColorlessColorless less."
        self.assertEqual(trainer.symbols("The Retreat Cost of the Pokémon this card is attached to is "
                                         "ColorlessColorless less."),
                         "The Retreat Cost of the Pokémon this card is attached to is {C}{C} less.")
        # neo4-92 Broken Ground Gym
        self.assertEqual(trainer.symbols("Each player pays Colorless more to retreat"), "Each player pays {C} more to retreat")
        self.assertEqual(trainer.symbols("a Fire or Water Energy card"), "a {R} or {W} Energy card")
        self.assertEqual(trainer.symbols("1 of your Grass Pokémon"), "1 of your {G} Pokémon")

    def test_a_list_with_a_serial_comma(self):
        # ecard3-129 Miracle Sphere α
        self.assertEqual(trainer.symbols("1 of your Evolved Fire, Lightning, or Fighting Pokémon"),
                         "1 of your Evolved {R}, {L}, or {F} Pokémon")
        self.assertEqual(trainer.symbols("Grass, Fire or Water Pokémon"), "{G}, {R} or {W} Pokémon")

    def test_a_word_between_type_and_pokemon(self):
        # ecard3-129's attack (its scan prints symbols); ex16-71 Mt. Moon
        self.assertEqual(trainer.symbols("has Fire and Lightning basic Energy cards attached"),
                         "has {R} and {L} basic Energy cards attached")
        self.assertEqual(trainer.symbols("Each player's Colorless Evolved Pokémon, Darkness Evolved Pokémon"),
                         "Each player's {C} Evolved Pokémon, {D} Evolved Pokémon")

    def test_a_pokemons_type(self):
        # ex11-94 Holon Ruins; ecard3-122 Crystal Shard (sv05-118 prints "it is {F} and {M} type")
        self.assertEqual(trainer.symbols("both their usual Energy type and Metal type"),
                         "both their usual Energy type and {M} type")
        self.assertEqual(trainer.symbols("that Pokémon’s type (color) is Colorless."),
                         "that Pokémon’s type (color) is {C}.")

    def test_other_words_stay(self):
        for text in ("Fighting Fury Belt", "the Water's edge", "Psychic attacks", "a Double Colorless Energy card"):
            self.assertEqual(trainer.symbols(text), text)


class OldRules(unittest.TestCase):
    def test_supporter_rule_goes(self):
        # ex1-83 Lady Outing
        self.assertEqual(trainer.effect_paragraphs(tcgdex()["effect"]),
                         ["Search your deck for up to 3 different types of basic Energy cards, show them to your "
                          "opponent, and put them into your hand. Shuffle your deck afterward."])

    def test_stadium_rule_goes(self):
        # dp1-114 Speed Stadium
        raw = ("This card stays in play when you play it. Discard this card if another Stadium card comes into play. "
               "If another card with the same name is in play, you can't play this card. Once during each player's "
               "turn, the player may flip a coin until he or she gets tails. For each heads, that player draws a card.")
        self.assertEqual(trainer.effect_paragraphs(raw),
                         ["Once during each player’s turn, the player may flip a coin until he or she gets tails. For "
                          "each heads, that player draws a card."])

    def test_tool_attach_and_knock_out_reminder_go(self):
        # pl4-83 Bench Shield
        raw = ("Attach Bench Shield to 1 of your Pokémon that doesn't already have a Pokémon Tool attached to it. If "
               "that Pokémon is Knocked Out, discard this card. As long as the Pokémon this card is attached to is on "
               "your Bench, prevent all damage done to that Pokémon by attacks (both yours and your opponent's).")
        self.assertEqual(trainer.effect_paragraphs(raw)[0][:40], "As long as the Pokémon this card is atta")
        # neo4-93: the name holds a full stop
        raw = ("Attach EXP.ALL to 1 of your Pokémon that doesn't have a Pokémon Tool attached to it. During your "
               "opponent's turn, if your Active Pokémon would be Knocked Out")
        self.assertTrue(trainer.effect_paragraphs(raw)[0].startswith("During your opponent’s turn"))

    def test_a_restricting_attach_stays(self):
        # ex10-83 Energy Root: who it can attach to is the card's own rule
        raw = ("Attach Energy Root to 1 of your Pokémon (excluding Pokémon-ex and Pokémon that has Dark or an owner "
               "in its name) that doesn't already have a Pokémon Tool attached to it.")
        self.assertEqual(trainer.effect_paragraphs(raw), [edit.printed(raw)])

    def test_paragraphs_and_empty_text(self):
        self.assertEqual(trainer.effect_paragraphs("One.\n\nTwo."), ["One.", "Two."])
        self.assertEqual(trainer.effect_paragraphs(None), [])

    def test_black_and_white_rule_box_goes(self):
        # bw11-RC20 Elesa; dv1-19 First Ticket keeps its own parenthesis (TCGdex text)
        raw = ("Search your deck for 3 Pokémon Tool cards, reveal them, and put them into your hand. Shuffle your deck "
               "afterward. You may play only 1 Supporter card during your turn (before your attack).")
        self.assertTrue(trainer.effect_paragraphs(raw)[0].endswith("hand. Shuffle your deck afterward."))
        raw = ("If both players play First Ticket, flip the coin as normal. (You may play only 1 First Ticket before you "
               "flip that coin.) You may play as many Item cards as you like during your turn (before your attack).")
        self.assertTrue(trainer.effect_paragraphs(raw)[0].endswith("(You may play only 1 First Ticket before you flip "
                                                                   "that coin.)"))

    def test_a_supporter_whose_text_needs_its_place_keeps_it(self):
        # dp6-130 Buck's Training
        raw = ("You can play only one Supporter card each turn. When you play this card, put it next to your Active "
               "Pokémon. When your turn ends, discard this card. Draw 2 cards. As long as Buck's Training is next to "
               "your Active Pokémon, each of your Active Pokémon's attacks does 10 more damage.")
        self.assertTrue(trainer.effect_paragraphs(raw)[0].startswith(
            "When you play this card, put it next to your Active Pokémon. When your turn ends, discard this card. Draw"))

    def test_a_choices_options_become_bullets(self):
        # swsh9-148 Roseanne's Backup: TCGdex runs the options together
        raw = ("Choose 1 or more:\nShuffle a Pokémon from your discard pile into your deck.Shuffle a Stadium card from "
               "your discard pile into your deck.")
        self.assertEqual(trainer.effect_paragraphs(raw),
                         ["Choose 1 or more:", "• Shuffle a Pokémon from your discard pile into your deck.",
                          "• Shuffle a Stadium card from your discard pile into your deck."])
        # swsh1-171 Pal Pad: already bulleted
        raw = "Choose 1 or both:\n• Shuffle up to 2 Pokémon.\n• Shuffle up to 2 basic Energy cards."
        self.assertEqual(trainer.effect_paragraphs(raw),
                         ["Choose 1 or both:", "• Shuffle up to 2 Pokémon.", "• Shuffle up to 2 basic Energy cards."])

    def test_curly_double_quotes(self):
        # dp2-115 Rock-Paper-Scissors
        self.assertEqual(trainer.effect_paragraphs('you and your opponent play "Rock-Paper-Scissors."'),
                         ["you and your opponent play “Rock-Paper-Scissors.”"])


class Kinds(unittest.TestCase):
    def test_untyped_trainers_by_their_rule(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # neo4-92 Broken Ground Gym, neo1-93 Gold Berry
            self.assertEqual(trainer.untyped_kind("neo4-92", "This card stays in play when you play it. Discard this "
                                                             "card if another Stadium card comes into play."), "Stadium")
            self.assertEqual(trainer.untyped_kind("neo1-93", "Attach Gold Berry to 1 of your Pokémon that doesn't "
                                                             "have a Pokémon Tool attached to it."), "Tool")
        with self.assertWarns(UserWarning):
            self.assertEqual(trainer.untyped_kind("base1-91", "Draw 2 cards."), "Item")       # base1-91 Bill

    def test_technical_machines_are_tools(self):
        self.assertEqual(trainer.KINDS["Technical Machine"], "Tool")
        self.assertEqual(trainer.KINDS["Rocket's Secret Machine"], "Item")

    def test_tool_text_makes_a_tool(self):
        # ex10-80 Curse Powder (TCGdex: Item); ecard2-121 Fighting Cube 01 (no type, "my" for "may"); ex4-79 Team
        # Aqua Technical Machine 01 (TCGdex: Item)
        curse = ("Attach Curse Powder to 1 of your Evolved Pokémon (excluding Pokémon-ex) that doesn't already have a "
                 "Pokémon Tool attached to it.")
        cube = ("Attach this card to 1 of your Fighting Pokémon in play. That Pokémon my use this card's attack "
                "instead of its own.")
        aqua = ("Attach this card to 1 of your Pokémon that has Team Aqua in its name. That Pokémon may use this "
                "card's attack instead of its own.")
        self.assertEqual(trainer.kind_of("ex10-80", "Item", curse), "Tool")
        self.assertEqual(trainer.kind_of("ecard2-121", None, cube), "Tool")
        self.assertEqual(trainer.kind_of("ex4-79", "Item", aqua), "Tool")
        # an Item that only names Tools stays one; another type is TCGdex's
        self.assertEqual(trainer.kind_of("x-1", "Item", "Discard 2 Pokémon Tools attached to Pokémon."), "Item")
        self.assertEqual(trainer.kind_of("x-1", "Supporter", curse), "Supporter")


class FossilHp(unittest.TestCase):
    def test_hp_goes_into_the_text_as_gen_9_words_it(self):
        # base1-70 Clefairy Doll; dp5-84 Old Amber (after symbols); sm12-197 Lillie's Poké Doll
        self.assertEqual(trainer.with_hp("Play Clefairy Doll as if it were a Basic Pokémon.", "10"),
                         "Play Clefairy Doll as if it were a 10-HP Basic Pokémon.")
        self.assertEqual(trainer.with_hp("Play Old Amber as if it were a {C} Basic Pokémon.", "50"),
                         "Play Old Amber as if it were a 50-HP Basic {C} Pokémon.")
        self.assertEqual(trainer.with_hp("Play this card as if it were a 30-HP {C} Basic Pokémon.", "30"),
                         "Play this card as if it were a 30-HP Basic {C} Pokémon.")

    def test_hp_spelled_without_a_hyphen(self):
        # xy4-102 Robo Substitute (after symbols)
        self.assertEqual(trainer.with_hp("as if it were a 30 HP {C} Basic Pokémon.", "30"),
                         "as if it were a 30-HP Basic {C} Pokémon.")

    def test_no_hp_no_change(self):
        self.assertEqual(trainer.with_hp("as if it were a Basic Pokémon", ""), "as if it were a Basic Pokémon")


class Text(unittest.TestCase):
    def test_supporter_character(self):
        self.assertEqual(trainer.split_name("Boss’s Orders (Giovanni)", "Supporter"), ("Boss’s Orders", "Giovanni"))
        self.assertEqual(trainer.split_name("Boss’s Orders (Giovanni)", "Item"), ("Boss’s Orders (Giovanni)", ""))

    def test_numbers_stay_with_the_next_word(self):
        self.assertEqual(trainer.glue_numbers("draw 3 cards, 1 of them"), f"draw 3{trainer.ts.GLUE}cards, "
                                                                            f"1{trainer.ts.GLUE}of them")
        self.assertEqual(trainer.glue_numbers("HP 30"), "HP 30")

    def test_first_baseline_snaps_to_the_grid(self):
        self.assertEqual(trainer.snap(trainer.GRID_TOP), trainer.GRID_TOP)
        self.assertAlmostEqual(trainer.snap(690), trainer.GRID_TOP - trainer.GRID_STEP)


class ReadCard(unittest.TestCase):
    def test_supporter(self):
        card = read(tcgdex())
        self.assertEqual((card["kind"], card["name"], card["character"], card["era"]),
                         ("Supporter", "Lady Outing", "", "ex"))
        self.assertTrue(card["effect"].startswith("Search your deck"))
        self.assertEqual((card["number"], card["set_code"], card["year"], card["rarity"]),
                         ("083/109", "RS", "2003", "uncommon"))

    def test_refusals(self):
        refused = [tcgdex(category="Pokemon"), tcgdex(trainerType="Energy"),
                   tcgdex(rarity="Special illustration rare"), tcgdex(rarity="ACE SPEC Rare"),
                   tcgdex(id="bw7-137", rarity="Rare"), tcgdex(name="Super Scoop Up ◇"),
                   # xy10-94 Chaos Tower
                   tcgdex(id="xy10-94", effect="Choose which way this card faces before you play it."),
                   # sm11-195 Flyinium Z (TCGdex text)
                   tcgdex(id="sm11-195", trainerType="Tool", effect="If the Pokémon this card is attached to has the Air "
                          "Slash attack, it can use the GX attack on this card."),
                   tcgdex(id="cel25cc-CC006", set={"id": "cel25cc"}), tcgdex(id="zz9-1", set={"id": "zz9"})]
        for record in refused:
            with self.subTest(record["id"], name=record["name"], rarity=record["rarity"]), \
                    self.assertRaises(SystemExit):
                read(record)
        with self.assertRaises(SystemExit):                 # A1-216: a TCG Pocket card
            read(tcgdex(id="A1-216", set={"id": "A1"}), dict(SET, serie={"id": "tcgp"}))
        with self.assertRaises(SystemExit):                 # xy4-97 (TCGdex name)
            read(tcgdex(id="xy4-97", name="Head Ringer Team Flare Hyper Gear", trainerType="Tool", set={"id": "xy4"}))

    def test_team_flare_gear_label_leaves_the_name(self):
        # xy4-102 (TCGdex name): the print sets "Team Flare Gear" beside the name
        card = read(tcgdex(id="xy4-102", name="Robo Substitute Team Flare Gear", trainerType="Item",
                           set={"id": "xy4"}, effect="Draw a card."))
        self.assertEqual(card["name"], "Robo Substitute")

    def test_tcgdex_typo(self):
        # ecard2-121 Fighting Cube 01 prints "may"
        self.assertEqual(trainer.effect_paragraphs("That Pokémon my use this card's attack."),
                         ["That Pokémon may use this card’s attack."])
        # ecard3-129 Miracle Sphere α: TCGdex's text discards "Miracle Plate α"
        self.assertEqual(trainer.effect_paragraphs("discard Miracle Plate α."), ["discard Miracle Sphere α."])

    def test_hp_from_the_text(self):
        # swsh10-157 Unidentified Fossil: TCGdex gives no hp
        card = read(tcgdex(id="swsh10-157", name="Unidentified Fossil", trainerType="Item", set={"id": "swsh10"},
                           effect="Play this card as if it were a 60-HP Basic Colorless Pokémon."))
        self.assertEqual((card["hp"], card["effect"]),
                         ("60", "Play this card as if it were a 60-HP Basic {C} Pokémon."))

    def test_fossil(self):
        # pl1-119 Armor Fossil: HP and a Poké-Body
        record = tcgdex(id="pl1-119", name="Armor Fossil", trainerType="Item", hp=50,
                        effect="Play Armor Fossil as if it were a Colorless Basic Pokémon.",
                        abilities=[{"type": "Poke-BODY", "name": "Armor Stone",
                                    "effect": "Whenever Armor Fossil would be damaged by your opponent's attack, "
                                              "flip a coin until you get tails. For each heads, reduce that damage "
                                              "by 10."}])
        card = read(record)
        self.assertEqual(card["hp"], "50")
        self.assertEqual(card["effect"], "Play Armor Fossil as if it were a 50-HP Basic {C} Pokémon.")
        self.assertEqual(card["abilities"][0]["name"], "Armor Stone")

    def test_an_ability_tcgdex_has_no_text_for_is_left_out_with_a_warning(self):
        # dp5-84 Old Amber: TCGdex lists only the Poké-Body's type
        with self.assertWarns(UserWarning):
            card = read(tcgdex(id="dp5-84", trainerType="Item", hp=50, abilities=[{"type": "Poke-BODY"}]))
        self.assertEqual(card["abilities"], [])

    def test_blank_text_comes_from_another_print(self):
        # np-26 Tropical Wind has no text on TCGdex; dpp-DP05 has it
        dp05 = tcgdex(id="dpp-DP05", name="Tropical Wind", trainerType=None,
                      effect="Flip a coin. If heads, remove 2 damage counters from each Active Pokémon.")
        briefs = [{"id": "np-26"}, {"id": "dpp-DP05"}]
        with mock.patch.object(trainer.tcg, "search", return_value=briefs), \
                mock.patch.object(trainer.tcg, "card", side_effect=lambda i: dp05), \
                self.assertWarns(UserWarning):
            effect = trainer.reprint_effect("np-26", "Tropical Wind")
        self.assertTrue(effect.startswith("Flip a coin."))

    def test_same_era_prints_are_asked_first(self):
        asked = []
        def card(card_id):
            asked.append(card_id)
            return tcgdex(id=card_id, effect="")
        briefs = [{"id": "sv01-1"}, {"id": "ex3-5"}, {"id": "ex1-9"}]
        with mock.patch.object(trainer.tcg, "search", return_value=briefs), \
                mock.patch.object(trainer.tcg, "card", side_effect=card), self.assertWarns(UserWarning):
            self.assertEqual(trainer.reprint_effect("np-26", "X"), "")
        self.assertEqual(asked, ["ex1-9", "ex3-5", "sv01-1"])


class Eras(unittest.TestCase):
    def test_eras(self):
        for card_id, era in (("swsh1-156", "swsh"), ("sm12-197", "sm"), ("xy1-121", "xy"), ("bw5-92", "bw"),
                             ("hgss1-89", "hgss"), ("col1-76", "hgss"), ("ex10-83", "ex"), ("base1-91", "base"),
                             ("xy12-76", "evolutions"), ("xyp-XY203", "evolutions"), ("xyp-XY177", "xy"),
                             ("zz9-1", "old")):
            self.assertEqual(trainer.era(card_id), era, card_id)

    def test_window_by_era(self):
        scan = np.zeros((1024, 733, 4), np.float32)
        self.assertEqual(trainer.window({"era": "sm", "printed_kind": "Item"}, scan), trainer.ERA_ART["sm"])
        # a Black & White / XY Tool's band covers the art's foot; an edited kind does not move the window
        self.assertEqual(trainer.window({"era": "xy", "printed_kind": "Tool", "kind": "Item"}, scan),
                         trainer.ERA_TOOL_ART["xy"])
        with mock.patch.object(oldwindow, "register", return_value=((0.0, 1.0, 0.9), (0.0, 1.0, 0.9))):
            got = trainer.window({"era": "evolutions", "id": "xy12-76", "printed_kind": "Item"}, scan)
        x0, y0, x1, y1 = oldwindow.LAYOUTS["evolutions"]["window"]
        i = oldwindow.INSET
        self.assertEqual(got, (x0 + i, y0 + i, x1 - i, y1 - i))


class OldWindow(unittest.TestCase):
    def test_layouts_by_set(self):
        for card_id, layout in (("base1-91", "base"), ("neo4-92", "base"), ("ecard1-137", "ecard"),
                                ("ex10-83", "ex"), ("pop5-6", "ex"), ("np-26", "ex"), ("dp1-114", "dp"),
                                ("pl4-83", "dp"), ("dpp-DP05", "dp"), ("xy12-88", "evolutions"),
                                ("xyp-XY210", "evolutions")):
            self.assertEqual(oldwindow.layout_of(card_id), layout, card_id)
        self.assertIsNone(oldwindow.layout_of("sv01-1"))
        self.assertIsNone(oldwindow.layout_of("xyp-XY202"))

    def test_pools_are_their_layouts_prints(self):
        for name, spec in oldwindow.LAYOUTS.items():
            for card_id in spec["pool"]:
                self.assertEqual(oldwindow.layout_of(card_id), name, card_id)

    def test_fit_finds_a_known_shift_and_scale(self):
        rng = np.random.default_rng(1)
        ref = np.zeros(733)
        ref[rng.choice(np.arange(30, 700), 14, replace=False)] = rng.uniform(4, 10, 14)
        ref = np.convolve(ref, [0.25, 0.5, 1, 0.5, 0.25], "same") + 1
        shift, scale = 6.0, 0.97
        xs = np.arange(733.0)
        card = np.interp((xs - 366.5 - shift) / scale + 366.5, xs, ref)      # the frame moved and shrunk
        d, s, corr = oldwindow.fit(card, ref, oldwindow.ACROSS_SPAN)
        self.assertAlmostEqual(d, shift, delta=0.5)
        self.assertAlmostEqual(s, scale, delta=0.003)
        self.assertGreater(corr, 0.95)

    def test_window_carries_the_layouts_window_onto_the_card(self):
        x0, y0, x1, y1 = oldwindow.LAYOUTS["ex"]["window"]
        with mock.patch.object(oldwindow, "register", return_value=((3.0, 1.0, 0.9), (-2.0, 1.0, 0.9))):
            got = oldwindow.window("ex10-83", None)
        i = oldwindow.INSET
        self.assertEqual(got, (x0 + 3 + i, y0 - 2 + i, x1 + 3 - i, y1 - 2 - i))

    def test_poor_match_warns_and_unknown_set_raises(self):
        with mock.patch.object(oldwindow, "register", return_value=((0.0, 1.0, 0.2), (0.0, 1.0, 0.9))), \
                self.assertWarns(UserWarning):
            oldwindow.window("ex10-83", None)
        with self.assertRaises(ValueError):
            oldwindow.window("zz1-1", None)


class Badges(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((40, 60), bool)
        self.mask[10:20, 30:50] = True
        self.look = np.zeros((40, 60, 3), np.float32)
        self.look[self.mask] = 200

    def scan(self, badge_value):
        s = np.full((40, 60, 4), 50, np.float32)
        s[self.mask, :3] = badge_value
        return s

    def test_found_by_how_close_the_pixels_are(self):
        with mock.patch.object(badges, "look", return_value=(self.mask, self.look)):
            self.assertEqual(badges.found("sm", self.scan(210), (0, 0, 60, 40)), ["tag team"])
            self.assertEqual(badges.found("sm", self.scan(120), (0, 0, 60, 40)), [])
            self.assertEqual(badges.found("ex", self.scan(200), (0, 0, 60, 40)), [])      # no badges that era

    def test_erase_paints_only_round_the_badge_inside_the_window(self):
        def paint(rgba, need):
            out = rgba.copy()
            out[need, :3] = 0
            return out
        window = (5, 2, 58, 38)
        with mock.patch.object(badges, "look", return_value=(self.mask, self.look)), \
                mock.patch.object(badges.inpaint, "inpaint", side_effect=paint) as lama:
            out, names = badges.erase(self.scan(200), "sm", window)
        self.assertEqual(names, ["tag team"])
        changed = (out != self.scan(200)).any(-1)
        grown = np.zeros_like(self.mask)
        grown[10 - badges.GROW:20 + badges.GROW, 30 - badges.GROW:50 + badges.GROW] = True
        self.assertTrue(changed[self.mask].all())
        self.assertFalse((changed & ~grown).any())
        art, need = lama.call_args[0]
        self.assertEqual(art.shape[:2], (36 + 2 * badges.PAD, 53 + 2 * badges.PAD))
        with mock.patch.object(badges, "look", return_value=(self.mask, self.look)):
            out, names = badges.erase(self.scan(120), "sm", window)
        self.assertEqual(names, [])

    def test_pools_are_their_eras_prints(self):
        for name, (era, ids) in badges.POOL.items():
            self.assertIn(era, trainer.ERA_ART)
            for card_id in ids:
                self.assertEqual(trainer.era(card_id), era, (name, card_id))


def trainer_card(**over):
    base = {"id": "ex1-83", "kind": "Supporter", "printed_kind": "Supporter", "prefix": "", "name": "Lady Outing",
            "character": "", "hp": "", "effect": "Search your deck.\nShuffle your deck afterward.", "abilities": [],
            "attacks": [], "illustrator": "Ken Sugimori", "reg_mark": "", "set_code": "RS", "number": "083/109",
            "year": "2003", "rarity": "uncommon", "era": "old"}
    base.update(over)
    return base


class EditTrainer(unittest.TestCase):
    def test_round_trip(self):
        c = trainer_card()
        view = edit.editable(c)
        self.assertEqual(set(view), {"name", "character", "illustrator", "kind", "hp", "effect", "abilities", "attacks"})
        out = edit.apply(c, view)
        self.assertEqual(out, c)
        self.assertIsNot(out, c)

    def test_edits(self):
        out = edit.apply(trainer_card(), {"kind": "Item", "hp": "60", "name": "Pokemon Outing",
                                          "effect": "  Draw 2 cards.  \n\n\nDon't   flip.\n"})
        self.assertEqual((out["kind"], out["hp"], out["name"]), ("Item", "60", "Pokémon Outing"))
        self.assertEqual(out["effect"], "Draw 2 cards.\nDon’t flip.")
        self.assertEqual(out["printed_kind"], "Supporter")          # the art window follows the print
        self.assertEqual(edit.apply(trainer_card(hp="60"), {"hp": ""})["hp"], "")

    def test_rejections(self):
        bad = [{"kind": "Energy"}, {"hp": "55"}, {"name": ""}, {"effect": 3}, {"effect": "x" * 801},
               {"effect": "\n".join("p" for _ in range(7))}, {"attacks": [{"cost": [], "name": ""}]},
               {"abilities": [{"name": "A", "effect": ""}]}]
        for edits in bad:
            with self.subTest(edits), self.assertRaises(ValueError):
                edit.apply(trainer_card(), edits)

    def test_changes(self):
        before = trainer_card()
        after = edit.apply(before, {"kind": "Item", "effect": "Draw 2 cards.", "character": "Lady"})
        self.assertEqual(edit.changes(before, after), [
            "Character: none → Lady", "Trainer type: Supporter → Item",
            "Text: “Search your deck. / Shuffle your deck afterward.” → “Draw 2 cards.”"])
        self.assertEqual(edit.changes(before, before), [])


class GuiTrainer(unittest.TestCase):
    def test_trainers_get_the_trainer_layout(self):
        with mock.patch.object(gui.tcg, "card", return_value=tcgdex()):
            self.assertEqual(gui.layout_of("ex1-83"), "trainer")
        with mock.patch.object(gui, "read", return_value=("trainer", trainer_card())):
            summary = gui.card_summary("ex1-83")
        self.assertEqual((summary["layout"], summary["type"], summary["card"]["kind"]),
                         ("trainer", "Supporter", "Supporter"))

    def test_gen9_trainers_get_the_trainer_layout(self):
        # sv01-166 Arven, TCGdex: category Trainer, trainerType Supporter
        with mock.patch.object(gui.tcg, "card", return_value=tcgdex(id="sv01-166", name="Arven", set={"id": "sv01"})):
            self.assertEqual(gui.layout_of("sv01-166"), "trainer")
        arven = trainer_card(id="sv01-166", name="Arven", era="modern")
        with mock.patch.object(gui, "read", return_value=("trainer", arven)):
            self.assertTrue(gui.card_summary("sv01-166")["gen9"])
        with mock.patch.object(gui, "read", return_value=("trainer", arven)),                 mock.patch.object(gui.tcg, "card", return_value=tcgdex(id="sv01-166", name="Arven")):
            self.assertEqual(gui.card_changes("sv01-166"), ("Arven", []))

    def test_gen9_pokemon_keep_the_modern_layout(self):
        with mock.patch.object(gui.tcg, "card", return_value={"id": "sv09-080", "category": "Pokemon", "name": "Larvitar"}):
            self.assertEqual(gui.layout_of("sv09-080"), "modern")

    def test_no_holofoil_for_trainers(self):
        with mock.patch.object(gui, "read", return_value=("trainer", trainer_card())), \
                mock.patch.object(gui.tcg, "TCGLIVE", "x"), self.assertRaises(gui.UserError):
            gui.render("ex1-83", holo=True)

    def test_changes_note_the_restyle(self):
        with mock.patch.object(gui, "read", return_value=("trainer", trainer_card())), \
                mock.patch.object(gui.tcg, "card", return_value=tcgdex()):
            name, notes = gui.card_changes("ex1-83", attacks=False)
        self.assertEqual((name, notes), ("Lady Outing", ["Restyled as a Scarlet & Violet Trainer"]))

    def test_reading_warnings_reach_the_render(self):
        def read(card_id):
            warnings.warn("base1-91: TCGdex gives no Trainer type and its text holds no rule: set as an Item")
            return "trainer", trainer_card(id=card_id)
        gui._renders.clear()
        blank = np.zeros((4, 4, 4), np.float32)
        with mock.patch.object(gui, "read", side_effect=read), \
                mock.patch.object(gui.trainer, "compose_card", return_value=(blank, {})):
            _, notes, _, _ = gui.render("base1-91")
        gui._renders.clear()
        self.assertTrue(any("no Trainer type" in n for n in notes), notes)


class Cli(unittest.TestCase):
    def test_file_name_has_no_colon(self):
        # sm12-190 "Dragonium Z: Dragon Claw": a colon would name an NTFS stream, not a file
        card = trainer_card(id="sm12-190", name="Dragonium Z: Dragon Claw")
        with mock.patch.object(cli.trainer, "compose_card", return_value=(None, {"card": card})), \
                mock.patch.object(cli, "save_rgba"):
            path = cli.render_trainer("sm12-190", Path("out"))
        self.assertEqual(path.name, "sm12-190-dragonium-z-dragon-claw-trainer.png")
        self.assertEqual(cli.file_slug(" Type: Null"), "type-null")          # the Pokemon cards' names too


@unittest.skipUnless(os.environ.get("SVRESTYLE_FONTS"), "needs the Gill Sans faces (SVRESTYLE_FONTS)")
class GillSans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trainer.ts.configure(os.environ["SVRESTYLE_FONTS"])

    def test_missing_glyphs(self):
        self.assertEqual(trainer.ts.missing_glyphs("Miracle Sphere α", trainer.ts.BOLD), ["α"])
        self.assertEqual(trainer.ts.missing_glyphs("Pokémon’s • ×2 {C}", trainer.ts.MEDIUM), [])
        with self.assertWarns(UserWarning):
            trainer.warn_missing_glyphs(trainer_card(name="Miracle Sphere γ"))

    def test_options_hang_and_run_on(self):
        # sv06-154 Kieran: a blank line after "Choose 1:", none between the options, each hanging after its bullet
        lines, rows = trainer.lay_paragraphs(["Choose 1:", "• Switch your Active Pokémon.", "• Draw a card."], 1)
        self.assertEqual(rows, 4)
        self.assertEqual([(row, indent, bullet) for row, _, _, _, indent, bullet in lines],
                         [(0, 0, False), (2, trainer.BULLET_INDENT, True), (3, trainer.BULLET_INDENT, True)])
        self.assertEqual("".join(t for t, _ in lines[1][1]).replace(trainer.ts.GLUE, " "), "Switch your Active Pokémon.")

    def test_a_long_character_is_narrowed_and_leaves_the_name_room(self):
        canvas = np.zeros((1024, 733, 4), np.float32)
        with self.assertWarns(UserWarning):
            xscale = trainer.name_and_character(canvas, trainer_card(character="A Very Long Character Name Indeed"))
        self.assertGreater(xscale, 0)
        width = trainer.ts.text_width("Professor Magnolia", trainer.CHARACTER_FACE, trainer.CHARACTER_SIZE)
        self.assertLess(width, trainer.CHARACTER_MAX_W)


if __name__ == "__main__":
    unittest.main()
