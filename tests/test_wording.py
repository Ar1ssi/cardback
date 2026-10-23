"""Printed-wording helpers: term capitals and the Claude suggestion call (mocked, no network)."""
import io
import json
import unittest
import urllib.error
from unittest import mock

from svrestyle import edit, wording


class FixTerms(unittest.TestCase):
    def test_capitalises_fixed_terms_only(self):
        self.assertEqual(wording.fix_terms("switch your active pokémon with 1 of your benched pokémon."),
                         "Switch your Active Pokémon with 1 of your Benched Pokémon.".replace("Switch", "switch"))
        self.assertEqual(wording.fix_terms("your opponent's active pokémon is now paralyzed and confused."),
                         "your opponent's Active Pokémon is now Paralyzed and Confused.")
        self.assertEqual(wording.fix_terms("take 2 prize cards; discard an energy card and a special energy."),
                         "take 2 Prize cards; discard an Energy card and a Special Energy.")
        self.assertEqual(wording.fix_terms("if that pokémon is knocked out, put it in the lost zone."),
                         "if that Pokémon is Knocked Out, put it in the Lost Zone.")
        self.assertEqual(wording.fix_terms("choose a stage 2 card, a pokémon tool, or an item."),
                         "choose a Stage 2 card, a Pokémon Tool, or an Item.")

    def test_leaves_lower_case_words_and_symbols(self):
        text = "Flip a coin. If heads, this attack does 30 more damage. Put 2 damage counters on it. Attach a {R} Energy."
        self.assertEqual(wording.fix_terms(text), text)
        self.assertEqual(wording.fix_terms("attach a basic {r} energy card"), "attach a Basic {r} Energy card")

    def test_whole_words_possessives_and_hyphens(self):
        self.assertEqual(wording.fix_terms("Slowpokemon's items, benches"), "Slowpokemon's Items, benches")
        self.assertEqual(wording.fix_terms("your pokémon's retreat cost"), "your Pokémon's Retreat Cost")
        self.assertEqual(wording.fix_terms("pokémon-gx and pokémon-ex"), "Pokémon-GX and Pokémon-EX")
        self.assertEqual(wording.fix_terms("special conditions, abilities, rule boxes"),
                         "Special Conditions, Abilities, Rule Boxes")

    def test_keeps_shouting_length_and_empty(self):
        self.assertEqual(wording.fix_terms("KNOCKED OUT"), "KNOCKED OUT")
        self.assertEqual(wording.fix_terms(""), "")
        self.assertEqual(wording.fix_terms(None), "")
        for text in ("your active pokémon", "ACTIVE pokemon", "Energy cards"):
            self.assertEqual(len(wording.fix_terms(text)), len(text))
        fixed = wording.fix_terms("a bench, a benched pokémon, the active spot")
        self.assertEqual(wording.fix_terms(fixed), fixed)         # idempotent

    def test_edit_apply_fixes_effect_text_not_names(self):
        card = {"id": "swsh3-88", "type": "Fighting", "prefix": "", "name": "Tyranitar", "hp": "180",
                "illustrator": "x", "evolves_from": "Pupitar", "flavor": "f", "abilities": [], "attacks": [],
                "weakness": None, "resistance": None, "retreat": 4}
        out = edit.apply(card, {"attacks": [{"cost": [], "name": "active bite", "damage": "10",
                                             "effect": "your opponent's active pokemon is now asleep."}],
                                "abilities": [{"name": "energy sense", "effect": "once during your turn, look at a prize card."}]})
        self.assertEqual(out["attacks"][0]["name"], "active bite")
        self.assertEqual(out["attacks"][0]["effect"], "your opponent’s Active Pokémon is now Asleep.")
        self.assertEqual(out["abilities"][0]["effect"], "once during your turn, look at a Prize card.")
        trainer = {"id": "swsh1-1", "kind": "Item", "printed_kind": "Item", "prefix": "", "name": "Ball",
                   "character": "", "hp": "", "effect": "x", "abilities": [], "attacks": [], "illustrator": ""}
        out = edit.apply(trainer, {"effect": "search your deck for a basic pokemon.\nthen, shuffle your deck."})
        self.assertEqual(out["effect"], "search your deck for a Basic Pokémon.\nthen, shuffle your deck.")


def api_answer(text, notes=("x",)):
    body = json.dumps({"content": [{"type": "text", "text": json.dumps({"text": text, "notes": list(notes)})}]})
    return mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False, read=lambda: body.encode())


class Suggest(unittest.TestCase):
    def test_needs_a_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(wording.available())
            with self.assertRaises(wording.SuggestionError):
                wording.suggest("flip a coin", "attack")
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "k"}):
            self.assertTrue(wording.available())

    def test_calls_the_messages_api_and_fixes_terms_in_the_answer(self):
        urlopen = mock.MagicMock(return_value=api_answer("Flip a coin. If heads, your opponent's active pokémon is now Paralyzed."))
        out = wording.suggest("flip coin, heads = paralyze", "attack", {"name": "Pikachu", "type": "Lightning", "hp": "60",
                                                                       "illustrator": "no"}, urlopen=urlopen, api_key="sk-test")
        self.assertEqual(out, {"text": "Flip a coin. If heads, your opponent's Active Pokémon is now Paralyzed.",
                               "notes": ["x"]})
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, wording.API_URL)
        self.assertEqual(req.get_header("X-api-key"), "sk-test")
        self.assertEqual(req.get_header("Anthropic-version"), "2023-06-01")
        sent = json.loads(req.data)
        self.assertEqual((sent["model"], sent["temperature"]), (wording.MODEL, 0))
        self.assertIn("flip coin, heads = paralyze", sent["messages"][0]["content"])
        self.assertIn('"name": "Pikachu"', sent["messages"][0]["content"])
        self.assertNotIn("illustrator", sent["messages"][0]["content"])
        self.assertIn("Ultra Ball [Scarlet & Violet 196]", sent["messages"][0]["content"])

    def test_bad_input(self):
        for text, kind in (("", "attack"), ("x" * 1201, "attack"), ("fine", "flavor")):
            with self.subTest(text=text[:5], kind=kind), self.assertRaises(ValueError):
                wording.suggest(text, kind, api_key="k")

    def test_api_failures_become_suggestion_errors(self):
        http = urllib.error.HTTPError(wording.API_URL, 401, "Unauthorized", {}, io.BytesIO(b'{"error":"bad key"}'))
        for failure in (http, urllib.error.URLError("offline"), TimeoutError("slow")):
            with self.subTest(failure=type(failure).__name__), self.assertRaises(wording.SuggestionError) as caught:
                wording.suggest("text", "attack", urlopen=mock.MagicMock(side_effect=failure), api_key="k")
            self.assertTrue(str(caught.exception))
        self.assertIn("401", str(wording.suggest.__globals__["SuggestionError"]("Anthropic API error 401: x")))

    def test_answers_that_are_not_json_are_refused(self):
        plain = mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False,
                               read=lambda: json.dumps({"content": [{"type": "text", "text": "Sure! Here you go."}]}).encode())
        with self.assertRaises(wording.SuggestionError):
            wording.suggest("text", "attack", urlopen=mock.MagicMock(return_value=plain), api_key="k")
        prose = mock.MagicMock(__enter__=lambda s: s, __exit__=lambda *a: False,
                               read=lambda: json.dumps({"content": [{"type": "text",
                                                                     "text": 'Here: {"text": "Draw 2 cards.", "notes": []}'}]}).encode())
        self.assertEqual(wording.suggest("draw two", "trainer", urlopen=mock.MagicMock(return_value=prose), api_key="k"),
                         {"text": "Draw 2 cards.", "notes": []})


if __name__ == "__main__":
    unittest.main()
