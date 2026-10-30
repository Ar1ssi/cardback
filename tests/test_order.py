"""The order a set is listed in: Pokemon, Trainers (Item, Supporter, Stadium, Tool, each A-Z; ACE SPEC last, A-Z),
Energy. Only the list's order: ids and numbers stay. No network or scans.
Run: python -m unittest discover tests"""
import json
import tempfile
import unittest
from pathlib import Path

from svrestyle import export


def rec(number, name, category, kind=None, rarity="Common", source=None, types=None, stage=None):
    out = {"id": f"customhb-{number:03d}", "localId": f"{number:03d}", "name": name, "category": category,
           "rarity": rarity, "custom": {"source": source or f"sv01-{number}"}}
    if kind:
        out["trainerType"] = kind
    if types:
        out["types"] = types
    if stage:
        out["stage"] = stage
    return out


CARDS = [
    rec(1, "Zubat", "Pokemon", types=["Psychic"], stage="Basic"), rec(2, "Abra", "Pokemon", types=["Psychic"], stage="Basic"),
    rec(3, "Tool B", "Trainer", "Tool"), rec(4, "Stadium Z", "Trainer", "Stadium"),
    rec(5, "Boss's Orders", "Trainer", "Supporter"), rec(6, "ultra Ball", "Trainer", "Item"),
    rec(7, "Max Rod", "Trainer", "Item"), rec(8, "Hero's Cape", "Trainer", "Tool", "ACE SPEC Rare"),
    rec(9, "Basic Fire Energy", "Energy"), rec(10, "Counter Energy", "Energy"),
    rec(11, "Awakening Drum", "Trainer", "Tool"), rec(12, "Master Ball", "Trainer", "Item", "Rare"),
    rec(13, "Legacy Energy", "Energy", rarity="ACE SPEC Rare"), rec(14, "Arven", "Trainer", "Supporter"),
]


class DisplayOrder(unittest.TestCase):
    def names(self, records, **kw):
        return [r["name"] for r in export.display_order(records, **kw)]

    def test_pokemon_then_trainers_then_energy(self):
        self.assertEqual(self.names(CARDS), [
            "Abra", "Zubat",                                                       # Pokemon: A-Z
            "Master Ball", "Max Rod", "ultra Ball",                                # Items A-Z (case ignored)
            "Arven", "Boss's Orders",                                              # Supporters A-Z
            "Stadium Z",                                                           # Stadiums
            "Awakening Drum", "Tool B",                                            # Tools A-Z
            "Hero's Cape",                                                         # ACE SPEC Trainers last
            "Basic Fire Energy", "Counter Energy", "Legacy Energy"])               # Energy by number, ACE SPEC among it

    def test_the_ace_spec_trainers_follow_every_kind_and_go_a_to_z(self):
        cards = CARDS + [rec(15, "Amulet of Hope", "Trainer", "Tool", "ACE SPEC Rare"),
                         rec(16, "Brilliant Blender", "Trainer", "Item", "ACE SPEC Rare")]
        names = self.names(cards)
        self.assertEqual(names[names.index("Tool B") + 1:names.index("Basic Fire Energy")],
                         ["Amulet of Hope", "Brilliant Blender", "Hero's Cape"])

    def test_a_record_that_does_not_say_ace_spec_is_asked(self):
        cards = [rec(1, "Item B", "Trainer", "Item"), rec(2, "Old Ace", "Trainer", "Tool", source="sv05-9"),
                 rec(3, "Item A", "Trainer", "Item")]
        self.assertEqual(self.names(cards), ["Item A", "Item B", "Old Ace"])
        asked = []
        self.assertEqual(self.names(cards, is_ace=lambda source: asked.append(source) or source == "sv01-1"),
                         ["Item A", "Old Ace", "Item B"], "Item B is ACE SPEC: after every other Trainer")
        self.assertEqual(sorted(asked), ["sv01-1", "sv01-3", "sv05-9"])

    def test_ids_and_numbers_stay(self):
        before = {r["id"]: r["localId"] for r in CARDS}
        after = {r["id"]: r["localId"] for r in export.display_order(CARDS)}
        self.assertEqual(before, after)
        self.assertEqual([r["id"] for r in export.by_number(export.display_order(CARDS))], [r["id"] for r in CARDS])

    def test_unknown_things_go_last_and_nothing_is_lost(self):
        cards = CARDS + [rec(15, "Odd", "Other")]
        ordered = export.display_order(cards)
        self.assertEqual((ordered[-1]["name"], len(ordered)), ("Odd", len(cards)))


class PokemonOrder(unittest.TestCase):
    def test_by_type_a_to_z(self):
        cards = [rec(1, "Zubat", "Pokemon", types=["Darkness"], stage="Basic"),
                 rec(2, "Squirtle", "Pokemon", types=["Water"], stage="Basic"),
                 rec(3, "Bulbasaur", "Pokemon", types=["Grass"], stage="Basic"),
                 rec(4, "Charmander", "Pokemon", types=["Fire"], stage="Basic"),
                 rec(5, "Ditto", "Pokemon", types=["Colorless"], stage="Basic"),
                 rec(6, "Weedle", "Pokemon", types=["Grass"], stage="Basic")]
        self.assertEqual([r["name"] for r in export.display_order(cards)],
                         ["Ditto", "Zubat", "Charmander", "Bulbasaur", "Weedle", "Squirtle"])

    def test_an_evolution_line_stays_together(self):
        def evo(number, name, stage, types, frm=None):
            r = rec(number, name, "Pokemon", types=types, stage=stage)
            if frm:
                r["evolveFrom"] = frm
            return r
        cards = [evo(1, "Hippowdon", "Stage1", ["Fighting"], "Hippopotas"), evo(2, "Drilbur", "Basic", ["Fighting"]),
                 evo(3, "Excadrill", "Stage1", ["Fighting"], "Drilbur"), evo(4, "Hippopotas", "Basic", ["Fighting"]),
                 evo(5, "Zubat", "Basic", ["Fighting"]), evo(6, "Golbat", "Stage1", ["Fighting"], "Zubat"),
                 evo(7, "Crobat ex", "Stage2", ["Fighting"], "Golbat")]
        self.assertEqual([r["name"] for r in export.display_order(cards)],
                         ["Drilbur", "Excadrill", "Hippopotas", "Hippowdon", "Zubat", "Golbat", "Crobat ex"])

    def test_the_rookidee_line_is_listed_with_metal(self):
        cards = [rec(1, "Rookidee", "Pokemon", types=["Colorless"], stage="Basic"),
                 dict(rec(2, "Corviknight", "Pokemon", types=["Metal"], stage="Stage2"), evolveFrom="Corvisquire"),
                 dict(rec(3, "Corvisquire", "Pokemon", types=["Colorless"], stage="Stage1"), evolveFrom="Rookidee"),
                 rec(4, "Ditto", "Pokemon", types=["Colorless"], stage="Basic"), rec(5, "Aron", "Pokemon", types=["Metal"], stage="Basic")]
        self.assertEqual([r["name"] for r in export.display_order(cards)],
                         ["Ditto", "Aron", "Rookidee", "Corvisquire", "Corviknight"])

    def test_a_basic_nothing_evolves_from_keeps_its_own_type(self):
        cards = [rec(1, "Kartana", "Pokemon", types=["Grass"], stage="Basic"),
                 rec(2, "Kartana", "Pokemon", types=["Metal"], stage="Basic"),
                 rec(3, "Aron", "Pokemon", types=["Metal"], stage="Basic")]
        self.assertEqual([(r["name"], r["types"][0]) for r in export.display_order(cards)],
                         [("Kartana", "Grass"), ("Aron", "Metal"), ("Kartana", "Metal")])

    def test_gallade_ex_sits_in_the_ralts_line(self):
        cards = [rec(1, "Gallade ex", "Pokemon", types=["Fighting"], stage="Basic"),
                 rec(2, "Ralts", "Pokemon", types=["Psychic"], stage="Basic"),
                 dict(rec(3, "Kirlia", "Pokemon", types=["Psychic"], stage="Stage1"), evolveFrom="Ralts"),
                 dict(rec(4, "Gallade", "Pokemon", types=["Fighting"], stage="Stage2"), evolveFrom="Kirlia"),
                 dict(rec(5, "Gardevoir", "Pokemon", types=["Psychic"], stage="Stage2"), evolveFrom="Kirlia")]
        self.assertEqual([r["name"] for r in export.display_order(cards)],
                         ["Ralts", "Kirlia", "Gallade", "Gallade ex", "Gardevoir"])

    def test_a_record_with_no_type_or_stage_goes_after_the_others_of_its_group(self):
        cards = [rec(1, "Odd", "Pokemon"), rec(2, "Abra", "Pokemon", types=["Psychic"], stage="Stage2"),
                 rec(3, "Zubat", "Pokemon", types=["Psychic"])]
        self.assertEqual([r["name"] for r in export.display_order(cards)], ["Abra", "Zubat", "Odd"])


class ReadingASetInNumberOrder(unittest.TestCase):
    def test_a_set_listed_in_display_order_is_read_by_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {"format": export.FORMAT, "version": export.VERSION, "id": "customhb", "code": "HB", "name": "H",
                      "cards": export.display_order(CARDS)}
            (Path(tmp) / "set.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual([source for source, _ in export.existing_cards(tmp)], [r["custom"]["source"] for r in CARDS])
            held = export.unlisted_items(tmp, {"sv01-1"})
            self.assertEqual([place for place, _ in held], list(range(1, len(CARDS))), "places are card numbers - 1")


if __name__ == "__main__":
    unittest.main()
