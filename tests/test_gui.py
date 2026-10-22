"""GUI editing model and HTTP API. No network, model or fonts: card reading and rendering are mocked.
Run: python -m unittest discover tests"""
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from unittest import mock

from svrestyle import edit, gui


def card(**over):
    base = {"id": "swsh3-88", "type": "Fighting", "prefix": "", "name": "Tyranitar", "hp": "180", "illustrator": "x",
            "evolves_from": "Pupitar", "flavor": "f", "abilities": [],
            "attacks": [{"cost": ["Fighting"], "name": "Bedrock Breaker", "damage": 120, "effect": "Discard."}],
            "weakness": {"type": "Grass", "value": "×2"}, "resistance": None, "retreat": 4}
    base.update(over)
    return base


class Apply(unittest.TestCase):
    def test_round_trip_keeps_card(self):
        c = card()
        out = edit.apply(c, edit.editable(c))
        self.assertEqual(out["attacks"][0]["damage"], "120")
        self.assertEqual(out["weakness"], {"type": "Grass", "value": "×2"})
        self.assertIsNot(out, c)

    def test_pokemon_gets_its_accent(self):
        self.assertEqual(edit.printed("Your opponent's Active Pokemon"), "Your opponent’s Active Pokémon")
        self.assertEqual(edit.printed("POKEMON CHECKUP, Pokemon's, pokemon-ex"), "POKÉMON CHECKUP, Pokémon’s, pokémon-ex")
        self.assertEqual(edit.printed("Pokémon stays"), "Pokémon stays")
        self.assertEqual(edit.printed("Slowpokemon"), "Slowpokemon")        # only the word itself
        out = edit.apply(card(), {"attacks": [{"cost": [], "name": "Pokemon Power", "effect": "Heal each Pokemon."}]})
        self.assertEqual((out["attacks"][0]["name"], out["attacks"][0]["effect"]), ("Pokémon Power", "Heal each Pokémon."))

    def test_attack_edits(self):
        out = edit.apply(card(), {"attacks": [
            {"cost": ["Darkness", "Colorless"], "name": "Big Bite", "damage": "50x", "effect": "Don't flip."},
            {"cost": [], "name": "Nap", "damage": "", "effect": ""}]})
        self.assertEqual([a["damage"] for a in out["attacks"]], ["50×", ""])
        self.assertEqual(out["attacks"][0]["effect"], "Don’t flip.")      # printed apostrophe
        self.assertEqual(out["attacks"][1]["cost"], [])

    def test_weakness_resistance_retreat(self):
        out = edit.apply(card(), {"weakness": {"type": "Fire", "value": "+20"},
                                  "resistance": {"type": "Water", "value": "−30"}, "retreat": 0})
        self.assertEqual(out["weakness"]["value"], "+20")
        self.assertEqual(out["resistance"], {"type": "Water", "value": "-30"})
        self.assertEqual(out["retreat"], 0)
        self.assertIsNone(edit.apply(card(), {"weakness": None})["weakness"])

    def test_rejections(self):
        bad = [
            {"attacks": [{"cost": ["Fairy"], "name": "A", "damage": "10"}]},
            {"attacks": [{"cost": [], "name": "A", "damage": "+"}]},
            {"attacks": [{"cost": [], "name": "A", "damage": "ten"}]},
            {"attacks": [{"cost": [], "name": "", "damage": "10"}]},
            {"attacks": [{"cost": ["Fire"] * 6, "name": "A", "damage": "10"}]},
            {"attacks": [{"cost": [], "name": "A"}] * 5},
            {"abilities": [{"name": "A", "effect": ""}]},
            {"weakness": {"type": "Fire", "value": "2"}},
            {"resistance": {"type": "Fire", "value": "+30"}},
            {"retreat": 6}, {"retreat": True}, {"hp": "185"}, {"name": ""},
        ]
        for edits in bad:
            with self.subTest(edits), self.assertRaises(ValueError):
                edit.apply(card(), edits)

    def test_card_ids(self):
        for ok in ("swsh6-99", "sv03.5-065", "sma-SV33", "swshp-SWSH055", "swsh12.5-065", "sm10-19"):
            self.assertTrue(edit.valid_card_id(ok), ok)
        for bad in ("../etc", "swsh6/99", "", None, "a-b-c", "x" * 40):
            self.assertFalse(edit.valid_card_id(bad), bad)


class ImportSet(unittest.TestCase):
    def test_sorts_cards(self):
        data = {"name": "Mine", "cards": [
            {"id": "sv09-080", "name": "Larvitar", "supertype": "Pokémon"},
            {"id": "swsh6-97", "name": "Galarian Moltres V", "supertype": "Pokémon"},
            {"id": "swsh3-88", "name": "Tyranitar", "supertype": "Pokémon"},
            {"id": "swsh3-88", "name": "Tyranitar", "supertype": "Pokémon"},
            {"id": "me01-114", "name": "Boss's Orders", "supertype": "Trainer"},
            {"id": "swsh1-156", "name": "Air Balloon", "supertype": "Trainer"},
            {"id": "sv01-258", "name": "Basic Grass Energy", "supertype": "Energy"},
            {"id": "../x", "name": "bad"}]}
        out = gui.import_set(data)["cards"]
        self.assertEqual([(c["id"], c.get("layout"), c.get("skip")) for c in out],
                         [("sv09-080", "modern", None), ("swsh6-97", "ex", None), ("swsh3-88", "me", None),
                          ("me01-114", "trainer", None), ("swsh1-156", "trainer", None),
                          ("sv01-258", None, "not a Pokémon or Trainer")])
        self.assertEqual([c.get("gen9") for c in out if c.get("layout") == "trainer"], [True, False])

    def test_opened_project_looks_at_skipped_cards_again(self):
        # me01-117 Forest of Vitality: TCGdex category Trainer (Stadium); saved when Trainers were skipped
        cards = [{"id": "me01-117", "name": "Forest of Vitality", "skip": "not a Pokémon"},
                 {"id": "me03-086", "name": "Growing Grass Energy", "skip": "not a Pokémon"},
                 {"id": "swsh1-156", "name": "Air Balloon", "layout": "trainer"},
                 {"id": "me04-001", "name": "Weedle", "layout": "modern"}]
        category = {"me01-117": "Trainer", "me03-086": "Energy"}
        with mock.patch.object(gui.tcg, "card", side_effect=lambda i: {"id": i, "category": category[i]}):
            out = gui._current_entries(cards)
        self.assertEqual(out, [{"id": "me01-117", "name": "Forest of Vitality", "layout": "trainer", "gen9": True},
                               {"id": "me03-086", "name": "Growing Grass Energy", "skip": "not a Pokémon or Trainer"},
                               {"id": "swsh1-156", "name": "Air Balloon", "layout": "trainer", "gen9": False},
                               {"id": "me04-001", "name": "Weedle", "layout": "modern"}])

    def test_skipped_card_stays_when_tcgdex_fails(self):
        card = {"id": "me01-117", "name": "Forest of Vitality", "skip": "not a Pokémon"}
        with mock.patch.object(gui.tcg, "card", side_effect=gui.cache.DownloadError("offline")):
            self.assertEqual(gui._current_entries([card]), [card])

    def test_rejects_other_json(self):
        with self.assertRaises(gui.UserError):
            gui.import_set([1, 2])


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = gui.make_server(0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, None if body is None else json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def test_page_and_meta(self):
        status, ctype, body = self.call("/")
        self.assertEqual((status, ctype.split(";")[0]), (200, "text/html"))
        self.assertIn(b"Card Restyle", body)
        status, _, body = self.call("/api/meta")
        meta = json.loads(body)
        self.assertEqual(meta["energy"], list(edit.ENERGY))
        self.assertIn("+20", meta["weakness"])
        self.assertIn("-30", meta["resistance"])
        self.assertIn("foil", meta)

    def test_bad_requests(self):
        self.assertEqual(self.call("/api/card?id=../../etc/passwd")[0], 400)
        self.assertEqual(self.call("/api/icon/Fairy.png")[0], 400)
        self.assertEqual(self.call("/nope")[0], 404)
        self.assertEqual(self.call("/api/render", {"id": "swsh3-88", "style": "comic"})[0], 400)

    def test_project_poll(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(gui.cache, "CACHE", Path(tmp)):
            status, _, body = self.call("/api/projects/poll?name=Poll%20test&after=-1&client=b")
            hello = json.loads(body)
            self.assertEqual((status, hello["hello"], hello["events"], hello["viewers"]), (200, {"rev": 0}, [], 1))
            answer = {}
            waiting = threading.Thread(target=lambda: answer.update(json.loads(self.call(
                f"/api/projects/poll?name=Poll%20test&after={hello['seq']}&client=b")[2])))
            waiting.start()
            status, _, body = self.call("/api/projects/patch", {"name": "Poll test", "client": "a",
                                                                "patch": {"setName": "S"}})
            self.assertEqual((status, json.loads(body)["rev"]), (200, 1))
            waiting.join(5)
            self.assertFalse(waiting.is_alive())                 # answered on the patch, not after POLL_WAIT
            self.assertEqual(answer["events"], [{"event": "patch",
                                                 "data": {"rev": 1, "client": "a", "patch": {"setName": "S"}}}])
            self.assertEqual(self.call("/api/projects/leave", {"name": "Poll test", "client": "b"})[0], 200)
        self.assertEqual(self.call("/api/projects/poll?name=../x&after=-1&client=b")[0], 400)
        self.assertEqual(self.call("/api/projects/poll?name=x&after=soon&client=b")[0], 400)
        self.assertEqual(self.call("/api/projects/poll?name=x&after=-1")[0], 400)

    def test_holofoil_needs_an_extract(self):
        with mock.patch.object(gui.tcg, "TCGLIVE", None):
            status, _, body = self.call("/api/render", {"id": "swsh3-88", "foil": True})
        self.assertEqual(status, 400)
        self.assertIn("--tcglive", json.loads(body)["error"])

    def test_render_and_zip(self):
        gui._renders.clear()
        fake = gui.Render(b"\x89PNG fake", ["a warning"], "me")
        with mock.patch.object(gui, "render", return_value=fake) as r, \
                mock.patch.object(gui, "card_changes", return_value=("Tyranitar", ["Retreat: 4 → 2"])):
            status, _, body = self.call("/api/render", {"id": "swsh3-88", "edits": {"retreat": 2}})
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertEqual((data["layout"], data["warnings"]), ("me", ["a warning"]))
            self.assertEqual(data["changes"], ["Retreat: 4 → 2"])
            self.assertNotIn("foil", data)
            r.assert_called_once_with("swsh3-88", {"retreat": 2}, "hybrid", True, False)
            status, ctype, body = self.call("/api/zip", {"items": [{"id": "swsh3-88"}, {"id": "sm10-19"}]})
        self.assertEqual((status, ctype), (200, "application/zip"))
        z = zipfile.ZipFile(io.BytesIO(body))
        self.assertEqual(z.namelist(), ["swsh3-88-me.png", "sm10-19-me.png", "CHANGELOG.md"])
        self.assertIn("- Retreat: 4 → 2", z.read("CHANGELOG.md").decode())

    def test_render_and_zip_with_holofoil(self):
        gui._renders.clear()
        record = {"front": "swsh3-51-me-front.png", "base": None,
                  "reverse": {"shader": "TPCi/Cards3D/HoloFoil/FlatSilver", "material": "MAT_Cards3D_FlatSilver",
                              "wp": "swsh3-51-me-wp-ph.png", "cc": None, "etch": None, "cf": None}}
        files = {"swsh3-51-me-front.png": b"f", "swsh3-51-me-wp-ph.png": b"p", "swsh3-51-me-foil.json": b"{}"}
        fake = gui.Render(b"fake png", [], "me", (record, files, {"reverse": b"plate"}, "t0ken"))
        gui._renders["a request"] = fake             # what render() keeps, for the texture URLs
        with mock.patch.object(gui, "render", return_value=fake) as r,                 mock.patch.object(gui, "card_changes", return_value=("Kingler", [])):
            status, _, body = self.call("/api/render", {"id": "swsh3-51", "foil": True})
            data = json.loads(body)
            self.assertIsNone(data["foil"]["base"])
            self.assertEqual(data["foil"]["reverse"]["shader"], "TPCi/Cards3D/HoloFoil/FlatSilver")
            self.assertEqual(data["foil"]["front"], "/api/foil/file/t0ken/swsh3-51-me-front.png")
            self.assertEqual(data["foil"]["reverse"]["textures"],
                             {"wp": "/api/foil/file/t0ken/swsh3-51-me-wp-ph.png", "etch": None, "cf": None})
            r.assert_called_once_with("swsh3-51", None, "hybrid", True, True)
            status, _, body = self.call("/api/zip", {"items": [{"id": "swsh3-51", "foil": True}]})
        self.assertEqual(zipfile.ZipFile(io.BytesIO(body)).namelist(), ["swsh3-51-me.png", *files, "CHANGELOG.md"])
        self.assertEqual(self.call(data["foil"]["reverse"]["textures"]["wp"])[2], b"p")
        self.assertEqual(self.call("/api/foil/file/other/swsh3-51-me-wp-ph.png")[0], 404)
        self.assertEqual(self.call("/api/foil/file/t0ken/..%2Fsecret.txt")[0], 400)

    def test_foil_assets(self):
        status, ctype, body = self.call("/static/three/three.module.min.js")
        self.assertEqual((status, ctype.split(";")[0]), (200, "text/javascript"))
        self.assertIn(b"three.core.min.js", body)
        self.assertEqual(self.call("/static/foil.js")[0], 200)
        for bad in ("/static/../gui.py", "/static/%2e%2e/gui.py", "/static/three/nope.js", "/static/gui.html"):
            self.assertEqual(self.call(bad)[0], 404, bad)
        materials = json.dumps({"materials": {}, "textures": {"FX_T_Spectrum": {"url": "/web/x.webp"}}}).encode()
        with mock.patch.object(gui, "foil_materials", return_value=materials):
            self.assertEqual(self.call("/api/foil/tex/..%2F..%2Fx.png")[0], 400)
            self.assertEqual(self.call("/api/foil/tex/FX_T_Unknown.png")[0], 400)
        with mock.patch.object(gui.tcg, "TCGLIVE", None):
            gui.foil_materials.cache_clear()
            self.assertEqual(self.call("/api/foil/materials.json")[0], 400)
        gui.foil_materials.cache_clear()

    def test_edit_errors_come_back_as_400(self):
        with mock.patch.object(gui, "read", return_value=("me", card())):
            status, _, body = self.call("/api/render", {"id": "swsh3-88", "edits": {"retreat": 9}})
        self.assertEqual(status, 400)
        self.assertIn("retreat", json.loads(body)["error"])


if __name__ == "__main__":
    unittest.main()


class Projects(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(gui.cache, "CACHE", Path(self.tmp.name))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_save_list_open_delete(self):
        data = {"cards": [{"id": "swsh3-88", "name": "Tyranitar", "layout": "me"}],
                "edits": {"swsh3-88": {"hp": "190"}}, "current": "swsh3-88"}
        self.assertEqual(gui.list_projects(), {"projects": []})
        out = gui.save_project(" My Set (v2) ", data)
        self.assertEqual(out["name"], "My Set (v2)")
        listed = gui.list_projects()["projects"]
        self.assertEqual([(p["name"], p["cards"]) for p in listed], [("My Set (v2)", 1)])
        self.assertEqual(gui.load_project("My Set (v2)")["data"], data)
        gui.save_project("My Set (v2)", {**data, "cards": []})          # overwrites
        self.assertEqual(gui.list_projects()["projects"][0]["cards"], 0)
        self.assertEqual(gui.delete_project("My Set (v2)"), {"deleted": "My Set (v2)"})
        self.assertEqual(gui.list_projects(), {"projects": []})

    def test_rejections(self):
        for name in ("", "../x", "a/b", None, "x" * 61, " "):
            with self.subTest(name), self.assertRaises(gui.UserError):
                gui.save_project(name, {"cards": []})
        for data in (None, [], {"cards": "x"}, {"edits": []}):
            with self.subTest(data), self.assertRaises(gui.UserError):
                gui.save_project("ok", data)
        with self.assertRaises(gui.UserError):
            gui.load_project("missing")
        with mock.patch.object(gui, "MAX_PROJECT", 50), self.assertRaises(gui.UserError):
            gui.save_project("big", {"cards": [{"id": "x" * 100}]})

    def test_names_stay_inside_the_folder(self):
        self.assertEqual(gui.project_file("Base Set, 1999").parent, gui.projects_dir())
        self.assertEqual(gui.project_file("Base Set, 1999").name, "base-set-1999.json")

    def test_patches_change_only_what_they_name(self):
        base = {"cards": [{"id": "a"}, {"id": "b"}], "edits": {"a": {"name": "A", "hp": "60"}},
                "options": {"a": {"style": "hybrid"}}, "setName": "S"}
        out = gui.apply_patch(base, {"cards": {"removed": ["b"], "added": [{"index": 0, "card": {"id": "c"}}]},
                                     "edits": {"a": {"hp": "70", "weakness": None}, "c": {"name": "C"}},
                                     "options": {"a": None}})
        self.assertEqual([c["id"] for c in out["cards"]], ["c", "a"])
        self.assertEqual(out["edits"]["a"], {"name": "A", "hp": "70", "weakness": None})   # null is a value
        self.assertEqual(out["edits"]["c"], {"name": "C"})
        self.assertNotIn("a", out["options"])
        self.assertEqual(out["setName"], "S")
        self.assertEqual(base["edits"]["a"]["hp"], "60")                                   # a copy
        again = gui.apply_patch(out, {"cards": {"added": [{"index": 5, "card": {"id": "a"}}]}})
        self.assertEqual([c["id"] for c in again["cards"]], ["c", "a"])                    # no duplicates
        for bad in ([], {"cards": []}, {"edits": []}, {"edits": {"a": 3}}, {"cards": {"added": [{"card": {}}]}}):
            with self.subTest(bad), self.assertRaises(gui.UserError):
                gui.apply_patch(base, bad)

    def test_order_moves_only_the_cards_it_names(self):
        base = {"cards": [{"id": "a"}, {"id": "b"}, {"id": "new"}, {"id": "c"}]}
        out = gui.apply_patch(base, {"cards": {"order": ["c", "a", "b", "gone"]}})
        self.assertEqual([c["id"] for c in out["cards"]], ["c", "a", "new", "b"])        # "new" keeps its place
        moved = gui.apply_patch(base, {"cards": {"removed": ["a"], "added": [{"index": 0, "card": {"id": "d"}}],
                                                 "order": ["b", "c", "d"]}})
        self.assertEqual([c["id"] for c in moved["cards"]], ["b", "c", "new", "d"])
        with self.assertRaises(gui.UserError):
            gui.apply_patch(base, {"cards": {"order": "c"}})

    def test_two_people_editing_one_card_keep_both_fields(self):
        gui.patch_project("Shared", {"cards": {"added": [{"index": 0, "card": {"id": "a"}}]},
                                     "edits": {"a": {"name": "A", "hp": "60"}}})
        gui.patch_project("Shared", {"edits": {"a": {"name": "Alpha"}}}, client="one")
        out = gui.patch_project("Shared", {"edits": {"a": {"hp": "90"}}}, client="two")
        self.assertEqual(out["rev"], 3)
        self.assertEqual(gui.load_project("Shared")["data"]["edits"]["a"], {"name": "Alpha", "hp": "90"})
        self.assertEqual(gui.save_project("Shared", {"cards": []})["rev"], 4)

    def test_open_pages_hear_patches_saves_renames_and_deletes(self):
        seq = gui.poll_project("Live", -1, "me")["seq"]
        gui.patch_project("Live", {"setName": "L"}, client="me")
        gui.save_project("Live", {"cards": []}, client="me")
        news = gui.poll_project("Live", seq, "me")
        self.assertEqual([(e["event"], e["data"].get("rev")) for e in news["events"]], [("patch", 1), ("replaced", 2)])
        with mock.patch.object(gui, "POLL_WAIT", 0):
            seq = news["seq"]
            other = gui.HUB.poll("live", -1, "other", lambda: 2, wait=0)
            self.assertEqual((other["hello"], other["viewers"]), ({"rev": 2}, 2))
            news = gui.HUB.poll("live", seq, "me", lambda: 2, wait=0)
            self.assertEqual(news["events"], [{"event": "viewers", "data": {"viewers": 2}}])
            gui.leave_project("Live", "other")
            gui.delete_project("Live", renamed="Live 2", client="me")
            news = gui.HUB.poll("live", news["seq"], "me", lambda: 0, wait=0)
            self.assertEqual([e["event"] for e in news["events"]], ["viewers", "renamed"])
            self.assertEqual(news["events"][1]["data"], {"to": "Live 2", "client": "me"})
            self.assertEqual(news["viewers"], 1)

    def test_a_page_away_too_long_reads_the_project_again(self):
        with mock.patch.object(gui, "EVENT_LOG", 3):
            hub = gui.Hub()
            start = hub.poll("k", -1, "me", lambda: 0, wait=0)["seq"]
            for i in range(5):
                hub.publish("k", "patch", {"rev": i})
            self.assertEqual(hub.poll("k", start, "me", lambda: 5, wait=0)["hello"], {"rev": 5})
            self.assertEqual(hub.poll("k", 999, "me", lambda: 5, wait=0)["hello"], {"rev": 5})   # server restarted

    def test_quiet_pages_stop_counting(self):
        hub = gui.Hub()
        hub.poll("k", -1, "a", lambda: 0, wait=0)
        with mock.patch.object(gui.time, "monotonic", return_value=gui.time.monotonic() + gui.VIEWER_GONE + 1):
            self.assertEqual(hub.poll("k", -1, "b", lambda: 0, wait=0)["viewers"], 1)


class Changes(unittest.TestCase):
    def test_nothing_changed(self):
        c = card()
        self.assertEqual(edit.changes(c, edit.apply(c, edit.editable(c))), [])

    def test_each_kind(self):
        before = card(stage=2, attacks=[{"cost": ["Fighting"], "name": "Bedrock Breaker", "damage": "120", "effect": ""},
                                        {"cost": [], "name": "Roar", "damage": "", "effect": "Switch."}])
        after = edit.apply(before, {
            "name": "Tyranitar", "hp": "190", "stage": 1, "evolves_from": "Larvitar", "portrait": "sv09-080",
            "attacks": [{"cost": ["Fighting", "Fighting"], "name": "Bedrock Breaker", "damage": "150", "effect": ""},
                        {"cost": ["Colorless"], "name": "Smash", "damage": "30", "effect": ""}],
            "abilities": [{"name": "Sand Stream", "effect": "Each turn, hurt."}],
            "weakness": {"type": "Fire", "value": "×2"}, "resistance": {"type": "Psychic", "value": "-30"}, "retreat": 3})
        self.assertEqual(edit.changes(before, after), [
            "HP: 180 → 190", "Stage: Stage 2 → Stage 1", "Evolves from: Pupitar → Larvitar", "Portrait: from sv09-080",
            "Ability added: Sand Stream: “Each turn, hurt.”",
            "Attack Bedrock Breaker: cost: {F} → {F}{F}; damage: 120 → 150",
            "Attack Roar: name: Roar → Smash; cost: no cost → {C}; damage: none → 30; text: “Switch.” → none",
            "Weakness: Grass ×2 → Fire ×2", "Resistance: none → Psychic -30", "Retreat: 4 → 3"])

    def test_added_removed_reordered(self):
        a1 = {"cost": [], "name": "A", "damage": "10", "effect": ""}
        a2 = {"cost": [], "name": "B", "damage": "20", "effect": ""}
        before = card(attacks=[a1, a2])
        self.assertEqual(edit.changes(before, edit.apply(before, {"attacks": [a2, a1]})), ["Attack order: B, A"])
        self.assertEqual(edit.changes(before, edit.apply(before, {"attacks": [a1]})), ["Attack removed: no cost B 20"])

    def test_changelog_lists_changed_then_unchanged(self):
        answers = {"swsh3-88": ("Tyranitar", ["Restyled as a Mega Evolution card", "HP: 180 → 190"]),
                   "sv09-080": ("Larvitar", [])}
        with mock.patch.object(gui, "card_changes", side_effect=lambda i, *a: answers[i]):
            text = gui.changelog([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "sv09-080"}], "My set")
        self.assertTrue(text.startswith("# My set: changelog"))
        self.assertIn("## Tyranitar (swsh3-88)\n\n- Restyled as a Mega Evolution card\n- HP: 180 → 190", text)
        self.assertIn("## Unchanged\n\n- Larvitar (sv09-080)", text)
        with self.assertRaises(gui.UserError):
            gui.changelog([])
