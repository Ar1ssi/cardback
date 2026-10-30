"""The ptcg-sim custom-set export (export.py) and its re-import. No network: TCGdex lookups are mocked."""
import datetime
import io
import json
import time
import unittest
import warnings
import zipfile
from unittest import mock

from PIL import Image

from svrestyle import edit, export, gui

SET_ID = "customhbw"


def pokemon(**over):
    base = {"id": "swsh3-88", "family": "regular", "type": "Fighting", "prefix": "", "name": "Tyranitar", "hp": "180",
            "stage": 2, "evolves_from": "Pupitar", "portrait": "", "illustrator": "x", "reg_mark": "D", "rarity": "rare",
            "flavor": "Its roar.", "abilities": [{"name": "Sand Veil", "effect": "Prevent all effects."}],
            "attacks": [{"cost": ["Fighting", "Colorless"], "name": "Bedrock Breaker", "damage": "120",
                         "effect": "Discard a Stadium in play."},
                        {"cost": ["Fighting"], "name": "Growl", "damage": "", "effect": ""}],
            "weakness": {"type": "Grass", "value": "×2"}, "resistance": None, "retreat": 4}
    base.update(over)
    return base


def trainer(**over):
    base = {"id": "base1-91", "kind": "Supporter", "printed_kind": "Item", "prefix": "", "name": "Bill", "character": "",
            "hp": "", "effect": "Draw 2 cards.\nThen, shuffle your deck.", "abilities": [], "attacks": [],
            "illustrator": "y", "reg_mark": "", "rarity": "common"}
    base.update(over)
    return base


def png_bytes(color=(200, 30, 30, 255)):
    buf = io.BytesIO()
    Image.new("RGBA", (733, 1024), color).save(buf, "PNG")
    return buf.getvalue()


def holo_for(card_id, color=(10, 20, 30, 255), reverse_only=False):
    """What gui.render returns as its fourth part for a card with holofoil: (foil.json record, {file name: PNG}, plates,
    token), shaped as foil.files writes it."""
    stem = f"{card_id}-me"
    printing = {"shader": "TPCi/Cards3D/HoloFoil/SunPillar", "material": "MAT_Cards3D_SunPillar", "wp": f"{stem}-wp.png",
                "cc": "TEX_SouthernCross", "etch": None, "cf": None}
    record = {"id": card_id, "tcglive": card_id.replace("-", "_"), "front": f"{stem}-front.png",
              "base": None if reverse_only else printing, "reverse": printing if reverse_only else None}
    written = {record["front"]: png_bytes(color), printing["wp"]: png_bytes(color), f"{stem}-foil.json": b"{}"}
    return record, written, {}, "token"


class Records(unittest.TestCase):
    def test_pokemon_record_is_tcgdex_shaped(self):
        rec = export.tcgdex_record(pokemon(), "me", SET_ID, "001", {"id": "swsh3-88", "edits": {"hp": "180"}})
        self.assertEqual((rec["id"], rec["localId"], rec["category"], rec["name"]), ("customhbw-001", "001", "Pokemon", "Tyranitar"))
        self.assertEqual((rec["hp"], rec["types"], rec["stage"], rec["evolveFrom"]), (180, ["Fighting"], "Stage2", "Pupitar"))
        self.assertEqual(rec["attacks"][0], {"cost": ["Fighting", "Colorless"], "name": "Bedrock Breaker", "damage": 120,
                                             "effect": "Discard a Stadium in play."})
        self.assertEqual(rec["attacks"][1]["damage"], "")
        self.assertEqual(rec["abilities"], [{"type": "Ability", "name": "Sand Veil", "effect": "Prevent all effects."}])
        self.assertEqual((rec["weaknesses"], rec["resistances"], rec["retreat"]), ([{"type": "Grass", "value": "×2"}], [], 4))
        self.assertEqual((rec["regulationMark"], rec["rarity"], rec["description"]), ("D", "Rare", "Its roar."))
        self.assertEqual(rec["image"], "/src/assets/custom-sets/customhbw/001")
        self.assertEqual(rec["legal"], {"standard": False, "expanded": False})
        self.assertEqual(rec["custom"], {"source": "swsh3-88", "layout": "me", "edits": {"hp": "180"}, "style": "hybrid",
                                         "attacks": True, "foil": False})
        self.assertNotIn("suffix", rec)

    def test_ex_cards_get_their_suffix_and_plus_damage_stays_text(self):
        v = {k: v for k, v in pokemon(prefix="Galarian", name="Slowking", stage=0, evolves_from="").items() if k not in ("family", "rarity", "flavor")}
        v["attacks"][0]["damage"] = "90+"
        rec = export.tcgdex_record(v, "ex", SET_ID, "002", {"id": "swsh6-99", "style": "panel", "attacks": False, "foil": True})
        self.assertEqual((rec["name"], rec["suffix"], rec["stage"], rec["rarity"]), ("Galarian Slowking ex", "ex", "Basic", "Double rare"))
        self.assertEqual(rec["attacks"][0]["damage"], "90+")
        self.assertNotIn("evolveFrom", rec)
        self.assertEqual(rec["custom"]["style"], "panel")
        modern_ex = export.tcgdex_record(pokemon(family="ex"), "modern", SET_ID, "003", {"id": "sv03.5-065"})
        self.assertEqual(modern_ex["name"], "Tyranitar ex")
        regular_modern = export.tcgdex_record(pokemon(family="regular"), "modern", SET_ID, "004", {"id": "sv01-001"})
        self.assertEqual(regular_modern["name"], "Tyranitar")

    def test_mega_stage_follows_the_pre_evolution(self):
        mega = pokemon(stage=3, name="Tyranitar", evolves_from="Pupitar")
        with mock.patch.object(export.tcg, "search", return_value=[{"id": "swsh3-87"}]), \
                mock.patch.object(export.tcg, "card", return_value={"stage": "Stage1"}):
            self.assertEqual(export.tcgdex_record(mega, "me", SET_ID, "001", {"id": "x-1"})["stage"], "Stage2")
        with mock.patch.object(export.tcg, "search", return_value=[{"id": "a-1"}]), \
                mock.patch.object(export.tcg, "card", return_value={"stage": "Basic"}):
            self.assertEqual(export.mega_stage(mega), "Stage1")
        with mock.patch.object(export.tcg, "search", side_effect=OSError("offline")), self.assertWarns(UserWarning):
            self.assertEqual(export.mega_stage(mega), "Stage1")
        with mock.patch.object(export.tcg, "search", return_value=[]), self.assertWarns(UserWarning):
            self.assertEqual(export.mega_stage(mega), "Stage1")

    def test_trainer_record(self):
        rec = export.tcgdex_record(trainer(), "trainer", SET_ID, "005", {"id": "base1-91"})
        self.assertEqual((rec["category"], rec["trainerType"], rec["name"]), ("Trainer", "Supporter", "Bill"))
        self.assertEqual(rec["effect"], "Draw 2 cards.\n\nThen, shuffle your deck.")
        self.assertEqual(rec["rarity"], "Common")
        for key in ("hp", "attacks", "abilities", "types", "stage"):
            self.assertNotIn(key, rec)
        fossil = export.tcgdex_record(trainer(kind="Item", hp="50", abilities=[{"name": "Dig", "effect": "x"}]), "trainer",
                                      SET_ID, "006", {"id": "sm1-1"})
        self.assertEqual((fossil["trainerType"], fossil["hp"], fossil["abilities"][0]["type"]), ("Item", 50, "Ability"))

    def test_set_code_and_id(self):
        self.assertEqual(export.set_id_for("HBW"), "customhbw")
        self.assertEqual(export.set_id_for("A1"), "customa1")
        for bad in ("hbw", "H", "TOOLONG", "HB W", "", None, 12):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                export.set_id_for(bad)


class Zip(unittest.TestCase):
    def export(self, items, code="HBW"):
        reads = {"swsh3-88": ("me", pokemon()), "base1-91": ("trainer", trainer())}
        return export.export_zip(items, "Homebrew Wave 1", code, read=lambda i: reads[i],
                                 render=lambda *a: (png_bytes(), [], "me", None), apply=edit.apply,
                                 changelog=lambda items, title: f"# {title}", today=datetime.date(2026, 10, 23))

    def test_layout_and_set_json(self):
        data = self.export([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91"}])
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertEqual(sorted(z.namelist()), sorted([
            "customhbw/set.json", "customhbw/001/high.webp", "customhbw/001/low.webp",
            "customhbw/002/high.webp", "customhbw/002/low.webp", "customhbw/CHANGELOG.md"]))
        record = json.loads(z.read("customhbw/set.json"))
        self.assertEqual({k: record[k] for k in ("format", "version", "id", "code", "name", "releaseDate")},
                         {"format": "ptcg-sim.custom-set-data", "version": 1, "id": "customhbw", "code": "HBW",
                          "name": "Homebrew Wave 1", "releaseDate": "2026-10-23"})
        self.assertEqual([c["id"] for c in record["cards"]], ["customhbw-001", "customhbw-002"])
        self.assertEqual(record["cards"][0]["hp"], 190)                  # the edit applied
        self.assertEqual(record["cards"][0]["custom"]["edits"], {"hp": "190"})
        high = Image.open(io.BytesIO(z.read("customhbw/001/high.webp")))
        low = Image.open(io.BytesIO(z.read("customhbw/002/low.webp")))
        self.assertEqual((high.format, high.size, low.size), ("WEBP", (733, 1024), (245, 342)))
        self.assertTrue(z.read("customhbw/CHANGELOG.md").decode().startswith("# Homebrew Wave 1"))

    def test_rejections(self):
        with self.assertRaises(ValueError):
            self.export([], "HBW")
        with self.assertRaises(ValueError):
            self.export([{"id": "swsh3-88"}], "hbw")
        with self.assertRaises(ValueError):
            self.export(["swsh3-88"])

    def test_a_foil_card_brings_its_textures_and_a_foil_json(self):
        reads = {"swsh3-88": ("me", pokemon())}
        data = export.export_zip([{"id": "swsh3-88", "foil": True}], "Homebrew Wave 1", "HBW", read=lambda i: reads[i],
                                 render=lambda *a: (png_bytes(), [], "me", holo_for("swsh3-88")), apply=edit.apply,
                                 changelog=lambda items, title: f"# {title}", today=datetime.date(2026, 10, 23))
        z = zipfile.ZipFile(io.BytesIO(data))
        self.assertEqual(sorted(n for n in z.namelist() if "foil" in n),
                         ["customhbw/foil.json", "customhbw/foil/001-front.webp", "customhbw/foil/001-wp.webp"])
        self.assertEqual(json.loads(z.read("customhbw/foil.json"))["cards"]["001"]["front"], "001-front.webp")

    def test_round_trip_import(self):
        data = self.export([{"id": "swsh3-88", "edits": {"hp": "190"}, "style": "panel", "attacks": False, "foil": True},
                            {"id": "base1-91"}])
        record = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("customhbw/set.json"))
        self.assertTrue(export.is_set_data(record))
        with mock.patch.object(gui.modern, "is_modern", return_value=False):
            back = gui.import_set(record)
        self.assertEqual((back["name"], back["code"], back["foil"]), ("Homebrew Wave 1", "HBW", True))
        self.assertEqual([(c["id"], c["layout"]) for c in back["cards"]], [("swsh3-88", "me"), ("base1-91", "trainer")])
        self.assertEqual(back["edits"], {"swsh3-88": {"hp": "190"}})
        self.assertEqual(back["options"], {"swsh3-88": {"style": "panel", "attacks": False},
                                           "base1-91": {"style": "hybrid", "attacks": True}})
        with self.assertRaises(gui.UserError):
            gui.import_set({**record, "version": 2})
        # the older Custom Set tab export still imports as before
        with mock.patch.object(gui.modern, "is_modern", return_value=False):
            old = gui.import_set({"format": "ptcg-sim.custom-set", "name": "Picks", "cards": [{"id": "swsh3-88", "name": "Tyranitar", "supertype": "Pokémon"}]})
        self.assertEqual([c["layout"] for c in old["cards"]], ["me"])
        self.assertNotIn("edits", old)


if __name__ == "__main__":
    unittest.main()


class Http(unittest.TestCase):
    """The two routes over HTTP: /api/export-set (rendering mocked) and /api/suggest (no key)."""
    @classmethod
    def setUpClass(cls):
        import threading
        cls.server = gui.make_server(0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, path, body, headers=None):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type"), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type"), e.read()

    def get(self, path, headers=None):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(urllib.request.Request(self.base + path, None, headers or {}), timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def job(self, path, body, headers=None, seconds=20):
        """Starts a send or publish over HTTP and follows it to its end -> (the start's status, the job's last
        progress); a start the server refuses has no job: (status, its answer)."""
        status, _, raw = self.call(path, body, headers)
        job = json.loads(raw)
        end = time.time() + seconds
        while status == 200 and job["state"] == "running" and time.time() < end:
            time.sleep(0.02)
            job = json.loads(self.get(f"/api/set-job?job={job['id']}", headers)[1])
        return status, job

    def test_export_set(self):
        fake = gui.Render(png_bytes(), [], "me")
        with mock.patch.object(gui, "read", return_value=("me", pokemon())), \
                mock.patch.object(gui, "render", return_value=fake), \
                mock.patch.object(gui, "changelog", return_value="# log"):
            status, ctype, body = self.call("/api/export-set", {"items": [{"id": "swsh3-88"}], "title": "T", "code": "HBW"})
            self.assertEqual((status, ctype), (200, "application/zip"))
            names = zipfile.ZipFile(io.BytesIO(body)).namelist()
            self.assertEqual(sorted(names), ["customhbw/001/high.webp", "customhbw/001/low.webp", "customhbw/CHANGELOG.md",
                                             "customhbw/set.json"])
            status, _, body = self.call("/api/export-set", {"items": [{"id": "swsh3-88"}], "code": "bad code"})
            self.assertEqual(status, 400)
            self.assertIn("set code", json.loads(body)["error"])
            self.assertEqual(self.call("/api/export-set", {"items": [], "code": "HBW"})[0], 400)

    def test_suggest_without_a_key_or_cli_is_a_502_and_meta_says_so(self):
        import urllib.request
        from svrestyle import wording
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(wording.shutil, "which", return_value=None):
            status, _, body = self.call("/api/suggest", {"text": "flip coin", "kind": "attack"})
            self.assertEqual(status, 502)
            self.assertIn("ANTHROPIC_API_KEY", json.loads(body)["error"])
            with urllib.request.urlopen(self.base + "/api/meta", timeout=10) as r:
                meta = json.loads(r.read())
        self.assertFalse(meta["suggest"])
        self.assertIn("Active Pokémon", meta["terms"])
        self.assertEqual(self.call("/api/suggest", {"text": "", "kind": "attack"})[0], 400)

    def test_push_set_writes_into_the_simulator_folder(self):
        import tempfile
        import urllib.request
        from pathlib import Path
        fake = gui.Render(png_bytes(), [], "me")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gui, "read", return_value=("me", pokemon())), \
                mock.patch.object(gui, "render", return_value=fake), \
                mock.patch.object(gui, "changelog", return_value="# log"):
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": ""}):
                status, _, body = self.call("/api/push-set", {"items": [{"id": "swsh3-88"}], "code": "HBW"})
                self.assertEqual(status, 400)
                self.assertIn("SVRESTYLE_PTCG_SIM", json.loads(body)["error"])
            (Path(tmp) / "client" / "src").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp}):
                with urllib.request.urlopen(self.base + "/api/meta", timeout=10) as r:
                    self.assertTrue(json.loads(r.read())["ptcgSim"])
                status, job = self.job("/api/push-set", {"items": [{"id": "swsh3-88"}], "title": "T", "code": "HBW"})
                self.assertEqual((status, job["state"], job["kind"], job["setId"]), (200, "done", "push", "customhbw"))
                self.assertEqual({k: v for k, v in job["result"].items() if k != "written"},
                                 {"setId": "customhbw", "rendered": 1, "reused": 0, "unchanged": 0, "removed": []})
                self.assertEqual(job["pushed"], job["result"], "the files are in the folder")
                self.assertTrue((Path(tmp) / "client/src/assets/custom-sets/customhbw/001/high.webp").is_file())
                body = json.dumps({"items": [{"id": "swsh3-88"}], "code": "HBW"}).encode()
                for headers in ({"Origin": "https://evil.example"}, {"Cf-Connecting-Ip": "203.0.113.9"},
                                {"Host": "abc.trycloudflare.com"}):
                    with self.subTest(headers=headers):
                        req = urllib.request.Request(self.base + "/api/push-set", body,
                                                     {"Content-Type": "application/json", **headers})
                        with self.assertRaises(urllib.error.HTTPError) as caught:
                            urllib.request.urlopen(req, timeout=10)
                        self.assertEqual(caught.exception.code, 403)
                        meta = urllib.request.Request(self.base + "/api/meta", None, headers)
                        with urllib.request.urlopen(meta, timeout=10) as r:
                            self.assertFalse(json.loads(r.read())["ptcgSim"])

    def test_publish_key_lets_a_tunnel_browser_push(self):
        import tempfile
        import urllib.request
        from pathlib import Path
        fake = gui.Render(png_bytes(), [], "me")
        key = "k" * gui.MIN_PUBLISH_KEY
        body = json.dumps({"items": [{"id": "swsh3-88"}], "code": "HBW"}).encode()
        tunnel = {"Host": "abc.trycloudflare.com", "Cf-Connecting-Ip": "203.0.113.9", "Origin": "https://abc.trycloudflare.com"}

        def get_meta(headers):
            with urllib.request.urlopen(urllib.request.Request(self.base + "/api/meta", None, headers), timeout=10) as r:
                return json.loads(r.read())

        def push(headers):
            return self.job("/api/push-set", json.loads(body), headers)[0]

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gui, "read", return_value=("me", pokemon())), \
                mock.patch.object(gui, "render", return_value=fake), \
                mock.patch.object(gui, "changelog", return_value="# log"):
            (Path(tmp) / "client" / "src").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp, "SVRESTYLE_PUBLISH_TOKEN": key}):
                self.assertEqual((get_meta(tunnel)["ptcgSim"], get_meta(tunnel)["publishKey"]), (False, True))
                self.assertEqual(push(tunnel), 403)
                self.assertEqual(push({**tunnel, gui.PUBLISH_KEY_HEADER: key[:-1] + "x"}), 403, "a wrong key")
                self.assertTrue(get_meta({**tunnel, gui.PUBLISH_KEY_HEADER: key})["ptcgSim"])
                keyed = {**tunnel, gui.PUBLISH_KEY_HEADER: key}
                status, job = self.job("/api/push-set", json.loads(body), keyed)
                self.assertEqual((status, job["state"]), (200, "done"))
                url = f"/api/set-job?job={job['id']}"
                self.assertEqual(self.get(url, tunnel)[0], 403, "following a job needs the key too")
                self.assertEqual(self.call("/api/set-job/cancel", {"job": job["id"]}, tunnel)[0], 403)
                self.assertEqual(self.get(url, keyed)[0], 200)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp, "SVRESTYLE_PUBLISH_TOKEN": "short"}), \
                    warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertFalse(get_meta(tunnel)["publishKey"], "a key shorter than MIN_PUBLISH_KEY is ignored")
                self.assertEqual(push({**tunnel, gui.PUBLISH_KEY_HEADER: "short"}), 403)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp, "SVRESTYLE_PUBLISH_TOKEN": ""}):
                self.assertFalse(get_meta(tunnel)["publishKey"])
                self.assertEqual(push({**tunnel, gui.PUBLISH_KEY_HEADER: ""}), 403, "no key set: an empty one opens nothing")

    def test_publish_set_pushes_then_runs_the_simulator_publish_script(self):
        import tempfile
        import urllib.request
        from pathlib import Path
        fake = gui.Render(png_bytes(), [], "me")
        # stands in for the simulator's scripts/custom-sets/publish.mjs: echoes its arguments in the --json report
        script = ("const args = process.argv.slice(2);\n"
                  "if (process.env.FAKE_PUBLISH_FAIL) { console.error('rclone copy blobs failed: AccessDenied'); "
                  "process.exit(1); }\n"
                  "if (process.env.FAKE_PUBLISH_NOREPORT) { console.log('done, no json'); process.exit(0); }\n"
                  # a stalled upload: a child (rclone's place) that would write the marker if it outlived the kill
                  "if (process.env.FAKE_PUBLISH_HANG) { process.getBuiltinModule('child_process').spawn(process.execPath, ['-e', "
                  "`setTimeout(() => require('fs').writeFileSync(${JSON.stringify(process.env.FAKE_PUBLISH_HANG)}, "
                  "'x'), 4000)`], { stdio: 'inherit' }); setTimeout(() => {}, 60000); }\n"
                  "else {\n"
                  "console.log('progress line');\n"
                  "console.log(JSON.stringify({ sets: [{ setId: args[0], uploaded: 3, uploadedBytes: 4096, reused: 1, "
                  "skipped: [], unchanged: false }], deleted: 0, dryRun: false, args }));\n"
                  "}\n")
        meta = lambda: json.loads(urllib.request.urlopen(self.base + "/api/meta", timeout=10).read())["ptcgSite"]
        items = {"items": [{"id": "swsh3-88"}], "title": "T", "code": "HBW"}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(gui, "read", return_value=("me", pokemon())), \
                mock.patch.object(gui, "render", return_value=fake), \
                mock.patch.object(gui, "changelog", return_value="# log"):
            (Path(tmp) / "client" / "src").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp, "CUSTOM_SETS_R2_REMOTE": "r2:bucket/sets"}):
                self.assertFalse(meta(), "no publish script in the checkout")
                status, _, body = self.call("/api/publish-set", items)
                self.assertEqual(status, 400)
                self.assertIn("publish.mjs", json.loads(body)["error"])
                publish = Path(tmp, *gui.PUBLISH_SCRIPT)
                publish.parent.mkdir(parents=True)
                publish.write_text(script, encoding="utf-8")
                self.assertTrue(meta())
                status, job = self.job("/api/publish-set", items)
                self.assertEqual((status, job["state"], job["kind"]), (200, "done", "publish"))
                r = job["result"]
                self.assertEqual(r["setId"], "customhbw")
                self.assertTrue((Path(tmp) / "client/src/assets/custom-sets/customhbw/set.json").is_file())
                self.assertEqual(r["published"]["sets"][0]["uploaded"], 3)
                self.assertEqual(r["published"]["args"], ["customhbw", "--remote", "r2:bucket/sets", "--json"])
                with mock.patch.dict("os.environ", {"FAKE_PUBLISH_FAIL": "1"}):
                    status, failed = self.job("/api/publish-set", items)
                self.assertEqual((status, failed["state"]), (200, "failed"))
                self.assertEqual(failed["error"], "sent to ptcg-sim, but publishing failed: rclone copy blobs failed: "
                                                  "AccessDenied")
                self.assertEqual(failed["pushed"]["setId"], "customhbw", "the push result survives the failed upload")
                with mock.patch.dict("os.environ", {"FAKE_PUBLISH_NOREPORT": "1"}):
                    status, failed = self.job("/api/publish-set", items)
                self.assertEqual(failed["state"], "failed")
                self.assertIn("gave no report: 'done, no json'", failed["error"])
                marker = Path(tmp, "child-outlived-the-kill")
                with mock.patch.dict("os.environ", {"FAKE_PUBLISH_HANG": str(marker)}),                         mock.patch.object(gui, "PUBLISH_TIMEOUT", 1):
                    status, failed = self.job("/api/publish-set", items)
                self.assertEqual(failed["state"], "failed")
                self.assertIn("stopped it", failed["error"])
                time.sleep(5)
                self.assertFalse(marker.exists(), "the script's child process was killed with it")
                req = urllib.request.Request(self.base + "/api/publish-set", json.dumps(items).encode(),
                                             {"Content-Type": "application/json", "Cf-Connecting-Ip": "203.0.113.9"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                self.assertEqual(caught.exception.code, 403)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp, "CUSTOM_SETS_R2_REMOTE": ""}):
                self.assertFalse(meta(), "no bucket remote")
                status, _, body = self.call("/api/publish-set", items)
                self.assertEqual(status, 400)
                self.assertIn("CUSTOM_SETS_R2_REMOTE", json.loads(body)["error"])
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp, "CUSTOM_SETS_R2_REMOTE": "r2:bucket/sets"}), \
                    mock.patch.object(gui.shutil, "which", return_value=None):
                status, _, body = self.call("/api/publish-set", items)
                self.assertEqual(status, 400)
                self.assertIn("node not found", json.loads(body)["error"])



class SimFolder(unittest.TestCase):
    def test_sim_sets_dir_takes_the_checkout_or_the_folder_itself(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            sets = Path(tmp, "client", "src", "assets", "custom-sets")
            for value, want in (("", None), (str(Path(tmp, "missing")), None), (tmp, None)):
                with self.subTest(value=value), mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": value}):
                    self.assertEqual(gui.sim_sets_dir(), want)
            Path(tmp, "client", "src").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": tmp}):
                self.assertEqual(gui.sim_sets_dir(), sets)     # made by the first push, not here
            sets.mkdir(parents=True)
            with mock.patch.dict("os.environ", {"SVRESTYLE_PTCG_SIM": str(sets)}):
                self.assertEqual(gui.sim_sets_dir(), sets)


class Push(unittest.TestCase):
    """write_set_dir: the set written into the simulator's folder, re-rendering only what changed."""
    COLORS = {"swsh3-88": (200, 30, 30, 255), "base1-91": (30, 30, 200, 255), "swsh3-89": (30, 200, 30, 255)}

    def setUp(self):
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.set_dir = self.root / SET_ID
        self.renders = []
        self.makes_foil = True                     # False: a render with foil asked for comes back without any

    def tearDown(self):
        self._tmp.cleanup()

    def push(self, items, force=False, title="Homebrew Wave 1", **extra):
        reads = {"swsh3-88": ("me", pokemon()), "base1-91": ("trainer", trainer()),
                 "swsh3-89": ("me", pokemon(id="swsh3-89", name="Larvitar", stage=0, evolves_from=""))}

        def render(card_id, edits=None, style="hybrid", attacks=True, foil=False):
            self.renders.append(card_id)
            holo = holo_for(card_id, self.COLORS[card_id]) if foil and self.makes_foil else None
            return png_bytes(self.COLORS[card_id]), [], "me", holo
        return export.write_set_dir(items, title, "HBW", self.root, read=lambda i: reads[i], render=render,
                                    apply=edit.apply, changelog=lambda items, title: f"# {title}", force=force,
                                    today=datetime.date(2026, 10, 25), **extra)

    def record(self):
        return json.loads((self.set_dir / "set.json").read_text(encoding="utf-8"))

    def color_of(self, local_id):
        """The card front's colour channel ("r", "g" or "b"): WebP is lossy, so exact pixels drift."""
        pixel = Image.open(self.set_dir / local_id / "high.webp").convert("RGB").getpixel((360, 500))
        return "rgb"[pixel.index(max(pixel))]

    def test_first_push_writes_every_file_and_a_second_changes_nothing(self):
        first = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((first["setId"], first["rendered"], first["reused"], first["removed"]), (SET_ID, 2, 0, []))
        self.assertEqual(first["written"][-1], "set.json")           # last: the simulator reloads on it
        self.assertEqual(sorted(first["written"]), sorted(["001/high.webp", "001/low.webp", "002/high.webp",
                                                           "002/low.webp", "CHANGELOG.md", "set.json"]))
        again = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((again["rendered"], again["reused"], again["written"], again["unchanged"]), (0, 2, [], 6))
        self.assertEqual(self.renders, ["swsh3-88", "base1-91"])
        self.assertEqual(list(self.set_dir.glob("**/.*.tmp")), [])

    def test_an_edit_re_renders_that_card_only(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.renders.clear()
        r = self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 1, ["swsh3-88"]))
        self.assertEqual(r["written"], ["set.json"])          # the mock renders the same pixels: equal WebPs stay
        self.assertEqual(self.record()["cards"][0]["hp"], 190)
        self.renders.clear()
        r = self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91", "style": "panel"}])
        self.assertEqual(self.renders, ["base1-91"])

    def test_reordering_moves_kept_cards_without_rendering(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.renders.clear()
        r = self.push([{"id": "base1-91"}, {"id": "swsh3-88"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (0, 2, []))
        cards = sorted(self.record()["cards"], key=lambda c: c["id"])        # set.json lists the cards in display order
        self.assertEqual([(c["id"], c["custom"]["source"], c["image"]) for c in cards], [
            (f"{SET_ID}-001", "base1-91", f"/src/assets/custom-sets/{SET_ID}/001"),
            (f"{SET_ID}-002", "swsh3-88", f"/src/assets/custom-sets/{SET_ID}/002")])
        self.assertEqual((self.color_of("001"), self.color_of("002")), ("b", "r"))

    def test_a_shorter_list_removes_card_folders_and_keeps_other_files(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}, {"id": "swsh3-89"}])
        (self.set_dir / "ENGINE-CHECK.md").write_text("# not ours")
        r = self.push([{"id": "swsh3-89"}])
        self.assertEqual((r["rendered"], r["removed"]), (0, ["002", "003"]))
        self.assertEqual(sorted(p.name for p in self.set_dir.iterdir()), ["001", "CHANGELOG.md", "ENGINE-CHECK.md", "set.json"])
        self.assertEqual(self.color_of("001"), "g")

    def foil_json(self):
        return json.loads((self.set_dir / "foil.json").read_text(encoding="utf-8"))

    def foil_files(self):
        return sorted(p.name for p in (self.set_dir / "foil").iterdir()) if (self.set_dir / "foil").is_dir() else []

    def foil_color(self, local_id):
        pixel = Image.open(self.set_dir / "foil" / f"{local_id}-front.webp").convert("RGB").getpixel((360, 500))
        return "rgb"[pixel.index(max(pixel))]

    def test_a_foil_card_gets_its_textures_and_a_row_and_a_second_push_changes_nothing(self):
        items = [{"id": "swsh3-88", "foil": True}, {"id": "base1-91"}]
        first = self.push(items)
        self.assertEqual(sorted(first["written"]), sorted([
            "001/high.webp", "001/low.webp", "002/high.webp", "002/low.webp", "CHANGELOG.md", "set.json", "foil.json",
            "foil/001-front.webp", "foil/001-wp.webp"]))
        self.assertEqual(first["written"][-1], "set.json", "still last: the simulator reloads on it")
        self.assertEqual(self.foil_json(), {"v": 1, "cards": {"001": {
            "tcgl": "swsh3_88", "shader": "TPCi/Cards3D/HoloFoil/SunPillar", "material": "MAT_Cards3D_SunPillar",
            "front": "001-front.webp", "wp": "001-wp.webp", "cc": "TEX_SouthernCross"}}})
        cards = self.record()["cards"]
        self.assertEqual((set(cards[0]["custom"]["files"]), set(cards[1]["custom"]["files"])), ({"high", "low", "foil"}, {"high", "low"}))
        with Image.open(self.set_dir / "foil" / "001-front.webp") as front:
            self.assertEqual(front.size, (733, 1024))
        self.assertEqual(self.foil_color("001"), "r")
        self.renders.clear()
        again = self.push(items)
        self.assertEqual((again["rendered"], again["reused"], again["written"], again["unchanged"]), (0, 2, [], 9))
        self.assertEqual(self.renders, [])

    def test_an_edit_makes_the_foil_again_and_a_reorder_moves_it_with_its_card(self):
        a, b, c = {"id": "swsh3-88", "foil": True}, {"id": "base1-91"}, {"id": "swsh3-89", "foil": True}
        self.push([a, b, c])
        self.assertEqual((sorted(self.foil_json()["cards"]), self.foil_files()),
                         (["001", "003"], ["001-front.webp", "001-wp.webp", "003-front.webp", "003-wp.webp"]))
        self.renders.clear()
        r = self.push([c, a, b])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (0, 3, []), "a reorder renders nothing")
        self.assertEqual((sorted(self.foil_json()["cards"]), self.foil_files()),
                         (["001", "002"], ["001-front.webp", "001-wp.webp", "002-front.webp", "002-wp.webp"]))
        self.assertEqual((self.foil_color("001"), self.foil_color("002")), ("g", "r"), "each foil went with its card")
        self.assertEqual(self.foil_json()["cards"]["002"]["front"], "002-front.webp")
        edited = {**a, "edits": {"hp": "190"}}
        self.renders.clear()
        r = self.push([c, edited, b])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 2, ["swsh3-88"]))
        kept = export.exported_cards(self.set_dir)
        self.assertEqual({k: v[3] is not None for k, v in kept.items()}, {"swsh3-88": True, "swsh3-89": True, "base1-91": False})

    def test_foil_that_is_not_the_one_its_record_names_is_made_again(self):
        items = [{"id": "swsh3-88", "foil": True}, {"id": "swsh3-89", "foil": True}]
        self.push(items)
        (self.set_dir / "foil" / "001-front.webp").write_bytes((self.set_dir / "foil" / "002-front.webp").read_bytes())
        (self.set_dir / "foil" / "002-wp.webp").unlink()
        plan = export.plan_set(items, self.set_dir)
        self.assertEqual(sorted((item["id"], stale) for item, _, stale in plan["changed"]),
                         [("swsh3-88", "foil"), ("swsh3-89", "foil")])
        self.assertEqual(plan["unchanged"], 0)
        self.renders.clear()
        r = self.push(items)
        self.assertEqual((r["rendered"], sorted(self.renders)), (2, ["swsh3-88", "swsh3-89"]), "the plan said so")
        self.assertEqual((self.foil_color("001"), self.foil_color("002")), ("r", "g"))
        self.assertEqual(export.plan_set(items, self.set_dir)["unchanged"], 2, "and the push mended it")
        rows = self.foil_json()                                        # a row gone from foil.json: the same
        del rows["cards"]["001"]
        (self.set_dir / "foil.json").write_text(json.dumps(rows), encoding="utf-8")
        self.assertEqual([(i["id"], s) for i, _, s in export.plan_set(items, self.set_dir)["changed"]], [("swsh3-88", "foil")])

    def test_a_foil_made_outside_a_push_is_made_again_by_the_first_one(self):
        items = [{"id": "swsh3-88", "foil": True}, {"id": "base1-91"}]
        self.push(items)
        record = self.record()                                         # as import-foil.py left a set: rows, no digest
        record["cards"][0]["custom"]["files"].pop("foil")
        (self.set_dir / "set.json").write_text(json.dumps(record), encoding="utf-8")
        plan = export.plan_set(items, self.set_dir)
        self.assertEqual(([(i["id"], s) for i, _, s in plan["changed"]], plan["unchanged"]), ([("swsh3-88", "foil")], 1))
        self.renders.clear()
        r = self.push(items)
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 1, ["swsh3-88"]))
        self.assertIn("foil", self.record()["cards"][0]["custom"]["files"])
        self.assertEqual(export.plan_set(items, self.set_dir)["unchanged"], 2)

    def test_a_card_that_has_no_foil_to_make_is_set_without_and_not_made_again_every_push(self):
        self.makes_foil = False
        items = [{"id": "swsh3-88", "foil": True}, {"id": "base1-91"}]
        self.push(items)
        self.assertFalse((self.set_dir / "foil.json").exists() or (self.set_dir / "foil").exists())
        files = self.record()["cards"][0]["custom"]["files"]
        self.assertEqual((files["foil"], self.record()["cards"][0]["custom"]["foil"]), (export.NO_FOIL, True))
        self.assertIn("no holofoil could be made", (self.set_dir / "CHANGELOG.md").read_text(encoding="utf-8"))
        self.renders.clear()
        r = self.push(items)
        self.assertEqual((r["rendered"], r["reused"], self.renders), (0, 2, []))
        self.assertEqual(export.plan_set(items, self.set_dir)["unchanged"], 2)

    def test_foil_follows_its_cards_out_of_the_set(self):
        a, c = {"id": "swsh3-88", "foil": True}, {"id": "swsh3-89", "foil": True}
        self.push([a, c])
        self.assertEqual(sorted(self.foil_json()["cards"]), ["001", "002"])
        self.push([c])
        self.assertEqual((sorted(self.foil_json()["cards"]), self.foil_files(), self.foil_color("001")),
                         (["001"], ["001-front.webp", "001-wp.webp"], "g"), "the dropped card's row and textures went")
        (self.set_dir / "foil" / "notes.txt").write_text("not a texture")
        self.push([{"id": "swsh3-89"}])                                # foil turned off: nothing is left of it
        self.assertFalse((self.set_dir / "foil.json").exists())
        self.assertEqual(self.foil_files(), ["notes.txt"], "only the textures a push made are removed")
        self.assertFalse(self.record()["cards"][0]["custom"]["foil"])

    def test_force_renders_every_card_and_a_broken_set_json_is_rebuilt(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.renders.clear()
        self.assertEqual(self.push([{"id": "swsh3-88"}, {"id": "base1-91"}], force=True)["rendered"], 2)
        (self.set_dir / "set.json").write_text("{not json")
        self.renders.clear()
        r = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], self.renders), (2, ["swsh3-88", "base1-91"]))
        self.assertEqual(self.record()["code"], "HBW")

    def test_images_that_are_not_the_ones_the_record_names_are_rendered_again(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        (self.set_dir / "001" / "high.webp").write_bytes((self.set_dir / "002" / "high.webp").read_bytes())
        record = self.record()
        record["cards"][1]["custom"]["source"] = ["not", "a", "card id"]          # a hand edit
        (self.set_dir / "set.json").write_text(json.dumps(record), encoding="utf-8")
        self.renders.clear()
        r = self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (2, 0, ["swsh3-88", "base1-91"]))
        self.assertEqual(self.color_of("001"), "r")

    def snapshot(self):
        return {str(p.relative_to(self.set_dir)): p.read_bytes() for p in self.set_dir.rglob("*") if p.is_file()}

    def test_progress_reports_each_card_then_each_file(self):
        calls = []
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}], progress=lambda *a: calls.append(a))
        self.assertEqual([c for c in calls if c[0] == "rendering"],
                         [("rendering", 0, 2, "swsh3-88"), ("rendering", 1, 2, "base1-91"), ("rendering", 2, 2, "")])
        writing = [c for c in calls if c[0] == "writing"]
        self.assertEqual((writing[0], len(writing)), (("writing", 0, 6, ""), 7))          # 4 WebPs, CHANGELOG.md, set.json
        self.assertEqual((writing[-1][1:3], writing[-1][3]), ((6, 6), "set.json"))
        self.assertEqual(calls.index(writing[0]), 3, "all the cards come before any file")

    def test_a_cancel_before_the_first_file_leaves_the_folder_as_it_was(self):
        with self.assertRaises(export.Cancelled):
            self.push([{"id": "swsh3-88"}], cancelled=lambda: True)
        self.assertFalse(self.set_dir.exists())
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        before = self.snapshot()
        self.renders.clear()
        with self.assertRaises(export.Cancelled):                     # asked for while the first card renders
            self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91", "style": "panel"}],
                      cancelled=lambda: bool(self.renders))
        self.assertEqual((self.renders, self.snapshot()), (["swsh3-88"], before))
        calls = []
        with self.assertRaises(export.Cancelled):                     # asked for once every card is made, before a file
            self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}],
                       cancelled=lambda: calls.append(1) or len(calls) > 1)
        self.assertEqual(self.snapshot(), before)

    def test_what_a_list_would_drop_from_the_set_already_there(self):
        self.assertIsNone(export.existing_cards(self.set_dir), "no set yet")
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}, {"id": "swsh3-89"}])
        existing = export.existing_cards(self.set_dir)
        self.assertEqual(existing, [("swsh3-88", "Tyranitar"), ("base1-91", "Bill"), ("swsh3-89", "Larvitar")])
        ids = ["swsh3-88", "base1-91", "swsh3-89"]
        self.assertEqual(export.dropped_cards(existing, ids), [])
        self.assertEqual(export.dropped_cards(existing, [ids[2], ids[0], ids[1], "sv1-1"]), [], "reordered and grown")
        self.assertEqual(export.dropped_cards(existing, ids[:1]), [("base1-91", "Bill"), ("swsh3-89", "Larvitar")])
        self.assertEqual(export.dropped_cards(existing, ["swsh3-88", "sv1-1", "sv1-2"]),
                         [("base1-91", "Bill"), ("swsh3-89", "Larvitar")],
                         "swapped cards are dropped though the count holds")
        record = self.record()                                         # a set this editor did not write: no sources
        for card in record["cards"]:
            card.pop("custom")
        (self.set_dir / "set.json").write_text(json.dumps(record), encoding="utf-8")
        existing = export.existing_cards(self.set_dir)
        self.assertEqual([source for source, _ in existing], [None, None, None])
        self.assertEqual(export.dropped_cards(existing, ["a", "b"]), [(None, "Larvitar")], "by position: the end of the set")
        self.assertEqual(export.dropped_cards(existing, ["a", "b", "c", "d"]), [])
        for broken in ("{not json", '{"cards": "none"}', "[]"):
            (self.set_dir / "set.json").write_text(broken, encoding="utf-8")
            self.assertIsNone(export.existing_cards(self.set_dir), broken)

    def test_a_plan_says_what_a_push_would_change_without_rendering(self):
        cards = [{"id": "swsh3-88"}, {"id": "base1-91"}, {"id": "swsh3-89"}]
        plan = export.plan_set(cards, self.set_dir)
        self.assertEqual((plan["existing"], plan["added"], plan["changed"], plan["removed"], plan["moved"], plan["unchanged"]),
                         (None, cards, [], [], 0, 0), "no set yet: every card is new")
        self.push(cards)
        self.renders.clear()
        same = export.plan_set(cards, self.set_dir)
        self.assertEqual((same["existing"], same["added"], same["changed"], same["removed"], same["moved"], same["unchanged"]),
                         (3, [], [], [], 0, 3))
        edited = {"id": "swsh3-88", "edits": {"hp": "190"}}
        panel = {"id": "base1-91", "style": "panel", "foil": True}
        plan = export.plan_set([edited, panel, {"id": "sv1-1"}], self.set_dir)
        self.assertEqual(plan["added"], [{"id": "sv1-1"}])
        self.assertEqual([(item["id"], old, stale) for item, old, stale in plan["changed"]], [
            ("swsh3-88", {"edits": {}, "style": "hybrid", "attacks": True, "foil": False}, False),
            ("base1-91", {"edits": {}, "style": "hybrid", "attacks": True, "foil": False}, False)])
        self.assertEqual(plan["removed"], [("swsh3-89", "Larvitar")])
        self.assertEqual((plan["moved"], plan["unchanged"]), (0, 0), "the cards keep their places")
        moved = export.plan_set([cards[2], cards[0], cards[1]], self.set_dir)
        self.assertEqual((moved["moved"], moved["unchanged"], moved["removed"]), (3, 3, []), "a reorder renumbers them")
        self.assertEqual(self.renders, [], "planning renders nothing")
        self.assertEqual(export.existing_title(self.set_dir), "Homebrew Wave 1")
        self.assertEqual(export.existing_title(self.set_dir / "nope"), "")

    def strip_digests(self, *local_ids):
        """The set as an export from before records kept image digests wrote it (those cards, or all)."""
        record = self.record()
        for card in record["cards"]:
            if not local_ids or card["localId"] in local_ids:
                card["custom"].pop("files")
        (self.set_dir / "set.json").write_text(json.dumps(record), encoding="utf-8")

    def test_a_set_exported_before_records_kept_digests_has_its_images_taken_as_they_are(self):
        self.push([{"id": "swsh3-88"}, {"id": "base1-91"}])
        self.strip_digests()
        high, low = (self.set_dir / "001" / "high.webp").read_bytes(), (self.set_dir / "001" / "low.webp").read_bytes()
        kept = export.exported_cards(self.set_dir)
        self.assertEqual(sorted(kept), ["base1-91", "swsh3-88"])
        self.assertEqual(kept["swsh3-88"][0]["custom"]["files"], {"high": export._digest(high), "low": export._digest(low)})
        self.renders.clear()
        r = self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 1, ["swsh3-88"]), "only the changed card")
        self.assertTrue(all(set(c["custom"]["files"]) == {"high", "low"} for c in self.record()["cards"]),
                        "every record keeps digests now: the next push checks them")
        self.strip_digests()
        (self.set_dir / "002" / "high.webp").unlink()                 # a legacy card with an image gone is rendered
        self.renders.clear()
        r = self.push([{"id": "swsh3-88", "edits": {"hp": "190"}}, {"id": "base1-91"}])
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 1, ["base1-91"]))

    def test_the_plan_is_what_the_push_does_with_digests_missing_wrong_or_fine(self):
        cards = [{"id": "swsh3-88"}, {"id": "base1-91"}, {"id": "swsh3-89"}]
        self.push(cards)
        self.strip_digests("001")                                     # taken as it is
        (self.set_dir / "002" / "low.webp").write_bytes(b"not the image the record names")      # a hand edit
        plan = export.plan_set(cards, self.set_dir)
        self.assertEqual([(item["id"], stale) for item, _, stale in plan["changed"]], [("base1-91", True)])
        self.assertEqual((plan["unchanged"], plan["added"]), (2, []))
        self.renders.clear()
        r = self.push(cards)
        self.assertEqual((r["rendered"], r["reused"], self.renders), (1, 2, ["base1-91"]), "the plan said so")
        again = export.plan_set(cards, self.set_dir)
        self.assertEqual((again["changed"], again["unchanged"]), ([], 3), "and the push mended it")

    def test_rejects_like_the_zip(self):
        with self.assertRaises(ValueError):
            self.push([])
        with self.assertRaises(ValueError):
            export.write_set_dir([{"id": "swsh3-88"}], "", "hbw", self.root, read=None, render=None, apply=None,
                                 changelog=None)
        self.assertFalse(any(self.root.iterdir()))
