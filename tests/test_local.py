"""Cards set up from scans found online (local.py): their data, ids and search, the scans' registration, the Japanese
prints' logo and rule box, the Mega ex name's ink, and the holofoil made for them (custom_foil.py). No network, model
or fonts: scans and sprites are made up.
Run: python -m unittest discover tests"""
import unittest
from unittest import mock

import cv2
import numpy as np

from svrestyle import anglicize, compose, custom_foil, edit, foil, local, megarule, modern, tcg

W, H = 733, 1024
IDS = ("me07-058", "mej7-030", "mej7-067")


class Registry(unittest.TestCase):
    def test_ids_are_card_ids_the_editor_takes_as_mega_evolution_prints(self):
        for cid in IDS:
            self.assertTrue(edit.valid_card_id(cid), cid)
            self.assertTrue(modern.is_modern(cid), cid)
            self.assertTrue(local.is_local(cid))
        self.assertFalse(local.is_local("me01-086"))

    def test_tcg_answers_for_them_without_tcgdex(self):
        with mock.patch.object(tcg.cache, "fetch_json", side_effect=AssertionError("no TCGdex call")):
            self.assertEqual(tcg.card("mej7-067")["name"], "Mega Garchomp Z ex")
            self.assertEqual(tcg.set_info("me07")["name"], "Flowing Fists")
            self.assertIsNone(tcg.tcglive_front("me07-058", root="/nowhere"))

    def test_card_reads_are_copies(self):
        tcg.card("mej7-030")["attacks"][0]["damage"] = "999"
        self.assertEqual(tcg.card("mej7-030")["attacks"][0]["damage"], "160")

    def test_read_as_mega_ex_prints_that_keep_their_type(self):
        want = {"me07-058": ("Metal", 1, "Riolu", 2), "mej7-030": ("Psychic", 0, "", 1),
                "mej7-067": ("Dragon", 2, "Gabite", 1)}
        for cid, (type_name, stage, evolves, retreat) in want.items():
            card = modern.read_card(cid)
            self.assertEqual((card["family"], card["type"], card["stage"], card["evolves_from"], card["retreat"]),
                             ("ex", type_name, stage, evolves, retreat), cid)
            self.assertTrue(card["name"].startswith("Mega ") and card["name"].endswith(" Z"))
            self.assertFalse(card["retype"])            # no TCG Live printing to recolour on

    def test_type_change_is_refused_even_with_an_extract(self):
        with mock.patch.object(tcg, "TCGLIVE", "/extract"):
            self.assertFalse(modern.megaretype.supports("mej7-067", {"family": "ex"}))

    def test_search_by_name_finds_them_first(self):
        with mock.patch.object(tcg, "_fresh_json", return_value=[]):
            found = tcg.find("garchomp z")
        self.assertEqual([c["id"] for c in found["cards"]], ["mej7-067"])
        self.assertEqual(found["cards"][0]["thumb"], "/api/original/mej7-067.png")
        with mock.patch.object(tcg, "_fresh_json", return_value=[]):
            self.assertEqual(tcg.find("", {"types": ["Dragon"]})["cards"], [])     # filters: TCGdex's only

    def test_garchomp_text_prints_the_dragon_symbol(self):
        effect = tcg.card("mej7-067")["attacks"][0]["effect"]
        self.assertIn("{N}", effect)


class Badge(unittest.TestCase):
    def test_the_badge_names_the_pokemon_without_its_form_letter(self):
        self.assertEqual(megarule.label_of({"name": "Mega Garchomp Z"}), (["Garchomp"], True))
        self.assertEqual(megarule.label_of({"name": "Mega Charizard X"}), (["Charizard"], True))
        self.assertEqual(megarule.label_of({"name": "Mega Absol"}), (["Absol"], True))


class Registration(unittest.TestCase):
    def frame(self, dx=0.0):
        rng = np.random.default_rng(3)
        img = rng.uniform(80, 120, (H, W, 3)).astype(np.float32)
        for y in (878, 912, 930, 998):                      # the row's rules, the rule box's rims
            img[y:y + 3, 40:700] = 230
        for x in (60, 210, 260, 700):
            img[870:1000, x:x + 3] = 20
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        m = np.float32([[1, 0, dx], [0, 1, 0]])
        return cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REPLICATE)

    def test_a_print_two_pixels_off_is_moved_onto_the_layout(self):
        ref, off = self.frame(), self.frame(2.0)
        dx, dy = local.offset(off, ref)
        self.assertAlmostEqual(dx, 2.0, delta=0.3)
        self.assertAlmostEqual(dy, 0.0, delta=0.3)
        fixed = local.registered(off, ref)
        self.assertLess(abs(local.offset(fixed, ref)[0]), 0.3)

    def test_a_print_on_the_layout_is_left_alone(self):
        ref = self.frame()
        self.assertIs(local.registered(ref, ref), ref)


def logo_shape():
    """A made-up "ex" logo: strokes thicker than the logo's dark core (2 * LOGO_CORE + 1 px), as the print's are."""
    shape = np.zeros((50, 80), np.uint8)
    cv2.circle(shape, (20, 25), 14, 1, 13)                   # "e"
    cv2.line(shape, (42, 8), (72, 42), 1, 13)               # "x"
    cv2.line(shape, (72, 8), (42, 42), 1, 13)
    return shape


class Logo(unittest.TestCase):
    def test_the_logo_is_found_by_its_dark_strokes_and_cut_with_its_shape(self):
        shape = logo_shape()
        scan = np.full((H, W, 4), 230.0, np.float32)
        for y in range(20, 60, 6):                          # a name glyph's thin dark outline: not the logo
            scan[y, 150:250, :3] = 20
        core = cv2.erode(shape, np.ones((5, 5), np.uint8))     # the dark fill inside its light outline
        scan[39:89, 371:451, :3] = np.where(core[..., None] > 0, 15.0, np.where(shape[..., None] > 0, 200.0, 230.0))
        alpha = cv2.GaussianBlur(shape.astype(np.float32), (3, 3), 0.5)
        with mock.patch.object(anglicize.frame, "ex_logo",
                               return_value=((None, alpha, 0, 0, (0, 0)), 0, 80)):
            at = anglicize.find_logo(scan)
            self.assertLessEqual(abs(at[0] - 371) + abs(at[1] - 39), 3)    # the prints: 0-1 px (mej7-030, -067)
            (sprite, left, right), where = anglicize.logo_sprite(scan, at)
        self.assertEqual(sprite[4], (sprite[2], sprite[3]))
        self.assertGreater(where[39:89, 371:451][shape > 0].mean(), 0.9)
        self.assertLess(left, right)

    def test_no_logo_is_an_error(self):
        alpha = logo_shape().astype(np.float32)
        with mock.patch.object(anglicize.frame, "ex_logo", return_value=((None, alpha, 0, 0, (0, 0)), 0, 80)):
            with self.assertRaises(ValueError):
                anglicize.find_logo(np.full((H, W, 4), 230.0, np.float32))


class Row(unittest.TestCase):
    def test_the_japanese_slots_leave_the_retreat_circles_and_the_rules(self):
        off = anglicize.row_off()
        self.assertFalse(off[:, 527:].any())                 # Retreat circles (540.2 +- 12.8)
        self.assertFalse(off[880].any() or off[911].any())   # the row's rules
        self.assertFalse(off[895, 205:215].any())            # the bar after the weakness
        self.assertTrue(off[895, 60] and off[895, 320])


class MegaName(unittest.TestCase):
    def test_a_mega_ex_print_keeps_its_yellow_outlined_name_when_renamed(self):
        card = {"family": "ex", "type": "Dragon", "name": "Garchomp Test", "prefix": "", "hp": "360"}
        with mock.patch.object(modern.compose, "name_bar_text") as bar, \
                mock.patch.object(modern.frame, "profile", return_value={}):
            modern._name(None, card, mega=True)
            self.assertIs(bar.call_args.kwargs["name_ink"], compose.MEGA_NAME)
            modern._name(None, card)                          # not a Mega's name, not told it is one
            self.assertIsNone(bar.call_args.kwargs["name_ink"])
            modern._name(None, dict(card, name="Mega Garchomp Z"))
            self.assertIs(bar.call_args.kwargs["name_ink"], compose.MEGA_NAME)

    def test_mega_name_ink_is_yellow_with_a_dark_outline(self):
        fill, (edge, width, opacity) = compose.MEGA_NAME
        self.assertTrue(fill[0] > 200 and fill[1] > 200 and fill[2] < 60)
        self.assertLess(max(edge), 40)
        self.assertGreater(width, 2)


class Foil(unittest.TestCase):
    def test_a_local_card_is_its_own_printing(self):
        rec = foil.printings("mej7-067")
        self.assertIsNone(rec["reverse"])
        base = rec["base"]
        self.assertEqual(base["_local"], "mej7-067")
        self.assertEqual(foil.legacy_shader(base["_s"]), foil.ME_ULTRA)
        self.assertEqual(foil.material_name(base["_f"]), "MAT_Cards3D_SvUltra")
        self.assertTrue(foil.is_foil(base))
        self.assertEqual(foil.peers(base), [])

    def test_plate_texture_and_field_come_from_custom_foil(self):
        base = foil.printings("mej7-030")["base"]
        plate = np.full((H, W), 7.0, np.float32)
        etch = np.full((1024, 1024), 9, np.uint8)
        with mock.patch.object(custom_foil, "white_plate", return_value=plate), \
                mock.patch.object(custom_foil, "etch", return_value=etch), \
                mock.patch.object(custom_foil.tcg, "scan", return_value=np.full((H, W, 4), 255.0, np.float32)):
            self.assertIs(foil.plate(base), plate)
            self.assertIs(foil.texture(base, base["_e"]), etch)
            self.assertIsNone(foil.texture(base, "other"))
            field, own = foil.field(base)
        self.assertEqual(float(field.min()), 255.0)
        self.assertEqual(float(own.min()), 1.0)

    def test_files_name_no_tcg_live_card(self):
        front = np.zeros((1024, 1024, 3), np.uint8)
        plate = np.zeros((1024, 1024), np.uint8)
        built = {"front": front, "base": ((foil.ME_ULTRA, "MAT_Cards3D_SvUltra", None), plate, {"etch": plate}),
                 "reverse": None}
        record, files = foil.files(built, "mej7-067-modern", "mej7-067")
        self.assertIsNone(record["tcglive"])
        self.assertEqual(record["base"]["etch"], "mej7-067-modern-wp-etch.png")
        self.assertIn("mej7-067-modern-foil.json", files)

    def test_art_etch_follows_the_art_and_stays_plain_off_it(self):
        scan = np.full((H, W, 4), 120.0, np.float32)
        scan[300:600, 360:, :3] = 240.0                       # an edge in the art
        e = custom_foil.art_etch(scan)
        self.assertLess(e[450, 360], foil.ETCHED)             # etched along it
        self.assertEqual(e[450, 100], 255)                    # flat art: plain
        self.assertEqual(e[50, 360], 255)                     # the name bar: plain

    def test_rule_words_cut_letters_not_the_box(self):
        scan = np.full((H, W, 4), 160.0, np.float32)
        scan[948:995, 280:700, :3] = (235, 225, 30)           # the yellow field
        scan[960:972, 420:520, :3] = 20                       # a word in it
        scan[924:944, 300:400, :3] = 10                       # the black pill above it
        scan[930:938, 320:380, :3] = (235, 225, 30)           # its yellow letters
        words = custom_foil.rule_words(scan)
        self.assertTrue(words[965, 470] and words[934, 350])
        self.assertFalse(words[985, 600] or words[941, 395])  # the field and the pill stay foil


if __name__ == "__main__":
    unittest.main()
