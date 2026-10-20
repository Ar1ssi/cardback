"""Holofoil for restyled cards the way Pokemon TCG Live draws it: a 1024x1024 front and white-plate masks in the
game's texture layout (the 733x1024 card plus 145 px of bleed each side, tcg.BLEED_X), and the shader / material
to draw them with. Needs a TCG Live extract (tcg.TCGLIVE): the masks start from the game's own plates.

The foil is the target era's:
- ex: Scarlet & Violet ex foil (SunPillar with the TEX_SouthernCross pattern, as on every SV ex in refs.TYPES).
  Its white plate is the V card's own plate for the art (the Pokemon left out of the foil, as the game drew it),
  with what the V frame and text covered filled in, and Absol ex's plate for the SV frame; the HP, icons and text
  drawn on it are cut out (black), the ex logo is Absol's, the name takes the bar's value (as Absol's does).
- Mega Evolution: SvHolo on the art window when the old card's base printing was foil (the old plate's art
  window, Pokemon left out), and a FlatSilver reverse holo when the old card had one. The reverse plate is
  Mega Evolution's (me1 Abra for Basics, Meganium for Stage cards) with the reference card's icons, Ability
  badge, Ability name and footer filled in, then this card's cut out.

Stage pieces (stage.py): SV ex plates print the tab, ring and strip in the frame's silver value (Alakazam ex's), with
the tab's letters, the portrait and the "Evolves from" line left out; on every other plate they are left out. Where
the tab's lettering changes, the plate first takes the tab's own value around it (retab).

What a composer drew comes from layers.recording(): every stamp and sprite is recorded under its kind.
"""
import functools
import io
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import cache, compose, frame, layers, mecompose, meframe, refs, stage, swsh, tcg
from .imaging import dilate

W, H = 733, 1024
TEX = 1024
ASSET_VERSION = "2"

# (shader, material, cast-and-cure pattern), as TCG Live's MaterialManifests name them
EX_FOIL = ("TPCi/Cards3D/HoloFoil/SunPillar", "MAT_Cards3D_SunPillar", "TEX_SouthernCross")
ME_HOLO = ("TPCi/Cards3D/HoloFoil/SvHolo", "MAT_Cards3D_SvHolo", None)
ME_REVERSE = ("TPCi/Cards3D/HoloFoil/FlatSilver", "MAT_Cards3D_FlatSilver", None)
NON_FOIL = "NonFoil"      # in the material name of a non-foil printing (old bundles: Standard_NonFoil_J)

# Drawn kinds left out of the foil (black on the plate).
STAGE_CUT = ("stage", "stage_text", "portrait", "evolves")
EX_CUT = ("hp", "type_icon", "icon", "row_icon", "badge", "ability_name", "text", "row_text", "footer",
          "stage_text", "portrait", "evolves")
ME_REVERSE_CUT = ("icon", "row_icon", "badge", "ability_name", "footer") + STAGE_CUT
EX_STAGE_REF = "sv03.5-065"   # Alakazam ex (Stage 2): its plate gives the SV ex Stage pieces' foil value
ME_REVERSE_REFS = {"basic": "me01-054", "stage": "me01-010"}   # Abra, Meganium: their reverse plates
FILL_GROW = 4      # px around a reference card's own marks that are filled in when its plate becomes a template
PANEL_INSET = 6    # the Mega Evolution text panel (meframe.PANEL) less its rim: all foil but the marks on it
FOOTER_ZONE = (40, 935, 330, 996)   # x0, y0, x1, y1: illustrator, regulation mark, set code, number, rarity
ROW_REACH = 40     # px either side a filled pixel takes its row's value from
EX_FILL_GROW = 5   # Absol's plate: its HP digits are a little bolder there than as set here
BORDER_FOIL = 255  # the SV ex border on Absol's plate where its art does not spill over: all foil


# ---------------------------------------------------------------- TCG Live data
@functools.lru_cache(maxsize=1)
def _manifests():
    """bundle -> [MaterialManifest records] from the extract's cardart/manifests.jsonl."""
    out = {}
    with open(Path(tcg.TCGLIVE, "cardart", "manifests.jsonl"), encoding="utf-8") as f:
        for line in f:
            records = json.loads(line)
            if records:
                out[records[0]["bundle"]] = records
    return out


def printings(card_id):
    """{'base': record, 'reverse': record or None} of the card's TCG Live printing (record: MaterialManifest JSON)."""
    if not tcg.TCGLIVE:
        raise SystemExit("foil needs a Pokemon TCG Live extract: pass --tcglive DIR (or set SVRESTYLE_TCGLIVE)")
    set_id, number = tcg.tcglive_id(card_id).rsplit("_", 1)
    records = _manifests().get(f"{set_id}_en_{int(number):03d}")
    if not records:
        raise ValueError(f"{card_id}: no MaterialManifest in the TCG Live extract")
    by = {r["manifest"]: r for r in records}
    return {"base": by.get("MaterialManifest"), "reverse": by.get("MaterialManifest_ph")}


def is_foil(record):
    return bool(record) and NON_FOIL not in (record.get("_f") or "") + (record.get("_s") or "")


def plate(record):
    """The record's white plate in card space: float32 733x1024, 0 (no foil) .. 255. A foil printing without a
    plate is foil all over."""
    name = record.get("_w")
    if not name:
        return np.full((H, W), 255, np.float32)
    png = Path(tcg.TCGLIVE, "cardart", "masks", name.split("_")[0], f"{name}.png")
    if png.exists():
        img = Image.open(png)
    else:
        import UnityPy
        bundle = Path(tcg.TCGLIVE, "cardart", "bundles", record["bundle"])
        img = next(o.read().image for o in UnityPy.load(str(bundle)).objects
                   if o.type.name == "Texture2D" and o.peek_name() == name)
    img = img.convert("L").resize((TEX, TEX), Image.BICUBIC)
    return np.asarray(img, np.float32)[:, tcg.BLEED_X:tcg.BLEED_X + W].copy()


def fill(img, where, radius=5):
    """Fill the `where` pixels of a 0..255 plate from their surroundings."""
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.inpaint(u8, where.astype(np.uint8), radius, cv2.INPAINT_TELEA).astype(np.float32)


def fill_rows(img, where, reach=ROW_REACH):
    """Fill the `where` pixels with the median of their row's unfilled pixels within `reach`: for plates whose
    shapes run across (the weakness row's lines), where inpainting leaves crosses. Pixels with none in reach
    (inside long marks) are inpainted."""
    out = img.copy()
    left = np.zeros_like(where)
    for y in np.flatnonzero(where.any(axis=1)):
        keep = ~where[y]
        for x in np.flatnonzero(where[y]):
            lo, hi = max(0, x - reach), min(W, x + reach + 1)
            near = img[y, lo:hi][keep[lo:hi]]
            if near.size:
                out[y, x] = np.median(near)
            else:
                left[y, x] = True
    return fill(out, left) if left.any() else out


def cut(img, drawn, kinds):
    """Black out what was drawn of the given kinds."""
    cover = np.zeros((H, W), np.float32)
    for k in kinds:
        if k in drawn:
            np.maximum(cover, drawn[k], out=cover)
    return img * (1 - np.clip(cover, 0, 1))


def _cached(name, build):
    path = cache.path("assets", f"foil-v{ASSET_VERSION}", name)
    if path.exists():
        return np.asarray(Image.open(path), np.float32)
    img = build()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    return img


# ---------------------------------------------------------------- templates
@functools.lru_cache(maxsize=1)
def _absol():
    """Absol ex's plate and what the composer draws on its name bar (calibrated on Absol: same places)."""
    with layers.recording() as drawn:
        compose.name_bar_text(np.zeros((H, W, 4), np.float32), {"prefix": "", "name": "Absol", "hp": "210"},
                              frame.profile("Darkness"))
    return plate(printings(refs.ABSOL.card_id)["base"]), drawn


def ex_template():
    """Absol ex's plate with its name, ex logo and HP filled in, and its border plain foil (Absol's art spills
    over its border, and the plate with it)."""
    def build():
        absol, drawn = _absol()
        marks = (drawn["name"] > 0.1) | (drawn["logo"] > 0.1) | (drawn["hp"] > 0.1)
        out = fill(absol, dilate(marks, EX_FILL_GROW))
        out[:, :frame.INNER_X0] = out[:, frame.INNER_X1:] = out[frame.PANEL_Y1:] = BORDER_FOIL
        return out

    return _cached("ex-template.png", build)


def put_logo(out, logo):
    """Absol's ex logo plate, moved to where this card's logo was drawn (same height, other x)."""
    absol, drawn = _absol()
    cols = lambda a: np.flatnonzero(a.max(axis=0) > 0.5)
    dx = int(cols(logo)[0] - cols(drawn["logo"])[0])
    moved = np.roll(absol, dx, axis=1)
    return out * (1 - logo) + moved * logo


def me_reverse_template(stage):
    """Mega Evolution reverse-holo plate for a Basic (stage 0) or Stage card, the reference card's own marks
    filled in."""
    which = "stage" if stage else "basic"
    ref = ME_REVERSE_REFS[which]

    def build():
        with layers.recording() as drawn:
            mecompose.compose_card(ref)
        marks = cut(np.ones((H, W), np.float32), drawn, ME_REVERSE_CUT) < 0.9
        reverse = plate(printings(ref)["reverse"])
        # On the panel, take every dark mark: TCGdex spells energy in words, so the reference's attacks are not
        # always set where its print has them.
        x0, y0, x1, y1 = mecompose.meframe.PANEL
        i = PANEL_INSET
        out = reverse.copy()
        fx0, fy0, fx1, fy1 = FOOTER_ZONE
        for zx0, zy0, zx1, zy1 in ((x0 + i, y0 + i, x1 - i, y1 - i), (fx0, fy0, fx1, fy1)):
            zone = np.zeros((H, W), bool)
            zone[zy0:zy1, zx0:zx1] = True
            dark = dilate(zone & (reverse < 128), FILL_GROW) & zone
            out[dark] = np.median(reverse[zone & ~dark])       # both zones are one flat value
            marks &= ~zone
        return fill_rows(out, dilate(marks, FILL_GROW))

    return _cached(f"me-reverse-{which}.png", build)


@functools.lru_cache(maxsize=1)
def ex_stage_value():
    """Plate value of the SV ex ring and strip (Alakazam ex)."""
    pieces = (meframe.stage_sprite(2)[..., 3] > 240) & stage.part_mask(("ring", "strip"))
    return float(np.median(plate(printings(EX_STAGE_REF)["base"])[pieces]))


def retab(img, drawn):
    """Old tab letters off the plate: where new ones were pasted, the tab's own value from just around them."""
    where = drawn.get("stage_tab", np.zeros((H, W), np.float32)) > 0.05
    box = np.zeros((H, W), bool)
    x0, y0, x1, y1 = stage.LETTER_BOX
    box[y0:y1, x0:x1] = True
    around = dilate(where, 4) & ~dilate(where, 1) & box
    if not around.any():
        return img
    out = img.copy()
    out[where] = np.median(img[around])
    return out


def ex_stage(img, drawn):
    """SV ex plate: the Stage pieces drawn take the frame's silver value."""
    if "stage" not in drawn:
        return img
    return img * (1 - drawn["stage"]) + ex_stage_value() * drawn["stage"]


# ---------------------------------------------------------------- masks per card
def ex_plate(card_id, drawn, alpha):
    """White plate of a V card restyled as ex (compose.compose, hybrid)."""
    art = plate(printings(card_id)["base"])
    covered, _ = swsh.text_mask(tcg.scan(card_id))
    # the art moved down with the art scan (swsh.clean)
    d = swsh.SHIFT_DOWN
    art[d:], art[:d] = art[:-d].copy(), art[0]
    moved = np.empty_like(covered)
    moved[d:], moved[:swsh.TOP_MASKED + d] = covered[:-d], True
    art = fill(art, moved)
    framed = drawn.get("frame", np.zeros((H, W), np.float32))
    out = art * (1 - framed) + ex_template() * framed
    if "logo" in drawn:
        out = put_logo(out, drawn["logo"])
    return cut(ex_stage(retab(out, drawn), drawn), drawn, EX_CUT) * (alpha / 255)


def me_holo_plate(card_id, card, drawn, alpha):
    """SvHolo plate of a regular card restyled as Mega Evolution: the old plate's art window only."""
    old = plate(printings(card_id)["base"])
    if card.get("old_stage"):
        boxes = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in mecompose.OLD_STAGE_OVERLAY:
            boxes[y0:y1, x0:x1] = True
        old = fill(old, boxes)
    x0, y0, x1, y1 = mecompose.meframe.ART
    out = np.zeros((H, W), np.float32)
    out[y0:y1, x0:x1] = old[y0:y1, x0:x1]
    return cut(out, drawn, STAGE_CUT) * (alpha / 255)


def me_reverse_plate(card, drawn, alpha):
    return cut(retab(me_reverse_template(card["stage"]), drawn), drawn, ME_REVERSE_CUT) * (alpha / 255)


# ---------------------------------------------------------------- textures
def front_texture(canvas):
    """733x1024 card -> 1024x1024 RGB front; the bleed repeats the card's edge columns."""
    rgb = np.clip(canvas[..., :3], 0, 255).astype(np.uint8)
    right = TEX - W - tcg.BLEED_X
    return np.pad(rgb, ((0, 0), (tcg.BLEED_X, right), (0, 0)), mode="edge")


def plate_texture(img):
    right = TEX - W - tcg.BLEED_X
    return np.pad(np.clip(img, 0, 255).astype(np.uint8), ((0, 0), (tcg.BLEED_X, right)))


def build(card_id, layout, canvas, drawn, card):
    """-> {'front': RGB 1024x1024, 'base': (foil, plate) or None, 'reverse': (foil, plate) or None}
    foil = (shader, material, pattern); plate = L 1024x1024. layout: 'ex' or 'me'."""
    alpha = canvas[..., 3]
    old = printings(card_id)
    if layout == "ex":
        base, reverse = (EX_FOIL, ex_plate(card_id, drawn, alpha)), None
    else:
        base = (ME_HOLO, me_holo_plate(card_id, card, drawn, alpha)) if is_foil(old["base"]) else None
        reverse = (ME_REVERSE, me_reverse_plate(card, drawn, alpha)) if old["reverse"] else None
    pack = lambda p: None if p is None else (p[0], plate_texture(p[1]))
    return {"front": front_texture(canvas), "base": pack(base), "reverse": pack(reverse)}


# ---------------------------------------------------------------- Scarlet & Violet / Mega Evolution cards, edited in place
def legacy_shader(name):
    """Shader name as the client knows it (old bundles use pre-rename names); WebApp tools/assets/cards_import.py."""
    if name.startswith("TPCi/Cards3D"):
        return name
    rep = {"Cards/": "TPCi/Cards3D/", "_Amplify": "", "_J": "", "SW": "Sw", "SV": "Sv", "Standard_": "",
           "Foil/": "", "HoloFoil_": "HoloFoil/", "Cracked_Ice": "CrackedIce"}
    return re.sub(r"(^Cards/)|(_J)|(_Amplify)|(Standard_)|(Foil/)|(HoloFoil_)|(SW)|(SV)|(Cracked_Ice)",
                  lambda m: rep.get(m.group(0), m.group(0)), name)


def material_name(f):
    """MaterialManifest _f -> material name; WebApp tools/assets/cards_import.py."""
    if f == "HoloFoil_SWSecret_J":
        f = "HoloFoil_SwSecret_Amplify_J"
    if f == "HoloFoil_SWHolo_J":
        f = "HoloFoil_SwHolo_Amplify_J"
    rep = {"HoloFoil": "", "Standard": "", "Amplify": "", "J": "", "SW": "Sw", "SV": "Sv", "_": ""}
    return "MAT_Cards3D_" + re.sub(r"(HoloFoil)|(Standard)|(Amplify)|(J)|(_)|(SW)|(SV)", lambda m: rep[m.group(0)], f)


def texture(record, name):
    """A texture of the record's bundle at its own size (uint8 array), or None."""
    if not name:
        return None
    png = Path(tcg.TCGLIVE, "cardart", "masks", name.split("_")[0], f"{name}.png")
    if png.exists():
        return _plain(Image.open(png))
    import UnityPy
    bundle = Path(tcg.TCGLIVE, "cardart", "bundles", record["bundle"])
    img = next((o.read().image for o in UnityPy.load(str(bundle)).objects
                if o.type.name == "Texture2D" and o.peek_name() == name), None)
    return None if img is None else _plain(img)


def _plain(img):
    return np.asarray(img if img.mode in ("L", "RGB", "RGBA") else img.convert("RGBA"))


def modern_front(card_id, canvas):
    """The card's own 1024x1024 front (its real bleed) with the edited card in the middle."""
    front = tcg.tcglive_front(card_id)
    if front is None or front.size != (TEX, TEX):
        return front_texture(canvas)
    out = np.asarray(front.convert("RGB")).copy()
    out[:, tcg.BLEED_X:tcg.BLEED_X + W] = np.clip(canvas[..., :3], 0, 255).astype(np.uint8)
    return out


def build_modern(card_id, card, canvas, drawn, taken_off, original):
    """Foil of a Scarlet & Violet / Mega Evolution card edited in place (modern.compose_card): its own printings,
    each plate with what was taken off filled in and what was set cut out (ex: everything printed, as on SV ex
    plates; reverse holo: what Mega Evolution reverse plates leave out). Same shape as build()."""
    from . import modern
    alpha = canvas[..., 3]
    old = printings(card_id)
    out = {"front": modern_front(card_id, canvas), "base": None, "reverse": None}
    for key in ("base", "reverse"):
        record = old[key]
        if not is_foil(record):
            continue
        img = plate(record)
        if taken_off is not None and taken_off.any():
            img = fill(img, taken_off)
        img = retab(img, drawn)
        if card["family"] == "ex" and key == "base":
            if "logo" in drawn:
                before = modern._drawn(modern._name, original)["logo"]
                cols = lambda a: np.flatnonzero(a.max(axis=0) > 0.5)
                moved = np.roll(plate(record), int(cols(drawn["logo"])[0] - cols(before)[0]), axis=1)
                img = img * (1 - drawn["logo"]) + moved * drawn["logo"]
            img = cut(ex_stage(img, drawn), drawn, EX_CUT)
        elif key == "reverse":
            img = cut(img, drawn, ME_REVERSE_CUT)
        else:
            img = cut(img, drawn, STAGE_CUT)
        spec = (legacy_shader(record.get("_s") or ""), material_name(record.get("_f") or ""), record.get("_cc") or None)
        extra = {"etch": texture(record, record.get("_e")), "cf": texture(record, record.get("_cf"))}
        out[key] = (spec, plate_texture(img * (alpha / 255)), extra)
    return out


def files(foil, stem, card_id):
    """-> (record, {file name: PNG bytes}): <stem>-front.png, -wp.png, -wp-ph.png; record is <stem>-foil.json's
    content (file names in it relative to it)."""
    out = {}

    def png(name, arr):
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "PNG")
        out[name] = buf.getvalue()
        return name

    record = {"id": card_id, "tcglive": tcg.tcglive_id(card_id), "front": png(f"{stem}-front.png", foil["front"])}
    for key, suffix in (("base", "wp"), ("reverse", "wp-ph")):
        if foil[key] is None:
            record[key] = None
            continue
        (shader, material, pattern), img, *more = foil[key]
        extra = more[0] if more else {}
        record[key] = {"shader": shader, "material": material, "wp": png(f"{stem}-{suffix}.png", img),
                       "cc": pattern}
        for tex in ("etch", "cf"):          # carried over unchanged from the card's own printing
            name = f"{stem}-{suffix}-{tex}.png"
            record[key][tex] = png(name, extra[tex]) if extra.get(tex) is not None else None
    out[f"{stem}-foil.json"] = json.dumps(record, indent=2).encode()
    return record, out


def save(foil, out_dir, stem, card_id):
    """Write files() into out_dir; -> the -foil.json path."""
    _, written = files(foil, stem, card_id)
    for name, data in written.items():
        (Path(out_dir) / name).write_bytes(data)
    return Path(out_dir) / f"{stem}-foil.json"


def card_space(texture):
    """A 1024x1024 texture's card part (733x1024), for previews."""
    return texture[:, tcg.BLEED_X:tcg.BLEED_X + W]
