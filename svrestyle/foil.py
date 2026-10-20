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
import warnings
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
# Edits whose plates stay as printed on cards edited in place, and the kinds those edits draw.
PLATE_KEPT = ("moves", "row")
PLATE_KEPT_KINDS = ("text", "icon", "badge", "ability_name", "row_icon", "row_text")
EX_STAGE_REF = "sv03.5-065"   # Alakazam ex (Stage 2): its plate gives the SV ex Stage pieces' foil value
ME_REVERSE_REFS = {"basic": "me01-054", "stage": "me01-010"}   # Abra, Meganium: their reverse plates
FILL_GROW = 4      # px around a reference card's own marks that are filled in when its plate becomes a template
PANEL_INSET = 6    # the Mega Evolution text panel (meframe.PANEL) less its rim: all foil but the marks on it
FOOTER_ZONE = (40, 935, 330, 996)   # x0, y0, x1, y1: illustrator, regulation mark, set code, number, rarity
ROW_REACH = 40     # px either side a filled pixel takes its row's value from
EX_FILL_GROW = 5   # Absol's plate: its HP digits are a little bolder there than as set here
BORDER_FOIL = 255  # the SV ex border on Absol's plate where its art does not spill over: all foil
# A card's own foil field (field()): its plate with what is printed on it taken off.
MARK_DARK = 48       # plate value below which a pixel is a cut-out (text, icons, art window), not foil
MARK_GROW = 3        # px around cut-outs: their anti-aliased rims
PEERS = 24           # plates a card's foil field is rebuilt from
PEER_SIZE = 128      # px: plates are compared at this size
PEER_SHARED = 0.12   # share of the card both plates must show foil on to be compared
PEER_MATCH = 0.5     # correlation of two plates' detail for them to share foil shapes
FIT_RADII = (16, 40, 90)   # px: windows a peer is fitted to the card in
FIT_MIN = 0.15       # share of a window the card and the peer must both show foil on for the fit
FIT_EPS = 25.0       # plate levels squared: keeps the gain sane on flat foil
FIT_GAIN = 3.0
FIT_TOL = 6.0        # plate levels: fits this much worse weigh e times less
BRIDGE_SIGMAS = (6, 12, 24, 48, 96, 192)   # px: blurs that bridge what no peer shows
OWN_FEATHER = 3      # px: soft edge between a card's own foil and the template's
# How a card's own plate prints a kind of ink (ink_value)
INK_MIN = 20         # px: less printed ink than this is not judged
INK_CLEAR = 2        # px: the ground is read this far out from all the ink
INK_SIGMA = 4        # px: and interpolated over this
INK_VISIBLE = 24     # plate levels: ink this far from its ground shows on the plate (block compression: ~10)
TAB_BIN = 8          # plate levels per histogram bin when finding the tab's own value
INK_AREA = 10        # px around a kind's ink that its taking off may reach (modern REACH + MARK_GROW + rims)


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


def _plate_file(record):
    name = record.get("_w")
    png = Path(tcg.TCGLIVE, "cardart", "masks", name.split("_")[0], f"{name}.png") if name else None
    return png if png and png.exists() else None


def _thumb(png):
    """A plate PNG at PEER_SIZE, card part only, float32 0..255."""
    img = Image.open(png).convert("L").resize((PEER_SIZE, PEER_SIZE), Image.BOX)
    a = np.asarray(img, np.float32)
    x0 = round(tcg.BLEED_X * PEER_SIZE / TEX)
    return a[:, x0:x0 + round(W * PEER_SIZE / TEX)]


def _detail(thumb):
    return thumb - cv2.GaussianBlur(thumb, (0, 0), 3)


def _foil_at(thumb):
    return cv2.erode((thumb >= MARK_DARK).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def peers(record):
    """Other printings of the same set and kind (base / reverse) whose plates have this one's foil shapes (the
    detail correlates where neither is cut out): the same pattern under other cards' text. Most alike first."""
    own_png = _plate_file(record)
    if own_png is None:
        return []
    set_id = record["bundle"].split("_en_")[0]
    own = _thumb(own_png)
    own_ok, own_detail = _foil_at(own), _detail(own)
    out = []
    for bundle, records in _manifests().items():
        if bundle == record["bundle"] or bundle.split("_en_")[0] != set_id:
            continue
        other = next((r for r in records if r["manifest"] == record["manifest"]), None)
        png = _plate_file(other) if other and is_foil(other) else None
        if png is None:
            continue
        thumb = _thumb(png)
        both = own_ok & _foil_at(thumb)
        if both.mean() < PEER_SHARED:
            continue
        a, b = own_detail[both], _detail(thumb)[both]
        spread = np.sqrt((a * a).mean() * (b * b).mean())
        alike = (a * b).mean() / spread if spread > 0.5 else 0
        if alike >= PEER_MATCH:
            out.append((alike, other))
    out.sort(key=lambda t: -t[0])
    return [r for _, r in out[:PEERS]]


def _box(img, r):
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), normalize=False)


def transfer(img, ok, others, radius):
    """Predict img where it is not ok from other plates of the same pattern: around each pixel, each other plate is
    fitted to img (gain and offset, on the pixels both show), and the fits are blended by how well they hold there.
    -> (prediction, bool: predicted). others: [(plate, its ok)]."""
    num = np.zeros((H, W), np.float32)
    den = np.zeros((H, W), np.float32)
    for other, other_ok in others:
        m = (ok & other_ok).astype(np.float32)
        n = _box(m, radius)
        enough = n > FIT_MIN * (2 * radius + 1) ** 2
        n = np.maximum(n, 1)
        mo, mi = _box(other * m, radius) / n, _box(img * m, radius) / n
        cov = _box(other * img * m, radius) / n - mo * mi
        var = _box(other * other * m, radius) / n - mo * mo
        gain = np.clip(cov / (var + FIT_EPS), 0, FIT_GAIN)
        offset = mi - gain * mo
        guess = gain * other + offset
        err = _box(np.abs(img - (gain * other + offset)) * m, radius) / n
        w = np.where(enough & other_ok, np.exp(-err / FIT_TOL), 0).astype(np.float32)
        num += w * guess
        den += w
    have = den > 1e-3
    return np.where(have, num / np.maximum(den, 1e-6), 0).astype(np.float32), have


def bridge(img, gaps):
    """Fill gaps with the foil around them, smoothly (normalised blurs, wider until every gap is reached): for the
    art window, where inpainting draws streaks to the middle."""
    out = img.copy()
    known = ~gaps
    left = gaps.copy()
    for sigma in BRIDGE_SIGMAS:
        value, weight = _smooth(img, known, sigma)
        take = left & (weight > 0.02)
        out[take] = value[take]
        left &= ~take
    out[left] = float(np.median(img[known])) if known.any() else 255
    return out


def _smooth(img, ok, sigma):
    """Gaussian blur of img over the ok pixels only (normalised convolution), and how much ok each pixel saw."""
    weight = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    value = cv2.GaussianBlur(np.where(ok, img, 0).astype(np.float32), (0, 0), sigma)
    return value / np.maximum(weight, 1e-6), weight


def field(record):
    """The printing's own foil with everything printed on it taken off -> (plate, own: 0..1 how much of each pixel
    is this printing's foil). Cut-outs (text, icons, figure) are filled from the same set's printings of the same
    kind that share its foil shapes (peers), whose text and icons sit elsewhere, toned to this card's own foil
    around each gap (transfer). What no peer shows either (the art window, a border without foil) is bridged
    smoothly and is not own. Restyled plates keep this card's pattern and strength where it has them: the target
    era's template says where the foil goes, and gives it where this card has none."""
    def build():
        img = plate(record)
        marks = dilate(img < MARK_DARK, MARK_GROW)
        if not marks.any():
            return img, np.ones((H, W), np.float32)
        others = []
        for other in peers(record):
            p = plate(other)
            others.append((p, ~dilate(p < MARK_DARK, MARK_GROW)))
        out = img.copy()
        left = marks.copy()
        for radius in FIT_RADII:                     # tight fits first; wider ones reach the middle of big gaps
            if not others or not left.any():
                break
            guess, have = transfer(img, ~marks, others, radius)
            take = left & have
            out[take] = guess[take]
            left &= ~take
        own = cv2.GaussianBlur((~left).astype(np.float32), (0, 0), OWN_FEATHER)
        if left.any():
            out = bridge(out, left)
        speckless = cv2.medianBlur(np.clip(out, 0, 255).astype(np.uint8), 5).astype(np.float32)
        out[marks] = speckless[marks]        # peers' fits leave single-pixel misses on text rims
        return out, own * 255

    stem = f"field-{record['bundle']}-{record['manifest']}"
    paths = [cache.path("assets", f"foil-v{ASSET_VERSION}", f"{stem}{end}.png") for end in ("", "-own")]
    if not all(path.exists() for path in paths):
        for path, img in zip(paths, build()):
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)
    img, own = (np.asarray(Image.open(path), np.float32) for path in paths)
    return img, own / 255


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


def _tab_letters(img, drawn):
    """(bool HxW: the tab lettering the plate prints where a tab was pasted, the tab's own value) or None."""
    where = drawn.get("stage_tab", np.zeros((H, W), np.float32)) > 0.05
    box = np.zeros((H, W), bool)
    x0, y0, x1, y1 = stage.LETTER_BOX
    box[y0:y1, x0:x1] = True
    letters_at = box & where
    if letters_at.sum() < INK_MIN:
        return None
    # the tab's own value: the commonest there (letters are thin strokes on it; the rim around differs)
    counts = np.bincount(np.clip(img[letters_at] / TAB_BIN, 0, 255 // TAB_BIN).astype(int), minlength=256 // TAB_BIN + 1)
    common = int(np.argmax(np.convolve(counts, np.ones(3), "same")))
    near = letters_at & (np.abs(img / TAB_BIN - (common + 0.5)) <= 1.5)
    tab = float(np.median(img[near]))
    return letters_at & (np.abs(img - tab) > INK_VISIBLE), tab


def tab_letter_value(img, drawn):
    """The plate value of the tab's printed letters, or None when the plate does not show them."""
    found = _tab_letters(img, drawn)
    if found is None or found[0].sum() < INK_MIN:
        return None
    return float(np.median(img[found[0]]))


def retab(img, drawn):
    """Old tab letters off the plate where a new tab was pasted: the letters it prints and the new ones' place take
    the tab's own value. The rest of the tab stays as printed."""
    found = _tab_letters(img, drawn)
    if found is None:
        return img
    old, tab = found
    new = drawn.get("stage_text", np.zeros((H, W), np.float32)) > 0.05
    where = (dilate(old, 1) | new) & (drawn["stage_tab"] > 0.05)
    out = img.copy()
    out[where] = tab
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


def me_reverse_plate(card_id, card, drawn, alpha):
    """Mega Evolution reverse holo: where the template has foil, this card's own reverse foil (field)."""
    template = me_reverse_template(card["stage"])
    own_foil, own = field(printings(card_id)["reverse"])
    mixed = own_foil * (template / 255) * own + template * (1 - own)
    return cut(retab(mixed, drawn), drawn, ME_REVERSE_CUT) * (alpha / 255)


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
        reverse = (ME_REVERSE, me_reverse_plate(card_id, card, drawn, alpha)) if old["reverse"] else None
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


def printed_ink(original, groups):
    """What the card prints for the edited groups, by kind, as the composers set it (coverage 0..1)."""
    from . import modern
    out = {}
    if "name" in groups:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # the printed name's fit was reported when the card was set
            out.update(modern._drawn(modern._name, original))
    if "stage" in groups and original["stage"]:
        out.update(modern._drawn(modern._evolves, original))
        cx, cy, r = meframe.PORTRAIT_DISC
        yy, xx = np.mgrid[:H, :W]
        out["portrait"] = np.clip(r - 0.5 - np.hypot(xx - cx, yy - cy), 0, 1).astype(np.float32)
    return out


def ink_value(img, cover, around):
    """The plate value a printed kind of ink has where it is printed (cover), against the plate interpolated from
    just around each stroke (outside `around`, all the ink): (value, ground), or None where the plate does not
    show it (the foil runs through, as on most reverse holos' names)."""
    core = cover > 0.5
    if core.sum() < INK_MIN:
        return None
    ground, weight = _smooth(img, ~dilate(around, INK_CLEAR), INK_SIGMA)
    judged = core & (weight > 0.05)
    if judged.sum() < INK_MIN:
        return None
    if np.median(np.abs(img[judged] - ground[judged])) < INK_VISIBLE:
        return None
    return float(np.median(img[judged])), float(np.median(ground[judged]))


def build_modern(card_id, card, canvas, drawn, original):
    """Foil of a Scarlet & Violet / Mega Evolution card edited in place (modern.compose_card): its own printings,
    edited. Edits to the name or the Stage pieces re-make their part of each plate the way the card's own plate
    prints that kind of ink (ink_value): ink the plate shows is taken off (cut-outs filled from the card's own
    foil field, a lit ink from the plate around it) and set again at the printed value; ink the plate does not
    show leaves it as printed. Kinds the card does not print (Stage pieces on a Basic made Stage) follow the era:
    ex: cut out, as on SV ex plates; reverse holo: what Mega Evolution reverse plates leave out. Abilities,
    attacks and the weakness row keep the plates as printed (PLATE_KEPT). Same shape as build()."""
    from . import modern
    alpha = canvas[..., 3]
    old = printings(card_id)
    groups = tuple(g for g in modern.changed(original, card) if g not in PLATE_KEPT)
    taken_off = modern.taken_off(card_id, original, groups, card) if groups else None
    printed = printed_ink(original, groups)
    shed = modern.stage_off(original, card) if "stage" in groups else np.zeros((H, W), bool)
    drawn = {k: v for k, v in drawn.items() if k not in PLATE_KEPT_KINDS}
    if original["stage"]:
        drawn.pop("stage", None)            # the card's own pieces, set again where they were: as printed
    out = {"front": modern_front(card_id, canvas), "base": None, "reverse": None}
    for key in ("base", "reverse"):
        record = old[key]
        if not is_foil(record):
            continue
        img = plate(record)
        if shed.any():
            img = fill(img, shed)
        learned = {}
        if taken_off is not None:
            mask, row = taken_off
            off = mask | row
            around = np.zeros((H, W), bool)
            for cover in printed.values():
                around |= cover > 0.05
            own_field = None
            for kind, cover in printed.items():
                value = ink_value(img, cover, around)
                learned[kind] = value
                if value is None:
                    continue
                area = off & dilate(cover > 0.05, INK_AREA)
                if value[0] < value[1]:                       # cut out: the foil under it, from the field
                    if own_field is None:
                        own_field = field(record)[0]
                    img = np.where(area, own_field, img)
                else:                                          # lit ink on plainer ground
                    img = fill(img, area)
        letters = tab_letter_value(img, drawn) if original["stage"] else None
        img = retab(img, drawn)
        if letters is not None and "stage_text" in drawn:
            learned["stage_text"] = (letters, None)
        for kind, value in learned.items():
            if value is not None and kind in drawn and kind != "logo":
                img = img * (1 - drawn[kind]) + value[0] * drawn[kind]
        if card["family"] == "ex" and key == "base" and "logo" in drawn and learned.get("logo"):
            before = printed["logo"]
            cols = lambda a: np.flatnonzero(a.max(axis=0) > 0.5)
            moved = np.roll(plate(record), int(cols(drawn["logo"])[0] - cols(before)[0]), axis=1)
            img = img * (1 - drawn["logo"]) + moved * drawn["logo"]
        rest = {k: v for k, v in drawn.items() if k not in learned}
        if card["family"] == "ex" and key == "base":
            img = cut(ex_stage(img, rest), rest, EX_CUT)
        elif key == "reverse":
            img = cut(img, rest, ME_REVERSE_CUT)
        else:
            img = cut(img, rest, STAGE_CUT)
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
