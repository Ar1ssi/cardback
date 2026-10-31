"""Holofoil for restyled cards the way Pokemon TCG Live draws it: a 1024x1024 front and white-plate masks in the
game's texture layout (the 733x1024 card plus 145 px of bleed each side, tcg.BLEED_X), and the shader / material
to draw them with. Needs a TCG Live extract (tcg.TCGLIVE): the masks start from the game's own plates.

The foil is the target era's:
- ex: Scarlet & Violet ex foil (SunPillar with the TEX_SouthernCross pattern, as on every SV ex in refs.TYPES).
  Its white plate is the V card's own plate for the art (the Pokemon left out of the foil, as the game drew it),
  with what the V frame and text covered filled in, and Absol ex's plate for the SV frame; the HP, icons and text
  drawn on it are cut out (black), the ex logo is Absol's, the name takes the bar's value (as Absol's does).
- Mega Evolution: SvHolo on the art window when the old card's base printing was foil (the old plate's art
  window, Pokemon left out), and on every card a FlatSilver reverse holo in Sword & Shield's (Gen 8) pattern of the
  card's type (gen8_plate): where Mega Evolution's reverse plate (me1 Abra for Basics, Meganium for Stage cards,
  with the reference card's icons, Ability badge, Ability name and footer filled in, then this card's cut out)
  shows foil. 1-prize Pokemon kept as printed and Trainers take the Gen 8 reverse holo too.

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

from . import cache, compose, frame, layers, mecompose, meframe, refs, retype, stage, swsh, tcg
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
STAGE_CUT = ("stage", "stage_text", "portrait", "evolves", "mega_rule")     # mega_rule: megarule.KIND
EX_CUT = ("hp", "type_icon", "icon", "row_icon", "badge", "ability_name", "text", "row_text", "footer",
          "stage_text", "portrait", "evolves", "mega_rule")
ME_REVERSE_CUT = ("icon", "row_icon", "badge", "ability_name", "footer") + STAGE_CUT
# Edits to the text blocks on cards edited in place (abilities and attacks, the weakness row), and the kinds they
# draw. Their plates are the card's own foil field where the old block's cut-outs were (block_cutout_value).
BLOCKS = ("moves", "row")
BLOCK_KINDS = {"moves": ("text", "icon", "badge", "ability_name"), "row": ("row_icon", "row_text")}
# Scarlet & Violet Trainers: TCG Live's reverse holo, the Poké Ball pattern plate every Trainer of Scarlet & Violet's sets
# (sv01 to sv08.5) shares; one print per kind (the plates differ only in the header's lettering and the rule pill).
# Mega Evolution's Trainers print a plain plate. What a print cuts out of it is its footer (the illustrator, the boxes).
TRAINER_FOIL_REFS = {"Item": "sv01-168", "Supporter": "sv01-166", "Stadium": "sv01-167", "Tool": "sv01-169"}
TRAINER_REVERSE_CUT = ("footer",)
# Gen 8 (Sword & Shield) reverse holo, the standard on every 1-prize Pokemon and every Trainer but ACE SPEC: TCG Live's
# chevron pattern, the card's type's symbol in each tile (Trainers: Poke Balls). Per type a Basic whose plate prints
# the pattern with the fewest marks (Ultra Ball for Trainers); its cut-outs, its weakness row and footer (from
# GEN8_ROWS_FROM: the row's icons and the footer's marks are not all black) and GEN8_HOLES (white lettering the foil
# runs through, a Trainer's header) are filled from the pattern GEN8_PERIOD px up or down. Fairy, which Sword & Shield
# does not print, is Psychic's with the Fairy symbol (fairy.coin_symbol) set in each tile (gen8_swapped).
GEN8_REFS = {"Grass": "swsh4-10", "Fire": "swsh2-31", "Water": "swsh2-39", "Lightning": "swsh4-53",
             "Psychic": "swsh4-67", "Fighting": "swsh4-90", "Darkness": "swsh8-162", "Metal": "swsh4-123",
             "Dragon": "swsh9-110", "Colorless": "swsh4-143", "Trainer": "swsh9-150"}
GEN8_DERIVED = {"Fairy": "Psychic"}
GEN8_VERSION = "1"
GEN8_PERIOD = 120               # px: the chevrons repeat this far down each column
GEN8_FACE = (32, 30, 702, 996)  # x0, y0, x1, y1: the part of the plate the pattern covers (not the card's rim)
GEN8_ROWS_FROM = 876
GEN8_HOLES = {"Darkness": ((0, 0, W, 108), (0, 630, W, 715)),   # Carvanha's name and HP, its attack
              "Trainer": ((0, 0, W, 140),)}                     # the header: TRAINER and the kind
GEN8_LIT = (36, 56)             # plate levels: a reverse plate shows no foil up to the first (SV's dark ground is
                                # 20-30), all from the second on (its pattern's darkest is 59-65)
GEN8_HEADER = 130              # px: a Gen 9 Trainer plate's header ends above this
# Symbols set in each tile (gen8_swapped): the base's own symbol is where these types' templates differ from it
GEN8_PEERS = ("Grass", "Fire", "Water", "Lightning", "Fighting", "Metal", "Colorless")
GEN8_SYMBOL_DIFF = 18           # plate levels: a symbol differs from its tile by more than this
GEN8_SYMBOL_REACH = 40          # px round a tile's centre its symbol lies in
GEN8_PANEL = (530, 860)         # y range the tile centres are found in (the print's own pattern)
GEN8_SV_REFS = {"basic": "sv01-001", "stage": "sv01-002"}   # Sprigatito, Floragato: the SV frame's reverse plates
EX_STAGE_REF = "sv03.5-065"   # Alakazam ex (Stage 2): its plate gives the SV ex Stage pieces' foil value
ME_REVERSE_REFS = {"basic": "me01-054", "stage": "me01-010"}   # Abra, Meganium: their reverse plates
FILL_GROW = 4      # px around a reference card's own marks that are filled in when its plate becomes a template
PANEL_INSET = 6    # the Mega Evolution text panel (meframe.PANEL) less its rim: all foil but the marks on it
BLOCK_REACH = 3     # px: a block's printed cut-outs may sit this far from where the composers set its ink
BLOCK_SHOWN = 0.3   # share of a kind's ink that must fall on cut-outs for the plate to cut that kind out
FOOTER_ZONE = (40, 935, 330, 996)   # x0, y0, x1, y1: illustrator, regulation mark, set code, number, rarity
ROW_REACH = 40     # px either side a filled pixel takes its row's value from
BADGE_ETCH = "me02.5-152"   # Mega Dragonite ex: its etch edges an Ability badge's pill (a Mega ex given an Ability)
ETCH_GROW = 4      # px round an icon or the badge whose etch is its own
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
# A type change between white words (Darkness) and dark ones: reverse plates cut white lettering out and run the foil
# through dark lettering (every ME type's plates, me1-me2), so the card's lettering and the row's labels follow.
TYPE_TEXT = ("name", "hp", "text", "row_text")          # kinds an edit sets in the type's word ink
LETTERING_ZONES = [(0, 0, W, 100), (0, 512, W, 914)]    # name bar; panel and weakness row (not the footer, always cut)
LABEL_REF, LABEL_HEIGHT = "me01-087", 16   # Spiritomb (Darkness): its reverse plate cuts the row's labels; px tall
GRAIN_SIGMA, GRAIN_SIZE, GRAIN_NEAR = 3, 0.7, 12   # px: the plate's grain is finer than GRAIN_SIGMA; its specks'
                                                 # size; it is read this near the fills
LETTER_FILL, LETTER_REACHED = (2, 5, 12, 30), 1e-3   # px blurs of the foil round a lettering cut-out that fill it
LETTER_MAX = 2000        # px: a cut-out shape larger than this is no lettering
LETTER_RIM = 6           # px round a cut-out the plate still shows it (its edge brightens up to 233 against 229)
SPECK_MAX, SPECK_REACH = 120, 8   # px: a cut-out this small this near the lettering is part of it
OWN_KNOWN = 0.5      # field's own weight above which a filled cut-out is the card's foil, not a guess
# How a card's own plate prints a kind of ink (ink_value)
INK_MIN = 20         # px: less printed ink than this is not judged
INK_CLEAR = 2        # px: the ground is read this far out from all the ink
INK_SIGMA = 4        # px: and interpolated over this
INK_VISIBLE = 24     # plate levels: ink this far from its ground shows on the plate (block compression: ~10)
TAB_BIN = 8          # plate levels per histogram bin when finding the tab's own value
# An ACE SPEC plate cuts the text out a little bolder than the print sets it: the set text's coverage is raised by
# this, fitted on Prime Catcher's effect text (dark plate pixels in it: 10754 printed, 10142 set again)
ACE_CUT_GAIN = 6.0
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


def me_reverse_plate(card, drawn, alpha):
    """Mega Evolution reverse holo: the template's plain FlatSilver plate, as every Mega Evolution card prints it,
    whatever pattern the old card's reverse had (Sword & Shield chevrons, XY and Sun & Moon type symbols)."""
    return cut(retab(me_reverse_template(card["stage"]), drawn), drawn, ME_REVERSE_CUT) * (alpha / 255)


# ---------------------------------------------------------------- Gen 8 reverse holo
def _gen8_filled(card_id, holes=()):
    """The reference's reverse plate with its cut-outs, its weakness row and footer and `holes` ((x0, y0, x1, y1)
    boxes) filled from the pattern GEN8_PERIOD px up or down, the nearest first: the pattern over the whole face."""
    img = plate(printings(card_id)["reverse"])
    x0, y0, x1, y1 = GEN8_FACE
    face = np.zeros((H, W), bool)
    face[y0:y1, x0:x1] = True
    valid = face & ~dilate(img < MARK_DARK, MARK_GROW)
    valid[GEN8_ROWS_FROM:] = False
    for bx0, by0, bx1, by1 in holes:
        valid[by0:by1, bx0:bx1] = False
    out, left = img.copy(), face & ~valid
    for k in range(1, H // GEN8_PERIOD + 1):
        for s in (k * GEN8_PERIOD, -k * GEN8_PERIOD):
            src, moved = np.zeros((H, W), bool), np.zeros((H, W), np.float32)
            if s > 0:
                src[:-s], moved[:-s] = valid[s:], img[s:]
            else:
                src[-s:], moved[-s:] = valid[:s], img[:s]
            take = left & src
            out[take] = moved[take]
            left &= ~take
    return fill(out, left) if left.any() else out


def _tile_centres(own):
    """The pattern's tile centres: found on the panel (GEN8_PANEL) where the symbols are, each column's run on
    every GEN8_PERIOD px over the card."""
    blur = cv2.GaussianBlur(own.astype(np.float32), (0, 0), 9)
    peak = (blur == cv2.dilate(blur, np.ones((61, 61), np.uint8))) & (blur > 0.15)
    peak[:GEN8_PANEL[0]] = False
    peak[GEN8_PANEL[1]:] = False
    ys, xs = np.nonzero(peak)
    columns, run = [], []
    for i in np.argsort(xs):
        if run and xs[i] - xs[run[-1]] > 30:
            columns.append(run)
            run = []
        run.append(i)
    columns.append(run)
    out = []
    for run in columns:
        if len(run) < 2:                 # a lone peak: a cut-out's rim, not a column of tiles
            continue
        x, phase = float(np.median(xs[run])), float(np.median(ys[run] % GEN8_PERIOD))
        out += [(x, phase + k * GEN8_PERIOD) for k in range(-1, H // GEN8_PERIOD + 2)]
    return out


def gen8_swapped(base, glyph):
    """base's Gen 8 pattern with its symbol swapped in every tile for glyph (coverage, 512x512, as
    fairy.coin_symbol): the base's symbol is taken off (inpainted from its tile), and glyph set at its size in its
    brightest value."""
    own_t = gen8_pattern(base)
    diff = np.zeros((H, W), bool)
    for peer in GEN8_PEERS:
        diff |= np.abs(gen8_pattern(peer) - own_t) > GEN8_SYMBOL_DIFF
    diff = dilate(diff, 2)
    rough = fill(own_t, diff, 4)
    own = diff & (np.abs(own_t - rough) > GEN8_SYMBOL_DIFF)
    centres = _tile_centres(own)
    yy, xx = np.mgrid[:H, :W]
    zone = np.zeros((H, W), bool)
    for cx, cy in centres:
        zone |= np.hypot(xx - cx, yy - cy) < GEN8_SYMBOL_REACH
    ground = fill(own_t, zone & diff, 4)
    sizes = []
    for cx, cy in centres:
        m = own & (np.hypot(xx - cx, yy - cy) < GEN8_SYMBOL_REACH)
        if GEN8_PANEL[0] < cy < GEN8_PANEL[1] and m.sum() > 300:
            ys, xs = np.nonzero(m)
            sizes.append(max(np.ptp(xs), np.ptp(ys)))
    gy, gx = np.nonzero(glyph > 0.5)
    scale = float(np.median(sizes)) / max(np.ptp(gx), np.ptp(gy))
    gcx, gcy = (gx.min() + gx.max()) / 2, (gy.min() + gy.max()) / 2
    x0, y0, x1, y1 = GEN8_FACE
    face = np.zeros((H, W), np.float32)
    face[y0:y1, x0:x1] = 1
    out = ground.copy()
    for cx, cy in centres:
        m = own & (np.hypot(xx - cx, yy - cy) < GEN8_SYMBOL_REACH)
        if m.sum() < 100:
            continue
        shape = cv2.warpAffine(glyph, np.float32([[scale, 0, cx - gcx * scale], [0, scale, cy - gcy * scale]]),
                               (W, H), flags=cv2.INTER_LINEAR) * face
        out = out * (1 - shape) + float(np.percentile(own_t[m], 90)) * shape
    return np.where(face > 0, out, own_t)


@functools.lru_cache(maxsize=None)
def gen8_pattern(kind):
    """The Gen 8 reverse-holo pattern over the whole face for a type, or "Trainer": float32 733x1024 (a type Sword &
    Shield has no pattern for, and no derived one, takes Colorless's)."""
    if kind not in GEN8_REFS and kind not in GEN8_DERIVED:
        kind = "Colorless"

    def build():
        if kind in GEN8_DERIVED:
            from . import fairy
            return gen8_swapped(GEN8_DERIVED[kind], fairy.coin_symbol(kind.lower()))
        return _gen8_filled(GEN8_REFS[kind], GEN8_HOLES.get(kind, ()))

    return _cached(f"gen8-v{GEN8_VERSION}-{kind.lower()}.png", build)


def generic_reverse(record):
    """Whether a print's reverse holo is the generic one (FlatSilver, nothing etched or cast on it) that the Gen 8
    reverse holo replaces; a card's own foil (any other shader, an etch) stays."""
    return (legacy_shader(record.get("_s") or "") == ME_REVERSE[0]
            and not any(record.get(k) for k in ("_e", "_cf", "_cc")))


def gen8_plate(kind, own):
    """A Pokemon's Gen 8 reverse plate: the type's pattern where its own reverse plate (Mega Evolution's flat one, or
    a Scarlet & Violet print's in its own pattern; what the card prints cut out) shows foil (GEN8_LIT)."""
    lo, hi = GEN8_LIT
    return gen8_pattern(kind) * np.clip((own - lo) / (hi - lo), 0, 1)


def gen8_trainer_plate(kind):
    """A Trainer's Gen 8 reverse plate: the Gen 9 plate of its kind (TRAINER_FOIL_REFS: the header's lettering, the
    rule pill) with Sword & Shield's Poke Ball chevrons on its pattern's field (its largest lit shape under the
    header)."""
    gen9 = plate(printings(TRAINER_FOIL_REFS[kind])["reverse"])
    lit = (gen9 > MARK_DARK).astype(np.uint8)
    lit[:GEN8_HEADER] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lit, connectivity=4)
    field = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(field, gen8_pattern("Trainer"), gen9)


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
        # every one, whether the old print had a reverse holo or not: the Gen 8 one is the standard
        reverse = (ME_REVERSE, gen8_plate(card["type"], me_reverse_plate(card, drawn, alpha)))
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


def printed_blocks(original, groups, card=None):
    """What the card prints in the edited text blocks, by kind, as the composers set it (coverage 0..1); in the
    weakness row, only what the edit to card takes off."""
    from . import modern
    out = {}
    if "moves" in groups:
        out.update(modern._drawn(modern._moves, original, compose.WHITE))
    if "row" in groups:
        out.update(modern._drawn(modern._row, modern.row_off(original, card) if card else original, compose.WHITE))
    return out


def block_cutout_value(img, cover, marks):
    """The plate value of a block's printed ink where the plate cuts it out (energy circles, the Ability badge),
    or None where the foil runs through it. Read off the cut-outs near where the composers set the ink: TCGdex
    spells energy in words, so a block is not always set exactly where its print has it."""
    core = cover > 0.5
    if core.sum() < INK_MIN:
        return None
    hit = dilate(core, BLOCK_REACH) & marks
    if hit.sum() < BLOCK_SHOWN * core.sum():
        return None
    return float(np.median(img[hit]))


@functools.lru_cache(maxsize=1)
def row_labels():
    """Coverage (HxW) of the weakness row's labels as a white-lettering plate cuts them (LABEL_REF's: the labels
    sit in the same place on every card): its small cut-outs in the row, not the rules or icons."""
    from . import modern
    img = plate(printings(LABEL_REF)["reverse"]).astype(np.float32)
    x0, y0, x1, y1 = modern.ROW_LABELS
    cut_px = np.zeros((H, W), np.uint8)
    cut_px[y0:y1, x0:x1] = img[y0:y1, x0:x1] < MARK_DARK
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cut_px, connectivity=8)
    letter = np.zeros(n, bool)
    letter[1:] = (stats[1:, cv2.CC_STAT_AREA] <= SPECK_MAX) & (stats[1:, cv2.CC_STAT_HEIGHT] <= LABEL_HEIGHT)
    near = dilate(letter[labels], MARK_GROW)
    ground = float(np.median(img[y0:y1, x0:x1][img[y0:y1, x0:x1] >= MARK_DARK]))
    return np.where(near, np.clip(1 - img / ground, 0, 1), 0).astype(np.float32)


def lettering(words):
    """Where a type change moves lettering between white and dark: the retyped words (retype.WORDS) in the name
    bar, panel and row, and the row's labels."""
    zones = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in LETTERING_ZONES:
        zones[y0:y1, x0:x1] = 1
    words = np.zeros((H, W), np.float32) if words is None else words
    return np.maximum(words * zones, row_labels())


def fill_from(img, ok):
    """img filled from the ok pixels: each pixel from those nearest it (the finest of LETTER_FILL that reaches some)."""
    out, missing = img.astype(np.float32).copy(), np.ones(img.shape, bool)
    for sigma in LETTER_FILL:
        value, weight = _smooth(img, ok, sigma)
        take = missing & (weight > LETTER_REACHED)
        out[take] = value[take]
        missing &= ~take
    return out


def grain_like(img, ok):
    """Noise with the plate's own fine grain (its spread over the ok pixels, robust to the odd edge), so a fill
    does not show as a smooth patch."""
    img = img.astype(np.float32)
    fine = img - cv2.GaussianBlur(img, (0, 0), GRAIN_SIGMA)
    spread = 1.4826 * float(np.median(np.abs(fine[ok]))) if ok.any() else 0.0
    noise = cv2.GaussianBlur(np.random.default_rng(0).standard_normal(img.shape).astype(np.float32), (0, 0), GRAIN_SIZE)
    return noise * (spread / max(float(noise.std()), 1e-6))


def lettering_cutouts(img, letters):
    """The plate's cut-outs of lettering, with the rim round them the plate still shows: every cut-out shape the
    lettering touches, and the small marks beside it the word finder leaves (apostrophes, brackets, dots). Other
    cut-outs (energy symbols, the Ability badge and red name, the row's rules) stay, rims and all."""
    words = letters > 0.05
    cut_px = img < MARK_DARK
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cut_px.astype(np.uint8), connectivity=8)
    hit = np.zeros(n, bool)
    hit[np.unique(labels[dilate(words, MARK_GROW) & cut_px])] = True
    small = stats[:, cv2.CC_STAT_AREA] <= SPECK_MAX
    near = np.zeros(n, bool)
    near[np.unique(labels[dilate(words, SPECK_REACH) & cut_px])] = True
    zones = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in LETTERING_ZONES:
        zones[y0:y1, x0:x1] = True
    hit[0] = False
    hit &= stats[:, cv2.CC_STAT_AREA] <= LETTER_MAX     # not the border or the art window a letter touches
    lettered = (hit | (small & near))[labels] & zones
    kept = dilate(cut_px & ~lettered, MARK_GROW) & ~dilate(words, MARK_GROW)
    return dilate(lettered | (words & zones), LETTER_RIM) & ~kept


def build_modern(card_id, card, canvas, drawn, original):
    """Foil of a Scarlet & Violet / Mega Evolution card edited in place (modern.compose_card): its own printings,
    edited. Edits to the name or the Stage pieces re-make their part of each plate the way the card's own plate
    prints that kind of ink (ink_value): ink the plate shows is taken off (cut-outs filled from the card's own
    foil field, a lit ink from the plate around it) and set again at the printed value; ink the plate does not
    show leaves it as printed. Abilities, attacks and the weakness row (BLOCKS): the old block's cut-outs (energy
    circles, badge) take the card's own foil field, and the new block's ink of each kind is cut at the value the
    print gives it; kinds the foil runs through stay as printed. Kinds the card does not print (Stage pieces on a
    Basic made Stage) follow the era: ex: cut out, as on SV ex plates; reverse holo: what Mega Evolution reverse
    plates leave out. Same shape as build().

    A Mega ex made another type (megaretype) starts from its print recoloured: its white plate cuts its re-inked words
    out (to Darkness) or closes them (from it), a Mega Hyper Rare's etch takes the new symbol. An edit made in place
    (inplace: Energy, damage) keeps the plates, but for print in the text block the plate cuts out (white text): it is
    cut again. The etch
    of an ex whose abilities, attacks or weakness row are set again follows them (etch_edits)."""
    from . import inplace, megaretype, modern
    alpha = canvas[..., 3]
    old = printings(card_id)
    scan, printed_as = modern.print_of(card_id, original, card, record=False)
    mega = printed_as is not original          # an ex (Mega or Scarlet & Violet) recoloured on its new type
    edited = modern.changed(printed_as, card)
    swapped = any(k in drawn for k in (inplace.OFF, inplace.ON, inplace.ICON))
    groups = tuple(g for g in edited if g not in BLOCKS)
    blocks = tuple(g for g in edited if g in BLOCKS and not (g == "moves" and swapped))
    taken_off = modern.taken_off(card_id, printed_as, groups, card, scan) if groups else None
    block_off = None
    if blocks:
        mask, row = modern.taken_off(card_id, printed_as, blocks, card, scan)
        block_off = mask | row
    printed = printed_ink(printed_as, groups)
    in_blocks = printed_blocks(printed_as, blocks, card)
    shed = modern.stage_off(printed_as, card) if "stage" in groups else np.zeros((H, W), bool)
    if printed_as["stage"]:
        drawn.pop("stage", None)            # the card's own pieces, set again where they were: as printed
    retyped = drawn.pop(retype.WORDS, None)
    flip = ("type" in edited and printed_as["family"] == "regular"
            and retype.white_ink(printed_as["type"]) != retype.white_ink(card["type"]))
    white_now = flip and retype.white_ink(card["type"])
    letters_off = lettering(retyped) if flip else None
    out = {"front": modern_front(card_id, canvas), "base": None, "reverse": None}
    for key in ("base", "reverse"):
        record = old[key]
        if not is_foil(record):
            continue
        img = plate(record)
        if mega and key == "base":
            img = megaretype.plate(img, original["type"], card["type"],
                                   np.zeros((H, W), np.float32) if retyped is None else retyped,
                                   megaretype.family_of(original))
        if shed.any():
            img = fill(img, shed)
        if inplace.OFF in drawn:
            img = reprinted(img, drawn[inplace.OFF], drawn.get(inplace.ON))
        learned = {}
        if block_off is not None:
            marks = dilate(img < MARK_DARK, MARK_GROW)
            for kind, cover in in_blocks.items():
                value = block_cutout_value(img, cover, marks)
                learned[kind] = None if value is None else (value, None)
            gone = block_off & marks
            if gone.any():
                own_field, own = field(record)
                gone &= own > OWN_KNOWN       # what every printing cuts (the row's lines) stays cut, not bridged
                img = np.where(gone, own_field, img)
                # the filled cut-out's edge, softened so no rim shows where it was (cut-outs kept stay sharp)
                inner = cv2.erode(gone.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                seam = dilate(gone, 2) & ~inner & ~(marks & ~gone)
                img[seam] = cv2.GaussianBlur(img, (0, 0), 1.5)[seam]
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
        if flip and key == "reverse":
            for kind in TYPE_TEXT:
                learned[kind] = None          # set by the type below, not as the old type's plate printed it
            if not white_now:                 # the old white lettering's cut-outs: the plate's foil runs through
                gone = lettering_cutouts(img, letters_off)
                # from the foil round it only, past the rims: the row's rules and the icons are cut-outs too
                source = ~dilate(gone | (img < MARK_DARK), MARK_GROW)
                img = np.where(gone, fill_from(img, source) + grain_like(img, source & dilate(gone, GRAIN_NEAR)), img)
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
            if flip and white_now:
                img = cut(img, {**drawn, "lettering": letters_off}, TYPE_TEXT + ("lettering",))
        else:
            img = cut(img, rest, STAGE_CUT)
        spec = (legacy_shader(record.get("_s") or ""), material_name(record.get("_f") or ""), record.get("_cc") or None)
        if key == "reverse" and card["family"] == "regular" and generic_reverse(record):
            img, spec = gen8_plate(card["type"], img), ME_REVERSE     # a 1-prize Pokemon: the Gen 8 reverse holo
        extra = {"etch": texture(record, record.get("_e")), "cf": texture(record, record.get("_cf"))}
        if mega:
            extra["etch"] = megaretype.etch(extra["etch"], original, card["type"])
        if blocks and extra["etch"] is not None and printed_as["family"] == "ex" and card["name"].startswith("Mega "):
            extra["etch"] = etch_edits(extra["etch"], printed_as, card, drawn, blocks)
        out[key] = (spec, plate_texture(img * (alpha / 255)), extra)
    if card["family"] == "regular" and out["base"] is None and out["reverse"] is None:
        # a print with no foil at all: the Gen 8 reverse holo on its era's frame (a reference print's plate, the
        # marks on its text panel filled in: the card's own icons are not cut out)
        frame_refs = ME_REVERSE_REFS if card_id.startswith("me") else GEN8_SV_REFS
        img = plate(printings(frame_refs["stage" if card["stage"] else "basic"])["reverse"])
        x0, y0, x1, y1 = mecompose.meframe.PANEL
        i = PANEL_INSET
        panel = img[y0 + i:y1 - i, x0 + i:x1 - i]
        panel[panel < GEN8_LIT[1]] = 255
        img = cut(img, drawn, ME_REVERSE_CUT)
        out["reverse"] = (ME_REVERSE, plate_texture(gen8_plate(card["type"], img) * (alpha / 255)), {})
    return out


def reprinted(img, off, on):
    """A plate after print in the text block was set again in place (inplace: symbols, damage; off what was taken
    off, on what was set): where the plate cuts it out (white text), the old cut-outs close (filled from round them)
    and the new print is cut at the old one's value; where the foil runs through the text, it stays as it is."""
    gone = off > 0.5
    around = dilate(gone, 2 * MARK_GROW) & ~dilate(gone, MARK_GROW)
    if not gone.any() or not around.any():
        return img
    cut, ground = float(np.median(img[gone])), float(np.median(img[around]))
    if ground - cut < INK_VISIBLE:
        return img
    img = fill(img, dilate(off > 0.05, MARK_GROW))
    return img if on is None else img * (1 - on) + cut * on


def _parts(cover):
    """Connected shapes of a coverage: [(bool HxW, (x0, y0, x1, y1))]."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((cover > 0.5).astype(np.uint8))
    return [(lab == i, (st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3])) for i in range(1, n)]


@functools.lru_cache(maxsize=1)
def _badge_etch():
    """(BADGE_ETCH's etch in card space, its Ability badge's box): the badge found on its print by its red."""
    rgb = tcg.scan(BADGE_ETCH)[..., :3]
    badge = retype._badge(retype._print_red(rgb))
    ys, xs = np.nonzero(badge)
    r = retype.BADGE_RIM
    rec = printings(BADGE_ETCH)["base"]
    etch = _planes(texture(rec, rec["_e"]))[:, tcg.BLEED_X:tcg.BLEED_X + W, 0].astype(np.float32)
    return etch, (xs.min() + r, ys.min() + r, xs.max() + 1 - r, ys.max() + 1 - r)


def _planes(img):
    """A texture as rows x columns x channels (an L-mode one as one channel)."""
    return img[..., None] if img.ndim == 2 else img


def etch_edits(etch, original, card, drawn, groups):
    """A Mega ex's etch (1024x1024, TCG Live's layout) after its abilities, attacks or weakness row are set again:
    the etch rings the print's cost and row icons and edges the Ability badge's pill, so the old ones go (filled with
    the etch round them) and the new ones get the print's own: an icon this card prints, the badge BADGE_ETCH's."""
    from . import modern
    if etch.shape[0] != TEX:
        return etch
    out = etch.copy()
    view = _planes(out)[:, tcg.BLEED_X:tcg.BLEED_X + W]
    colours = min(3, view.shape[2])                      # not an alpha channel
    e = _planes(etch)[:, tcg.BLEED_X:tcg.BLEED_X + W, 0].astype(np.float32)
    printed_as = lambda draw, c, kinds: modern._cover(modern._drawn(draw, c, compose.WHITE), kinds)
    kinds = []
    if "moves" in groups:
        icons = printed_as(modern._moves, original, ("icon",))
        kinds += [("icon", icons, _parts(icons)), ("badge", printed_as(modern._moves, original, ("badge",)), [])]
    if "row" in groups:
        # only the row's changed sides are taken off and set again (modern.row_off / row_on)
        kinds += [("row_icon", printed_as(modern._row, modern.row_off(original, card), ("row_icon",)),
                   _parts(printed_as(modern._row, original, ("row_icon",))))]
    for kind, gone, own in kinds:
        for mask, _ in _parts(gone):                                # the old ones go
            hole = dilate(mask, ETCH_GROW)
            ring = dilate(hole, 4) & ~hole
            view[hole, :colours] = np.median(e[ring])
        new = drawn.get(kind)
        donor = (e, own[0][1]) if own else _badge_etch() if kind == "badge" else None
        if new is None or donor is None:
            continue
        src, (sx0, sy0, sx1, sy1) = donor
        scx, scy = (sx0 + sx1) // 2, (sy0 + sy1) // 2
        for mask, (x0, y0, x1, y1) in _parts(new):                  # the new ones get the print's own, centred
            hw, hh = (x1 - x0) // 2 + ETCH_GROW, (y1 - y0) // 2 + ETCH_GROW
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            patch = src[scy - hh:scy + hh, scx - hw:scx + hw]
            where = dilate(mask, ETCH_GROW)[cy - hh:cy + hh, cx - hw:cx + hw]
            region = view[cy - hh:cy + hh, cx - hw:cx + hw]
            for ch in range(colours):
                region[..., ch] = np.where(where, patch, region[..., ch])
    return out


def build_trainer(card_id, card, canvas, drawn):
    """Foil of a Trainer set on the Scarlet & Violet frame of its kind (trainer.compose_card; not an ACE SPEC card:
    build_ace): the Gen 8 reverse holo, FlatSilver on Sword & Shield's Poké Ball chevrons in the Gen 9 plate of its
    kind (gen8_trainer_plate), with the card's footer cut out, whether the old print had a reverse holo or not.
    Same shape as build()."""
    modern_print = card.get("era") == "modern"
    out = {"front": modern_front(card_id, canvas) if modern_print else front_texture(canvas), "base": None,
           "reverse": None}
    img = cut(gen8_trainer_plate(card["kind"]), drawn, TRAINER_REVERSE_CUT)
    out["reverse"] = (ME_REVERSE, plate_texture(img * (canvas[..., 3] / 255)), {})
    return out


def build_ace(card_id, card, canvas, drawn, original):
    """Foil of a Scarlet & Violet ACE SPEC Trainer edited in place (ace.compose_card): its own AceFoil printing,
    edited. What the old name, text and illustrator print on the plate is taken off where the plate shows it (cut
    out: the foil under it from the card's own foil field; lit: filled from the plate around it) and the new ink is
    set at the value the print gives that kind (ink_value); ink the plate does not show leaves it as printed. A card
    nothing is edited on keeps its own plate. Same shape as build()."""
    from . import ace
    alpha = canvas[..., 3]
    record = printings(card_id)["base"]
    out = {"front": modern_front(card_id, canvas), "base": None, "reverse": None}
    if not is_foil(record):
        return out
    img = plate(record)
    groups = ace.changed(original, card)
    if groups:
        scan = tcg.scan(card_id)
        printed = ace.printed_ink(scan, original, groups)
        around = np.zeros((H, W), bool)
        for cover in printed.values():
            around |= cover > 0.05
        learned, own_field = {}, None
        for kind, cover in printed.items():
            value = ink_value(img, cover, around)
            learned[kind] = value
            if value is None:
                continue
            area = dilate(cover > 0.05, MARK_GROW + 1)         # the print is found whole (its cut-out's rim too)
            if value[0] < value[1]:                           # cut out: the foil under it, from the field
                if own_field is None:
                    own_field = field(record)[0]
                img = np.where(area, own_field, img)
            else:                                              # lit ink on plainer ground
                img = fill(img, area)
        for kind, value in learned.items():
            if value is not None and kind in drawn:
                cover = np.clip(drawn[kind] * ACE_CUT_GAIN, 0, 1)
                img = img * (1 - cover) + value[0] * cover
    spec = (legacy_shader(record.get("_s") or ""), material_name(record.get("_f") or ""), record.get("_cc") or None)
    extra = {"etch": texture(record, record.get("_e")), "cf": texture(record, record.get("_cf"))}
    out["base"] = (spec, plate_texture(img * (alpha / 255)), extra)
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
