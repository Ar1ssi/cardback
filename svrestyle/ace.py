"""Edit Scarlet & Violet ACE SPEC Trainers in place: the card keeps its own scan, frame and art; only the name, the
effect text and the illustrator can change, and what changed is taken off and set again in white, as these cards
print it.

Their frame is not the Trainers' (trainerframe.py builds those from prints that all share one frame): it is pink,
shaded a little differently from set to set, with the ACE SPEC banners down both sides, so there is no one template to
set text on. The name, the effect text and the illustrator line are white on the frame: the white pixels in their zones
(and a few of the edge round them) are found on the card itself and rebuilt by LaMa, cached per card and set of
edited parts. Ancient and Future prints texture the panel, so a plain "stands out from the panel" test would take
the texture for text. The kind stays as printed: its tab and rule pill belong to the print. The text sits where the
print has it (a Tool's pill can move it up), found by setting the printed text and measuring how far off it falls.
"""
import hashlib

import cv2
import numpy as np

from . import cache, compose, inpaint, layers, tcg, trainer
from . import typeset as ts
from .imaging import dilate, load_rgba, save_rgba

W, H = 733, 1024
# x0, y0, x1, y1. The effect keeps clear of the banners (x < 78 and x > 658): its lines run from x 88 to 646.
NAME_ZONE = (40, 78, 690, 136)
TEXT_ZONE = (80, 548, 656, 884)
ILLUS_ZONE = (42, 936, 232, 959)
TEXT_X, TEXT_WIDTH = 88, 558
GROUPS = {"name": ("name",), "text": ("effect",), "illus": ("illustrator",)}
ZONES = {"name": NAME_ZONE, "text": TEXT_ZONE, "illus": ILLUS_ZONE}
# the least min(R, G, B) of a white glyph's core; the illustrator's thin italics have lighter cores. The edge round
# the core (anti-aliasing) is taken with it, MARK_GROW px.
WHITE_CORE = {"name": 225, "text": 225, "illus": 170}
MARK_GROW = 3
SHIFT_MAX = 30          # px: the text sits no further from where it is set than this
SHIFT_ROW = 3           # white pixels a pixel row needs to hold text
PLAIN_WINDOW, PLAIN_STD = 31, 5.0    # a hole whose 31 px surroundings vary less than this is plain panel
CLEAN_VERSION = 2       # bump when cleaned() changes what it makes, so cached cleanups are made again


def changed(original, card):
    """Which of GROUPS the edits change."""
    return tuple(g for g, keys in GROUPS.items() if any(original.get(k) != card.get(k) for k in keys))


def _white(scan, group):
    """bool HxW: the white print of a group in its zone (cores, before the edge is taken)."""
    x0, y0, x1, y1 = ZONES[group]
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = np.clip(scan[y0:y1, x0:x1, :3], 0, 255).min(axis=2) >= WHITE_CORE[group]
    return out


def taken_off(scan, groups):
    """bool HxW: the print of the edited groups, grown by its soft edge."""
    mask = np.zeros((H, W), bool)
    for g in groups:
        mask |= _white(scan, g)
    return dilate(mask, MARK_GROW)


def _rebuilt(scan, mask):
    """The scan with the mask taken off: plain panel is filled flat (LaMa leaves a faint ghost of big holes in it),
    anything busier (the name bar's shading, the footer's brushing, Ancient and Future textures) by LaMa."""
    rgb = np.clip(scan[..., :3], 0, 255)
    keep = (~mask).astype(np.float32)
    box = (PLAIN_WINDOW, PLAIN_WINDOW)
    weight = np.maximum(cv2.boxFilter(keep, -1, box), 1e-3)
    spread = np.zeros(mask.shape, np.float32)
    for c in range(3):
        mean = cv2.boxFilter(rgb[..., c] * keep, -1, box) / weight
        spread = np.maximum(spread, np.sqrt(np.maximum(cv2.boxFilter(rgb[..., c] ** 2 * keep, -1, box) / weight
                                                         - mean ** 2, 0)))
    plain = mask & (spread < PLAIN_STD)
    art = inpaint.inpaint(scan, mask)
    if plain.any():
        flat = cv2.inpaint(rgb.astype(np.uint8), plain.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        art[plain, :3] = flat[plain]
    return art


def cleaned(card_id, groups):
    """The scan with the edited groups taken off (cached). -> (RGBA float32, mask taken off)"""
    scan = tcg.scan(card_id)
    mask = taken_off(scan, groups)
    key = hashlib.sha256(mask.tobytes()).hexdigest()[:12]
    out = cache.path("art", f"ace-v{CLEAN_VERSION}-{card_id}-{'-'.join(groups)}-{key}.png")
    if out.exists():
        return load_rgba(out), mask
    art = _rebuilt(scan, mask) if mask.any() else scan.copy()
    save_rgba(art, out)
    return art, mask


def _illustrator(canvas, card):
    style = compose.FOOTER["illus"]
    dx, dy = trainer.FOOTER_SHIFT["illus"]
    with layers.kind("footer"):
        ts.stamp(canvas, ts.coverage([dict(text=f"Illus. {card['illustrator']}", face=ts.BOLD_ITALIC,
                                           size=style["size"], x=style["x"] + dx, base=style["base"] + dy,
                                           xscale=style["xscale"])]), compose.WHITE)


def _rows(white):
    """(first, last) pixel row that holds text, or None."""
    rows = np.flatnonzero(white.sum(axis=1) >= SHIFT_ROW)
    return (int(rows[0]), int(rows[-1])) if len(rows) else None


def text_shift(scan, original):
    """How far down the print sets its effect text from where it is set here: the printed text's first and last row
    against the same text set here; 0 when the two do not move alike (the lines break differently)."""
    printed = _rows(_white(scan, "text"))
    with layers.recording() as drawn:
        trainer.body(np.zeros((H, W, 4), np.float32), original, TEXT_X, TEXT_WIDTH, compose.WHITE)
    here = _rows(drawn["text"] > 0.5)
    if not printed or not here:
        return 0
    top, bottom = printed[0] - here[0], printed[1] - here[1]
    return round((top + bottom) / 2) if abs(top - bottom) <= 2 and abs(top) <= SHIFT_MAX else 0


def _text(canvas, card, dy=0):
    return trainer.body(canvas, card, TEXT_X, TEXT_WIDTH, compose.WHITE, dy)


def printed_ink(scan, original, groups):
    """What the card prints for the edited groups, by the kind foil.build_ace judges ("name", "text", "footer"): the
    white print itself, as found on the card (where the composers would set it is not exactly where it is)."""
    kinds = {"name": "name", "text": "text", "illus": "footer"}
    return {kinds[g]: _white(scan, g).astype(np.float32) for g in groups}


def compose_card(card_id, card, original):
    """-> (RGBA float32 733x1024, report dict). card: trainer.read_card's dict, edited; original: as printed."""
    groups = changed(original, card)
    if not groups:
        return tcg.scan(card_id).copy(), {"card": card, "edited": [], "taken_off": None}
    trainer.warn_missing_glyphs(card)
    canvas, mask = cleaned(card_id, groups)
    canvas = canvas.copy()
    scale = None
    if "name" in groups:
        trainer.name_and_character(canvas, card, compose.WHITE)
    if "text" in groups:
        scale = _text(canvas, card, text_shift(tcg.scan(card_id), original))
    if "illus" in groups:
        _illustrator(canvas, card)
    return canvas, {"card": card, "edited": list(groups), "taken_off": mask, "body_scale": scale}
