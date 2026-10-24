"""Scarlet & Violet / Mega Evolution Trainer frame assets, built once from real prints and cached.

template(kind)   that kind's (Item, Supporter, Stadium, Tool) card with the name, effect text and footer taken off:
                 RGBA, alpha = how much of the frame shows (0 in the art window, the print's own soft rim at its
                 edge); the rule pill, the kind's tab label and the TRAINER tab are kept

Every Trainer of a kind from the sets in POOL prints the same frame, pixel for pixel in the TCG Live extract (the
name bar, panel and footer silver are shared by all four kinds; only the tab label, the side stripes and the rule
pill differ). So each pixel is the median of the prints that have no ink there; what every print covers (the
footer's "Illus.", boxes and copyright line) is filled from the silver around it.

Positions are at 733x1024, measured on these prints.
"""
import warnings

import numpy as np

from . import cache, meframe, tcg
from .imaging import dilate, hsv, load_rgba, save_rgba

W, H = 733, 1024
ASSET_VERSION = "2"
KINDS = ("Item", "Supporter", "Stadium", "Tool")

# Uncommon / Rare Trainers of Scarlet & Violet, Paldea Evolved, Paradox Rift, Paldean Fates, Shrouded Fable, Stellar
# Crown, Prismatic Evolutions and Mega Evolution: their frames are identical. The other sets are left out (sv03,
# sv03.5, sv05, sv06 and sv08 print the same design a few levels lighter or darker).
POOL = {
    "Item": ["me01-115", "me01-116", "me01-118", "me01-121", "me01-124", "me01-125", "me01-126", "me01-128",
             "me01-130", "me01-131", "sv01-168", "sv01-170", "sv01-171", "sv01-172", "sv01-173", "sv01-181",
             "sv01-182", "sv01-184", "sv01-185", "sv01-186", "sv01-187", "sv01-188", "sv01-191", "sv01-194",
             "sv01-196", "sv02-178", "sv02-181", "sv02-183", "sv02-188", "sv02-189", "sv04-160", "sv04-175",
             "sv04.5-079", "sv04.5-083", "sv04.5-084", "sv04.5-089", "sv04.5-091", "sv06.5-061", "sv07-129",
             "sv07-130", "sv07-135", "sv08.5-101", "sv08.5-102", "sv08.5-110", "sv08.5-118", "sv08.5-127"],
    "Supporter": ["me01-113", "me01-114", "me01-119", "me01-120", "me01-123", "me01-132", "sv01-166", "sv01-175",
                  "sv01-176", "sv01-177", "sv01-179", "sv01-180", "sv01-183", "sv01-189", "sv01-190", "sv01-195",
                  "sv01-198", "sv02-172", "sv02-177", "sv02-179", "sv02-180", "sv02-182", "sv02-184", "sv02-185",
                  "sv02-187", "sv04-165", "sv04-167", "sv04-168", "sv04-169", "sv04-172", "sv04-173", "sv04-174",
                  "sv04-181", "sv04.5-077", "sv04.5-078", "sv04.5-080", "sv04.5-082", "sv04.5-085", "sv04.5-086",
                  "sv04.5-087", "sv04.5-088", "sv06.5-056", "sv06.5-057", "sv06.5-059", "sv06.5-064", "sv07-132",
                  "sv07-133", "sv07-138", "sv07-139", "sv08.5-093", "sv08.5-096", "sv08.5-097", "sv08.5-098",
                  "sv08.5-099", "sv08.5-100", "sv08.5-103", "sv08.5-105", "sv08.5-109", "sv08.5-112", "sv08.5-113",
                  "sv08.5-114", "sv08.5-115", "sv08.5-122", "sv08.5-123", "sv08.5-124", "sv08.5-125"],
    "Stadium": ["me01-117", "me01-122", "me01-127", "me01-129", "sv01-167", "sv01-178", "sv02-171", "sv02-174",
                "sv02-175", "sv02-186", "sv04.5-076", "sv04.5-081", "sv06.5-054", "sv07-131", "sv08.5-094",
                "sv08.5-108"],
    "Tool": ["sv01-169", "sv01-174", "sv01-192", "sv01-193", "sv01-197", "sv02-173", "sv02-176", "sv04-161",
             "sv04-162", "sv04-166", "sv04-176", "sv04-177", "sv04-178", "sv04-179", "sv04.5-090", "sv06.5-055",
             "sv06.5-063", "sv07-137", "sv07-140", "sv07-141", "sv08.5-095", "sv08.5-111", "sv08.5-126"],
}

ART = (58, 144, 675, 529)                 # x0, y0, x1, y1 of the art window; its rim blends over the outer 1-4 px
NAME_ZONE = (40, 76, 698, 137)            # name, and the Supporter's character at the right
PANEL = (44, 540, 690, 886)               # effect text; the rule pill starts below it
# illustrator, regulation mark, set code, number, rarity (the type's dashes run down x 27-40). It stops short of the
# rule pill's round left end, whose rim starts at x 232 (the rarity ends by 221): every print has the rim, so inside
# the zone it would be taken for ink and painted out with silver.
FOOTER = (42, 928, 228, 993)
COPYRIGHT = (100, 998, 633, 1023)
ZONES = (NAME_ZONE, PANEL, FOOTER, COPYRIGHT)
MIN_CLEAN = 3              # prints that must show a pixel's silver for their median to stand
# The footer prints the same boxes and words on every card, so a few prints where they were missed would stand
# in for the silver: there a pixel needs this share of the prints.
FOOTER_SHARE = 0.3
INK_THRESHOLD, INK_GROW = 24, 3
RIM = 5                    # px inside the art window's edge where the rim's blend is measured
RIM_PROBE = 6              # px inside the window: the art a rim pixel is blended with
FILL_SMOOTH = 9            # px: clean silver either side of a hole, averaged along its row
# px beyond a zone's left / right edge whose silver a hole's row may also be bridged from (the footer's right only
# up to the rule pill's rim); coloured pixels (the type's dashes) are not silver.
FILL_REACH = {FOOTER: (12, 3), COPYRIGHT: (20, 20)}
SILVER_SAT = 0.2
ROW_FIT_ROUNDS = 4


def _asset(name):
    return cache.path("assets", f"trainer-v{ASSET_VERSION}", name)


def _row_ink(rgb, box):
    """Ink against each row's own shading: the footer is brushed along x and shaded smoothly across it (its panel's
    edges run along x too, which a square window would take for ink). Each row's luminance is fitted with a cubic
    in x, the pixels far from it dropped and the fit repeated."""
    x0, y0, x1, y1 = box
    lum = (rgb[y0:y1, x0:x1] * [0.299, 0.587, 0.114]).sum(-1)
    xs = np.linspace(-1, 1, x1 - x0)
    mask = np.zeros((H, W), bool)
    for r, row in enumerate(lum):
        keep = np.ones_like(row, bool)
        for _ in range(ROW_FIT_ROUNDS):
            fit = np.polyval(np.polyfit(xs[keep], row[keep], 3), xs)
            off = np.abs(row - fit) > INK_THRESHOLD
            if keep.sum() - off.sum() < 8:
                break
            keep = ~off
        mask[y0 + r, x0:x1] = np.abs(row - fit) > INK_THRESHOLD
    return mask


def _ink(rgb):
    """What a print has set on its frame (words, numbers, boxes), grown a little."""
    mask = meframe.ink(rgb, NAME_ZONE, threshold=INK_THRESHOLD) | meframe.ink(rgb, PANEL, threshold=INK_THRESHOLD)
    mask |= _row_ink(rgb, FOOTER) | _row_ink(rgb, COPYRIGHT)
    return dilate(mask, INK_GROW)


def _zones():
    out = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in ZONES:
        out[y0:y1, x0:x1] = True
    return out


def _load(card_id):
    """(rgb as uint8, ink mask) of a print."""
    rgb = tcg.scan(card_id)[..., :3]
    return np.clip(rgb, 0, 255).astype(np.uint8), _ink(rgb)


def _fill(img, holes):
    """Holes (cleared from `holes` as they are filled) take their row's clean silver: the frame is brushed along x,
    so each row keeps its own streak, and a hole is bridged between the clean pixels either side (averaged over
    FILL_SMOOTH px). Rows with nothing clean on them are bridged down each column instead."""
    out = img.copy()
    k = np.ones(FILL_SMOOTH, np.float32)
    _, sat, _ = hsv(img)
    for box in ZONES:
        x0, y0, x1, y1 = box
        left, right = FILL_REACH.get(box, (0, 0))
        s0, s1 = max(0, x0 - left), min(W, x1 + right)
        xs = np.arange(s0, s1)
        for y in range(y0, y1):
            gap = np.zeros(s1 - s0, bool)
            gap[x0 - s0:x1 - s0] = holes[y, x0:x1]
            if not gap.any():
                continue
            ok = ~gap & (sat[y, s0:s1] < SILVER_SAT)
            if ok.sum() < 2:
                continue
            weight = np.convolve(ok.astype(np.float32), k, "same")
            for c in range(3):
                smooth = np.convolve(img[y, s0:s1, c] * ok, k, "same") / np.maximum(weight, 1e-3)
                out[y, s0:s1, c][gap] = np.interp(xs[gap], xs[ok], smooth[ok])
            holes[y, x0:x1] = False
        ys = np.arange(y0, y1)
        for x in range(x0, x1):
            gap = holes[y0:y1, x]
            if not gap.any() or gap.all():
                continue
            for c in range(3):
                out[y0:y1, x, c][gap] = np.interp(ys[gap], ys[~gap], out[y0:y1, x, c][~gap])
    return out


def _blend(stack, y, x, py, px):
    """Share of the art (at py, px) in the print at (y, x): the slope of one over the other across the prints."""
    art, here = stack[:, py, px], stack[:, y, x]
    spread = float(art.var(0).sum())
    if spread <= 3:
        return 0.0
    return min(max(float(((art - art.mean(0)) * (here - here.mean(0))).mean(0).sum() / spread), 0.0), 1.0)


def _rim(prints):
    """Frame alpha and colour along the inside of the art window's edge. A rim pixel prints a * frame + (1 - a) *
    art, with the art the window's RIM_PROBE px further in: a is fitted over the prints (their art differs, the
    frame does not), as one profile per side (the median along it), and the frame colour per pixel from it.
    Outside the window the frame is solid."""
    x0, y0, x1, y1 = ART
    stack = np.stack([rgb for rgb, _ in prints]).astype(np.float32)
    along = {"top": range(x0 + RIM, x1 - RIM), "bottom": range(x0 + RIM, x1 - RIM),
             "left": range(y0 + RIM, y1 - RIM), "right": range(y0 + RIM, y1 - RIM)}
    at = {"top": lambda d, t: (y0 + d, t), "bottom": lambda d, t: (y1 - 1 - d, t),
          "left": lambda d, t: (t, x0 + d), "right": lambda d, t: (t, x1 - 1 - d)}

    def share(side, d):
        return np.median([_blend(stack, *at[side](d, t), *at[side](d + RIM_PROBE, t)) for t in along[side]])
    profile = {}
    for side in at:
        # art RIM_PROBE px apart is not quite the same art: the slope deep inside, where the print is all art,
        # is what a share of 1 measures
        full = share(side, 2 * RIM)
        for d in range(RIM):
            profile[side, d] = min(1.0, share(side, d) / max(full, 1e-3))
    alpha = np.ones((H, W), np.float32)
    alpha[y0:y1, x0:x1] = 0
    colour = np.zeros((H, W, 3), np.float32)
    for y in range(y0, y1):
        for x in range(x0, x1):
            depths = {"top": y - y0, "bottom": y1 - 1 - y, "left": x - x0, "right": x1 - 1 - x}
            art_share = np.prod([profile[side, d] if d < RIM else 1.0 for side, d in depths.items()])
            a = 1 - art_share
            if a < 0.03:
                continue
            px = min(max(x, x0 + RIM_PROBE + RIM), x1 - 1 - RIM_PROBE - RIM)
            py = min(max(y, y0 + RIM_PROBE + RIM), y1 - 1 - RIM_PROBE - RIM)
            alpha[y, x] = a
            colour[y, x] = (stack[:, y, x] - art_share * stack[:, py, px]).mean(0) / a
    return alpha, colour


def template(kind):
    """RGBA float32 733x1024: the kind's frame with nothing printed on it; alpha 0 in the art window."""
    if kind not in POOL:
        raise ValueError(f"no Trainer frame for {kind!r} (have: {', '.join(POOL)})")
    out = _asset(f"template-{kind}.png")
    if out.exists():
        return load_rgba(out)
    prints = [_load(i) for i in POOL[kind]]
    zones = _zones()
    clean = np.zeros((H, W, 3), np.float32)
    for x0, y0, x1, y1 in ZONES:
        crop = np.stack([rgb[y0:y1, x0:x1] for rgb, _ in prints]).astype(np.float32)
        crop[np.stack([ink[y0:y1, x0:x1] for _, ink in prints])] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # pixels no print shows: NaN, taken below
            clean[y0:y1, x0:x1] = np.nanmedian(crop, axis=0)
    shown = (~np.stack([ink for _, ink in prints])).sum(0)
    for x0, y0, x1, y1 in (FOOTER, COPYRIGHT):
        few = shown[y0:y1, x0:x1] < FOOTER_SHARE * len(prints)
        shown[y0:y1, x0:x1][few] = 0
    # The silver under the name, text and footer is the same for every kind: where few prints of this kind show
    # it, the other kinds' prints are asked too.
    thin = zones & (shown < MIN_CLEAN)
    for x0, y0, x1, y1 in (FOOTER, COPYRIGHT):
        thin[y0:y1, x0:x1] = False                     # the same boxes and words on every kind
    if thin.any():
        values = [[] for _ in range(int(thin.sum()))]
        for k, ids in POOL.items():
            if k == kind:
                continue
            for card_id in ids:
                rgb, ink = _load(card_id)
                ok = ~ink[thin]
                for j in np.nonzero(ok)[0]:
                    values[j].append(rgb[thin][j])
        ys, xs = np.nonzero(thin)
        for j, (y, x) in enumerate(zip(ys, xs)):
            own = [p[0][y, x] for p in prints if not p[1][y, x]]
            if len(values[j]) + len(own) >= MIN_CLEAN:
                clean[y, x] = np.median(np.array(values[j] + own, np.float32), axis=0)
                shown[y, x] = MIN_CLEAN
    canvas = np.zeros((H, W, 4), np.float32)
    for y in range(0, H, 64):                        # the frame itself: identical on every print, art bleed aside
        canvas[y:y + 64, :, :3] = np.median(np.stack([rgb[y:y + 64] for rgb, _ in prints]), axis=0)
    canvas[zones, :3] = clean[zones]
    holes = zones & (shown < MIN_CLEAN)
    canvas[..., :3] = _fill(np.nan_to_num(canvas[..., :3]), holes)
    alpha, colour = _rim(prints)
    edge = (alpha > 0) & (alpha < 1)
    canvas[edge, :3] = colour[edge]
    canvas[..., 3] = alpha * 255
    save_rgba(canvas, out)
    return canvas


def corners():
    """The printed card's alpha (rounded corners), from the first Item print."""
    return tcg.scan(POOL["Item"][0])[..., 3]


def build_all():
    for kind in KINDS:
        template(kind)
