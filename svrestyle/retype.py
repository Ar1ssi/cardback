"""Change the type of a regular Scarlet & Violet / Mega Evolution Pokemon on its own print.

Every regular card of a type prints the same frame (meframe.template), and the frame is all that differs between two
types: name bar, panel, weakness row and footer take the type's colours and textures, the name-bar icon its symbol,
and Darkness prints its words white where the others print them dark. So the card's own print is lifted off the old
type's frame and set on the new one's:

- What the card prints that its type's frame does not (words, numbers, icons, boxes, the Stage pieces, a Trainer's
  medallion) is found against the card's own background, read by blending the print's surroundings into it. Print
  both frames have (the row's labels and rules, the name-bar icon) is the frame's, and the new frame brings its own.
- Words are ink of one colour: each pixel's share of ink is read off the line from the background to the ink, and the
  word is set again at that share over the new frame, in the new type's ink. Everything else printed (energy icons,
  the Ability badge and red Ability names, the footer's boxes, the Stage pieces, the medallion) keeps its colours;
  at its rim the pixel's share of it is its contrast over the background against the object's own. Red Ability names
  are set as the Mega Evolution layout sets them: in a white keyline on a dark panel, none on a light one.
- The old frame's texture where nothing is printed is the new frame's, pixel for pixel; the panel's mottling differs
  from card to card, and the template's comes from real prints of the new type.
- Silver parts (border, Pokedex strip, the row's rules, the Stage tab) are the same on every type and stay as printed.
"""
import functools

import cv2
import numpy as np

from . import compose, frame, mecompose, meframe, tcg
from .imaging import dilate, fill_holes, grow, luminance

W, H = 733, 1024
# Where the card prints on its frame: x0, y0, x1, y1 (name bar right of the tab, Pokedex strip, panel, footer above
# the panel's rim, copyright line); the weakness row's values are read along the row (meframe.row_ink).
PRINT_BOXES = [(136, 18, 712, 100), meframe.DEX_STRIP, meframe.PANEL, (36, 914, 697, 989), meframe.COPYRIGHT]
DARK_PANEL = 110     # panel luminance under which a type prints its words white (mecompose's rule)
PRINT_DIFF = 60      # how far print stands out from its local background; the panel's mottling stays under it
FRAME_GROW = 2       # px: print the old frame has within this reach is the frame's
GROW = 2             # px: anti-aliasing round each printed shape
OFF_INK = 0.35       # distance from the background-to-ink line, over that line's length, past which a pixel is not ink
INK_SHARE = 0.3      # least ink share a pixel needs to count as ink
INK_CORE = 0.6       # a word's strongest pixel has at least this ink share (mottling and grain stay under it)
OBJECT_SHARE = 0.35  # a shape this much off the ink line is an object (icon, badge, red Ability name, box)
ALIEN_SHARE = 0.2    # ... and this much of it not in the old frame's colours nearby (the bar's wave is)
FRAME_RANGE = 31     # px: window the old frame's colours are read in
FRAME_SLACK = 14     # RGB levels past those colours a pixel still counts as frame
OBJECT_MIN = 20      # px: smaller off-ink shapes are grain
OBJECT_PEAK = 70     # an object stands out at least this far somewhere
RIM_SCALE = 0.7      # an object's own contrast: this much of its strongest
RIM_MIN = 40         # ... and never less
RIM_FLOOR = 25       # contrast the panel's own grain reaches against the read ground: none of it is print
COST_COLUMN = (30, 512, 250, 870)   # x0, y0, x1, y1: where the panel's cost icons sit
ICON_MATCH = 0.75    # least match of a cost icon (normalized correlation inside its disc)
ICON_APART = 30      # px: two cost icons' centres are at least this far apart
# Objects (Ability badge, red Ability names) print only in the panel; elsewhere the name bar holds words and the
# frame's own pieces, the weakness row icons (found by their data) and the footer boxes (set where they print).
OBJECT_ZONE = meframe.PANEL
RED_MIN, RED_GB = 150, 70   # the print's red: R above, G (and B, a little more) under
BADGE_RIM = 5        # px: the badge's silver rim round its red
BADGE_W, BADGE_FILL = 100, 0.6   # px wide at least, and this much of its box red: the Ability badge (names are 0.35-0.47)
KEYLINE_REACH = 2    # px: a red name's white keyline on a dark panel, taken off on a light one
ICON_CLEAR = 3       # px past the name-bar icon's rim that are the icon's
MARK_FROM = (596, 380)       # x0, y0: a Trainer's medallion (and its figure) is found right of and below this
MARK_CORE = (662, 500, 30)   # centre x, y and radius inside the medallion on every such card (sv09, sv10)
MARK_DIFF = 40               # how far the medallion stands out from the old frame
PIECE_REACH = 2      # px: the card's own Stage pieces may reach this far past the Mega Evolution ones
DISC_FIT = 0.85      # overlap (intersection over union) of a disc's shape, holes filled, with its circle
# Energy icons are discs about as tall as wide, this wide (px), where cards print them: x0, y0, x1, y1 their centre
# lies in. Cost icons sit at the left of the panel; row icons in the weakness row. (A letter O is round too.)
DISC_ZONES = [((30, 46), (36, 512, 240, 870)), ((22, 32), (100, 876, 700, 914))]
DISC_MIN = 150       # px: smallest disc shape
TYPED = 14           # RGB distance between the two frames past which a pixel is the type's (else silver)
BOX_SLACK = 30       # a footer-box pixel at least this far from the old frame is the box's, not its rim's ground
SILVER = [(90, 484, 660, 510), meframe.COPYRIGHT]   # the Pokedex strip and the copyright line: as printed


@functools.lru_cache(maxsize=None)
def _dark_panel(type_name):
    x0, y0, x1, y1 = meframe.PANEL
    return float(np.median(luminance(meframe.template(type_name)[y0:y1, x0:x1]))) < DARK_PANEL


def ink(type_name):
    """The type's word ink, as mecompose sets it: white on a dark panel (Darkness), dark elsewhere."""
    return compose.WHITE.copy() if _dark_panel(type_name) else np.array(frame.DARK_INK, np.float32)


def frame_print(rgb):
    """What stands out from its local background where cards print. -> bool HxW"""
    mask = np.zeros((H, W), bool)
    for box in PRINT_BOXES:
        mask |= meframe.ink(rgb, box, threshold=PRINT_DIFF)
    for box in meframe.ROW_VALUES:
        mask |= meframe.row_ink(rgb, box)
    return mask


def _shapes(own):
    n, labels = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)
    return n, labels, np.maximum(np.bincount(labels.ravel(), minlength=n), 1)


def _per_shape(labels, n, values, how=np.add):
    out = np.zeros(n)
    how.at(out, labels.ravel(), values.ravel())
    return out


def _discs(rgb, ground, labels, n, size, peak):
    """Energy icons among the shapes: (coverage HxW float, bool per shape). The rim's share is its contrast."""
    cover, found = np.zeros((H, W), np.float32), np.zeros(n, bool)
    contrast = np.linalg.norm(rgb - ground, axis=-1)
    yy, xx = np.mgrid[:H, :W]
    for i in np.flatnonzero(size >= DISC_MIN):
        if i == 0:
            continue
        ys, xs = np.nonzero(labels == i)
        w, h = np.ptp(xs) + 1, np.ptp(ys) + 1
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        if abs(w - h) > 4 or not any(lo <= w <= hi and x0 <= cx < x1 and y0 <= cy < y1
                                     for (lo, hi), (x0, y0, x1, y1) in DISC_ZONES):
            continue
        shape = np.zeros((H, W), bool)
        shape[ys, xs] = True
        r = np.hypot(xx - cx, yy - cy)
        r0 = max(w, h) / 2
        filled, circle = fill_holes(shape), r <= r0
        if (filled & circle).sum() < DISC_FIT * (filled | circle).sum():
            continue
        found[i] = True
        rim = np.clip(contrast / max(RIM_SCALE * peak[i], RIM_MIN), 0, 1)
        cover = np.maximum(cover, np.where(r < r0 - 1, 1.0, np.where(r < r0 + 2, rim, 0)).astype(np.float32))
    return cover, found


def _disc(cx, cy, r, rim):
    """Coverage of an icon of radius r at (cx, cy): whole inside, its rim's share by contrast."""
    yy, xx = np.mgrid[:H, :W]
    d = np.hypot(xx - cx, yy - cy)
    return np.where(d < r - 1, 1.0, np.where(d < r + 2, rim, 0)).astype(np.float32)


def _printed_icons(rgb, card, rim):
    """Coverage of the energy icons the card prints, found from its data: each cost type matched against its icon in
    the panel's cost column, and the weakness row's icons where the composers set them (a white Colorless icon on
    the Colorless panel hardly stands out)."""
    cover = np.zeros((H, W), np.float32)
    x0, y0, x1, y1 = COST_COLUMN
    grey = lambda a: cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    area = grey(rgb[y0:y1, x0:x1])
    for kind in {t for a in card["attacks"] for t in (a.get("cost") or [])}:
        crop, alpha, scx, scy, (ox, oy) = meframe.icon(kind, "cost")[0]
        r = meframe.icon(kind, "cost")[1]
        score = cv2.matchTemplate(area, grey(crop), cv2.TM_CCOEFF_NORMED, mask=(alpha > 0.5).astype(np.uint8))
        score = np.nan_to_num(score, nan=0, posinf=0, neginf=0)
        while True:
            _, best, _, (bx, by) = cv2.minMaxLoc(score)
            if best < ICON_MATCH:
                break
            cover = np.maximum(cover, _disc(x0 + bx + scx - ox, y0 + by + scy - oy, r, rim))
            score[max(0, by - ICON_APART):by + ICON_APART, max(0, bx - ICON_APART):bx + ICON_APART] = 0
    for side in ("weakness", "resistance"):
        if card.get(side):
            cx, cy, r = mecompose.ROW_ICONS[side]
            cover = np.maximum(cover, _disc(cx, cy, r, rim))
    cx, cy, r, step = mecompose.ROW_ICONS["retreat"]
    for i in range(card["retreat"]):
        cover = np.maximum(cover, _disc(cx + i * step, cy, r, rim))
    return cover


def _print_red(rgb):
    """Pixels in the print's red (the Ability badge, red Ability names) where objects print; not a Fire frame's
    orange-red (its blue stays under its green)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (r > RED_MIN) & (g < RED_GB) & (b < RED_GB + 25) & (b + 8 >= g)
    x0, y0, x1, y1 = OBJECT_ZONE
    zone = np.zeros((H, W), bool)
    zone[y0:y1, x0:x1] = True
    return red & zone


def _badge(red):
    """The Ability badge: its red round its white word, filled, with its silver rim."""
    closed = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    fill = np.array([red[y:y + h, x:x + w].mean() if w * h else 0 for x, y, w, h, _ in stats])
    is_badge = (stats[:, cv2.CC_STAT_WIDTH] >= BADGE_W) & (fill >= BADGE_FILL)
    is_badge[0] = False
    return dilate(fill_holes(is_badge[labels]), BADGE_RIM)


def _red_names(rgb, ground, names, near):
    """The red Ability names' coverage and their red. A name's pixel mixes its red, the white keyline dark panels
    print round it, and the ground: both shares are read at once, so the keyline is not taken for red."""
    if not names.any():
        return np.zeros((H, W), np.float32), compose.ABILITY_RED
    core = cv2.erode(names.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ink = np.median(rgb[core if core.any() else names], axis=0)
    lift = rgb[near] - ground[near]
    basis = np.stack([ink - ground[near], compose.WHITE - ground[near]], axis=-1)        # N x 3 x 2
    cover = np.zeros((H, W), np.float32)
    cover[near] = np.clip(_unmix2(basis, lift)[:, 0], 0, 1)
    return cover, ink


def _unmix2(basis, lift):
    """Least-squares shares of two colours per pixel: basis N x 3 x 2, lift N x 3 -> N x 2."""
    gram = np.einsum("nci,ncj->nij", basis, basis) + np.eye(2) * 1e-3
    rhs = np.einsum("nci,nc->ni", basis, lift)
    return np.linalg.solve(gram, rhs[..., None])[..., 0]


def _trainer_mark(card_id, rgb, old):
    """A Trainer's Pokemon (Iono's Voltorb, Team Rocket's Porygon2) prints its Trainer's medallion over the art
    window's bottom right corner, the figure reaching out over the frame: what stands out from the old frame there,
    joined to the medallion's middle, filled. -> bool HxW"""
    if "'s " not in tcg.card(card_id).get("name", ""):
        return np.zeros((H, W), bool)
    x0, y0 = MARK_FROM
    differs = np.zeros((H, W), np.uint8)
    differs[y0:, x0:] = np.linalg.norm(rgb - old, axis=-1)[y0:, x0:] > MARK_DIFF
    differs = cv2.morphologyEx(differs, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, labels = cv2.connectedComponents(differs, connectivity=8)
    cx, cy, r = MARK_CORE
    yy, xx = np.mgrid[:H, :W]
    hit = np.unique(labels[(np.hypot(xx - cx, yy - cy) < r) & (labels > 0)])
    return fill_holes(np.isin(labels, hit) & (labels > 0))


def _footer_boxes(card_id, rgb, old):
    """Coverage of the regulation-mark and set boxes the card prints (as the composers set them), each grown a pixel
    where the print stands out from the old frame."""
    c = tcg.card(card_id)
    s = tcg.set_info(c["set"]["id"])
    boxes = compose.footer_boxes(c.get("regulationMark") or "", mecompose.set_code(c["set"]["id"], s))
    grown = cv2.dilate(boxes, np.ones((3, 3), np.uint8))
    off = np.clip(np.linalg.norm(rgb - old, axis=-1) / BOX_SLACK, 0, 1)
    return np.maximum(boxes, grown * off)


def recolour(rgba, card, new_type):
    """The card (RGBA float32 733x1024) on the new type's frame. card: modern.read_card's dict of it as printed
    (its type, Stage, costs and weakness row)."""
    card_id, old_type, stage_shown = card["id"], card["type"], card["stage"]
    rgb = rgba[..., :3]
    old, new = meframe.template(old_type)[..., :3], meframe.template(new_type)[..., :3]
    ink_old, ink_new = ink(old_type), ink(new_type)
    # the frames' own print (row labels and rules, Stage tab, name-bar icon) is where both frames print it; a blemish
    # in one template's fill is not
    frames = frame_print(old) & dilate(frame_print(new), FRAME_GROW)
    cx, cy, r = meframe.ICON
    yy, xx = np.mgrid[:H, :W]
    frames |= np.hypot(xx - cx, yy - cy) < r + ICON_CLEAR
    own = frame_print(rgb) & ~dilate(frames, FRAME_GROW)
    pieces = (meframe.stage_sprite(stage_shown)[..., 3] / 255 if stage_shown in (1, 2)
              else np.zeros((H, W), np.float32))
    mark = _trainer_mark(card_id, rgb, old)
    red = _print_red(rgb)
    badge = _badge(red)
    names = red & ~badge & ~dilate(mark, GROW)
    hole = (dilate(own, GROW) | dilate(pieces > 0.02, PIECE_REACH) | dilate(badge, GROW) | dilate(names, KEYLINE_REACH)
            | dilate(mark, GROW))
    filled = cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), hole.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    ground = np.where(hole[..., None], filled.astype(np.float32), rgb)

    # each pixel against the line from its background to the ink
    to_ink, lift = ink_old - ground, rgb - ground
    length2 = np.maximum((to_ink * to_ink).sum(-1), 1.0)
    share = np.clip((lift * to_ink).sum(-1) / length2, 0, 1)
    off_line = np.linalg.norm(lift - share[..., None] * to_ink, axis=-1) / np.sqrt(length2)
    contrast = np.linalg.norm(lift, axis=-1)
    inky = (share > INK_SHARE) & (off_line < OFF_INK)
    k = np.ones((FRAME_RANGE, FRAME_RANGE), np.uint8)
    frame_like = ((rgb >= cv2.erode(old, k) - FRAME_SLACK) & (rgb <= cv2.dilate(old, k) + FRAME_SLACK)).all(-1)

    n, labels, size = _shapes(own)
    labels = np.where(own, labels, 0)
    off = _per_shape(labels, n, own & (off_line >= OFF_INK)) / size
    alien = _per_shape(labels, n, own & ~frame_like) / size
    peak = _per_shape(labels, n, contrast, np.maximum)
    core = _per_shape(labels, n, np.where(inky, share, 0), np.maximum)
    ys, xs = np.mgrid[:H, :W]
    cy, cx = _per_shape(labels, n, ys) / size, _per_shape(labels, n, xs) / size
    x0, y0, x1, y1 = OBJECT_ZONE
    objects = ((alien > ALIEN_SHARE) & (off > OBJECT_SHARE) & (size >= OBJECT_MIN) & (peak >= OBJECT_PEAK)
               & (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1))
    words = ~objects & (core >= INK_CORE)
    discs, is_disc = _discs(rgb, ground, labels, n, size, peak)
    objects &= ~is_disc
    words &= ~is_disc
    objects[0] = words[0] = False

    grown = lambda m: dilate(m, GROW) & hole
    object_px = grown(objects[labels])
    word_px = grown(words[labels]) & ~object_px
    near = cv2.dilate(labels.astype(np.float32), np.ones((2 * GROW + 3,) * 2, np.uint8)).astype(int)
    kept = np.where(object_px, np.clip(contrast / np.maximum(RIM_SCALE * peak[near], RIM_MIN), 0, 1), 0)
    contrast_rim = np.clip((contrast - RIM_FLOOR) / RIM_MIN, 0, 1)    # a rim pixel's share of the print over it
    inside = cv2.erode(badge.astype(np.uint8), np.ones((2 * BADGE_RIM + 1,) * 2, np.uint8)) > 0
    kept = np.maximum(kept, np.where(inside, 1.0, np.where(dilate(badge, GROW), contrast_rim, 0)))
    red_names, red_ink = _red_names(rgb, ground, names, dilate(names, KEYLINE_REACH))
    discs = np.maximum(discs, _printed_icons(rgb, card, contrast_rim))
    boxes = _footer_boxes(card_id, rgb, old)
    # Stage pieces as Mega Evolution prints them; a Scarlet & Violet card's sit a pixel or two off at their edge
    piece_rim = (dilate(pieces > 0.5, PIECE_REACH) & hole) * contrast_rim
    mark_in = cv2.erode(mark.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mark_cov = np.where(mark_in, 1.0, np.where(dilate(mark, GROW), contrast_rim, 0))
    kept = np.maximum.reduce([kept, pieces, piece_rim, discs, boxes, mark_cov])
    names = dilate(red_names > 0.05, KEYLINE_REACH)       # the red names and their keylines: set below
    kept[names] = 0
    share = np.where(word_px & (kept <= 0.02) & ~names, share, 0)

    # kept print's rim mixes it with the old ground: that share of the new ground replaces it; elsewhere the new frame
    out = np.where((kept > 0)[..., None], rgb + (1 - kept)[..., None] * (new - ground), new)
    words_set = share[..., None] * ink_new + (1 - share[..., None]) * new
    out = np.where((share > 0)[..., None], words_set, out)
    # red Ability names over the new frame: a white keyline on a dark panel, none on a light one (as mecompose)
    if _dark_panel(new_type):
        line = np.clip(grow(red_names, 1.0) - red_names, 0, 1)[..., None]
        out = out * (1 - line) + compose.WHITE * line
    out = out * (1 - red_names[..., None]) + red_ink * red_names[..., None]

    typed = cv2.morphologyEx((np.linalg.norm(new - old, axis=-1) > TYPED).astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8)) > 0
    typed = dilate(typed, 1)
    for x0, y0, x1, y1 in SILVER:
        typed[y0:y1, x0:x1] = False
    typed = cv2.GaussianBlur(typed.astype(np.float32), (0, 0), 0.8)[..., None]
    result = rgba.copy()
    result[..., :3] = typed * out + (1 - typed) * rgb
    x0, y0, x1, y1 = meframe.ART
    result[y0:y1, x0:x1] = rgba[y0:y1, x0:x1]
    return result
