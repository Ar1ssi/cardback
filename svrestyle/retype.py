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
- The templates are Mega Evolution prints; a Scarlet & Violet print sits up to a pixel off them, and its Stage pieces
  a few pixels off Mega Evolution's. The print is moved onto its template first and back after, and each Stage piece
  is matched on the print where it lies, so no edge between the type's colour and the silver doubles.
"""
import functools

import cv2
import numpy as np

from . import compose, frame, layers, mecompose, meframe, tcg
from .imaging import dilate, drop_small, fill_holes, grow, hsv, luminance

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
PIECE_SHIFT = 6      # px: farthest a print's Stage piece sits from Mega Evolution's (Scarlet & Violet swooshes: 4)
PIECE_CLIP = 60      # RGB levels: a piece pixel differing more is the card's own (portrait, "Evolves from" line)
PIECE_GAP = 15       # px: breaks in Mega Evolution's swoosh closed over to find the print's own silver round it
SILVER_SAT, SILVER_RAMP = 0.16, 0.14   # saturation under which a pixel is all silver (meframe's), over which none
# The Stage pieces matched apart, x0, y0, x1, y1: the bar's swoosh (above the "Evolves from" strip it runs into),
# and the tab, portrait ring and strip; a piece pixel belongs to the first box it lies in.
PIECE_BOXES = [(440, 0, 600, 92), (0, 0, W, H)]
GROUND_SHADE = 6     # px (sigma): the print's shading against its template is read this close round its print
GROUND_OPEN = 0.25   # share of open pixels under which the shading is read wider
GROUND_OFF, GROUND_OFF_RAMP = 24, 10   # mean RGB residual round the print past which the template is not its ground
REGISTER_REACH = 3  # px: farthest a print sits off its type's template
# Where only the frame prints edges on every card, x0, y0, x1, y1: the side borders under the Stage pieces, the
# weakness row's rules (the labels are the frame's, the values sit between them).
REGISTER_BOXES = [(0, 250, 60, 1000), (676, 110, W, 1000), (30, 866, 703, 882), (30, 906, 703, 926)]
DISC_FIT = 0.85      # overlap (intersection over union) of a disc's shape, holes filled, with its circle
# Energy icons are discs about as tall as wide, this wide (px), where cards print them: x0, y0, x1, y1 their centre
# lies in. Cost icons sit at the left of the panel; row icons in the weakness row. (A letter O is round too.)
DISC_ZONES = [((30, 46), (36, 512, 240, 870)), ((22, 32), (100, 876, 700, 914))]
DISC_MIN = 150       # px: smallest disc shape
TYPED = 14           # RGB distance between the two frames past which a pixel is the type's (else silver)
BOX_REACH = 2        # px: a print's footer box may reach this far past where the composers set it
BOX_RING = 4         # px: the band round that whose median is the colour printed round the boxes
BOX_SPECK = 12       # px: smaller shapes off the colour round the boxes are grain
BOX_DIFF = 90        # RGB levels off that colour a box pixel is (Water's grain stays under it)
BOX_DARK = (38, 33, 38)   # the composers' box outline (compose._footer)
BOX_LIGHT = 128      # a box pixel this light is its white ring or fill
BOX_MIN_SHARE = 0.1  # least box share its colour is read back from (less is ground and noise)
BOX_OFF = 40         # RGB distance off the ground-to-outline line past which an edge pixel is all box
ICON_REACH = 3       # px: farthest a print's icon sits from where it is looked for
RIM_SIDE = 0.75      # px either side of an icon's rim where its inside and outside are read
ICON_GROW, ICON_STEP = 1.5, 0.25   # px: how far a print's icon radius may differ from the one looked for; fit step
SOFT_EDGE = 0.6      # px (sigma): the anti-aliasing of a medallion's edge
SILVER = [(90, 484, 660, 510), meframe.COPYRIGHT]   # the Pokedex strip and the copyright line: as printed
WORDS = "retyped_words"   # layers kind: the words set again in the new type's ink (foil.build_modern's plates)


@functools.lru_cache(maxsize=None)
def _dark_panel(type_name):
    x0, y0, x1, y1 = meframe.PANEL
    return float(np.median(luminance(meframe.template(type_name)[y0:y1, x0:x1]))) < DARK_PANEL


def white_ink(type_name):
    """True when the type prints its words white (Darkness)."""
    return _dark_panel(type_name)


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


def _discs(labels, n, size):
    """Energy icons among the shapes: ([(cx, cy, r) per icon], bool per shape)."""
    circles, found = [], np.zeros(n, bool)
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
        circles.append((cx, cy, r0))
    return circles, found


def _disc(cx, cy, r):
    """Coverage of an icon whose rim runs at radius r round (cx, cy), anti-aliased over a pixel. An icon's rim is
    a circle, so its share of a rim pixel is geometry: a share read off colours mistakes the ground's grain for it,
    and no card shows the ground under the icon to compare with."""
    yy, xx = np.mgrid[:H, :W]
    return np.clip(r + 0.5 - np.hypot(xx - cx, yy - cy), 0, 1).astype(np.float32)


def _on_rim(rgb, cx, cy, r, reach=ICON_REACH):
    """(cx, cy, r) moved to where the print's icon rim runs: centre within reach px, radius within ICON_GROW, both to
    ICON_STEP px (a Scarlet & Violet print's row icons sit a pixel or so off the Mega Evolution places, and the
    composers' radius is not the print's to the half pixel). The rim is the circle whose inside and outside differ
    all the way round: the median over the circle, as an icon's own figure (a bolt, a flame) is edged only in part.
    Searched whole pixels first, then in ICON_STEP round the best."""
    cx, cy, r = _rim_search(rgb, cx, cy, r, np.arange(-reach, reach + 1e-6, 1.0),
                            np.arange(-ICON_GROW, ICON_GROW + 1e-6, 0.5))
    fine = np.arange(-0.75, 0.75 + 1e-6, ICON_STEP)
    return _rim_search(rgb, cx, cy, r, fine, fine)


def _rim_search(rgb, cx, cy, r, shifts, grows):
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    ox, oy = (a[..., None] for a in np.meshgrid(shifts, shifts))
    best, where = -1.0, (cx, cy, r)
    for rr in r + grows:
        ring = lambda at: _bilinear(rgb, cx + ox + at * cos, cy + oy + at * sin)
        score = np.median(np.linalg.norm(ring(rr - RIM_SIDE) - ring(rr + RIM_SIDE), axis=-1), axis=-1)
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        if score[iy, ix] > best + 1e-6:
            best, where = float(score[iy, ix]), (cx + shifts[ix], cy + shifts[iy], rr)
    return where


def _bilinear(img, xs, ys):
    xs, ys = np.clip(xs, 0, W - 1.001), np.clip(ys, 0, H - 1.001)
    x0, y0 = xs.astype(int), ys.astype(int)
    fx, fy = xs - x0, ys - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx
    bottom = img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _printed_icons(rgb, card):
    """The energy icons the card prints, found from its data: [(cx, cy, r)] of each cost type matched against its
    icon in the panel's cost column, and of the weakness row's icons where the composers set them (a white Colorless
    icon on the Colorless panel hardly stands out)."""
    circles = []
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
            circles.append((x0 + bx + scx - ox, y0 + by + scy - oy, r))
            score[max(0, by - ICON_APART):by + ICON_APART, max(0, bx - ICON_APART):bx + ICON_APART] = 0
    circles += [mecompose.ROW_ICONS[side] for side in ("weakness", "resistance") if card.get(side)]
    cx, cy, r, step = mecompose.ROW_ICONS["retreat"]
    circles += [(cx + i * step, cy, r) for i in range(card["retreat"])]
    return circles


def _icons_cover(rgb, circles):
    """Coverage of the energy icons, each fitted to the print's own rim."""
    cover = np.zeros((H, W), np.float32)
    for cx, cy, r in circles:
        cover = np.maximum(cover, _disc(*_on_rim(rgb, cx, cy, r)))
    return cover


def _soft(solid):
    """Coverage of a solid printed shape (bool), anti-aliased over a pixel."""
    return np.maximum(solid, cv2.GaussianBlur(solid.astype(np.float32), (0, 0), SOFT_EDGE))


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


def _edges(rgb):
    grey = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1))


def _shift(img, dx, dy):
    """img moved dx px right and dy down; what it uncovers repeats its edge."""
    ax, ay = abs(dx), abs(dy)
    pad = np.pad(img, ((ay, ay), (ax, ax)) + ((0, 0),) * (img.ndim - 2), mode="edge")
    return pad[ay - dy:ay - dy + img.shape[0], ax - dx:ax - dx + img.shape[1]]


def offset(rgb, old):
    """(dx, dy): whole px the print sits right of and below its type's template, from the edges only the frame
    prints. Ties go to the smaller shift (a frame without edges across one axis does not move along it)."""
    r = REGISTER_REACH
    a, b = _edges(rgb), np.pad(_edges(old), r, mode="edge")
    best, where = -1.0, (0, 0)
    shifts = sorted(((dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)),
                    key=lambda d: abs(d[0]) + abs(d[1]))
    for dx, dy in shifts:
        num = den_a = den_b = 0.0
        for x0, y0, x1, y1 in REGISTER_BOXES:
            pa, pb = a[y0:y1, x0:x1], b[r + y0 - dy:r + y1 - dy, r + x0 - dx:r + x1 - dx]
            num, den_a, den_b = num + (pa * pb).sum(), den_a + (pa * pa).sum(), den_b + (pb * pb).sum()
        score = num / np.sqrt(den_a * den_b + 1e-6)
        if score > best + 1e-9:
            best, where = score, (dx, dy)
    return where


def _stage_pieces(rgb, stage_shown, old):
    """Coverage of the card's Stage pieces: Mega Evolution's, each piece moved to where it matches the print (a
    Scarlet & Violet card's swoosh sits a few pixels right of Mega Evolution's), and round them what stands out from
    the old frame (Mega Evolution's swoosh misses some of its own silver where it crosses the bar's light part)."""
    if stage_shown not in (1, 2):
        return np.zeros((H, W), np.float32)
    sprite = meframe.stage_sprite(stage_shown)
    out = _matched_pieces(rgb, sprite)
    close = np.ones((PIECE_GAP, PIECE_GAP), np.uint8)
    near = dilate(cv2.morphologyEx((out > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, close) > 0, PIECE_REACH)
    stands_out = np.clip((np.abs(rgb - old).max(-1) - RIM_FLOOR) / RIM_MIN, 0, 1)
    # the pieces are silver: the bar showing between the swoosh's strokes is not, unless the frame is grey itself
    silver = np.clip((SILVER_SAT + SILVER_RAMP - hsv(rgb)[1]) / SILVER_RAMP, 0, 1)
    silver = np.where(hsv(old)[1] < SILVER_SAT + SILVER_RAMP, 1.0, silver)
    return np.maximum(out, np.where(near, stands_out * silver, 0)).astype(np.float32)


def _matched_pieces(rgb, sprite):
    """The sprite's pieces, each at the shift where its colours best match the print's."""
    alpha = sprite[..., 3] / 255
    out, taken = np.zeros((H, W), np.float32), np.zeros((H, W), bool)
    for x0, y0, x1, y1 in PIECE_BOXES:
        part = np.zeros((H, W), np.float32)
        part[y0:y1, x0:x1] = alpha[y0:y1, x0:x1]
        part[taken] = 0
        taken |= part > 0
        if not part.any():
            continue
        ys, xs = np.nonzero(part > 0.5)
        r = PIECE_SHIFT
        by0, by1 = max(ys.min() - r, 0), min(ys.max() + r + 1, H)
        bx0, bx1 = max(xs.min() - r, 0), min(xs.max() + r + 1, W)
        shown = rgb[by0:by1, bx0:bx1]
        pad_part, pad_rgb = np.pad(part, r), np.pad(sprite[..., :3], ((r, r), (r, r), (0, 0)), mode="edge")
        best, where = None, (0, 0)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                window = np.s_[r + by0 - dy:r + by1 - dy, r + bx0 - dx:r + bx1 - dx]
                a = pad_part[window]
                diff = np.minimum(np.abs(pad_rgb[window] - shown).max(-1), PIECE_CLIP)
                cost = (a * diff).sum() / max(a.sum(), 1)
                if best is None or cost < best - 1e-6:
                    best, where = cost, (dx, dy)
        out = np.maximum(out, _shift(part, *where))
    return out


def _smooth(values, weight, sigma):
    """values (HxWxC) blurred over the pixels of weight (HxW) only: (the blur, the weight round each pixel)."""
    num = cv2.GaussianBlur(values * weight[..., None], (0, 0), sigma).reshape(values.shape)   # keeps a 1-channel axis
    den = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    return num / np.maximum(den, 1e-6), den


def _ground(rgb, old, hole):
    """The print's background where it prints (hole). Prints of a type share its frame pixel for pixel, so it is
    the old template there, its shading moved to the print's (a lossy or resampled scan shades a little off): the
    difference of the open pixels round the hole, smoothed. Where the print's surroundings still differ from the
    template (the art window, a Scarlet & Violet bar's wave), the surroundings are blended in (Telea)."""
    telea = cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), hole.astype(np.uint8), 3,
                        cv2.INPAINT_TELEA).astype(np.float32)
    open_px = (~dilate(hole, GROW)).astype(np.float32)
    x0, y0, x1, y1 = meframe.ART
    open_px[y0:y1, x0:x1] = 0
    diff = rgb - old
    near, weight = _smooth(diff, open_px, GROUND_SHADE)
    wide, _ = _smooth(diff, open_px, 4 * GROUND_SHADE)
    weight = np.clip(weight / GROUND_OPEN, 0, 1)
    shift = weight * near + (1 - weight) * wide
    off, _ = _smooth(np.abs(diff - shift).max(-1, keepdims=True), open_px, GROUND_SHADE)
    trust = np.clip((GROUND_OFF - off[..., 0]) / GROUND_OFF_RAMP, 0, 1)
    trust[y0:y1, x0:x1] = 0
    trust = trust[..., None]
    filled = trust * (old + shift) + (1 - trust) * telea
    return np.where(hole[..., None], filled, rgb)


def _footer_boxes(card_id, rgb):
    """The regulation-mark and set boxes the card prints (bool HxW): near where the composers set them (a Scarlet &
    Violet print's set box is narrower), what stands out from the colour printed round them. No card shows the
    footer under its boxes, so the template's ground there is a guess, too far off for a box's edge."""
    c = tcg.card(card_id)
    s = tcg.set_info(c["set"]["id"])
    boxes = compose.footer_boxes(c.get("regulationMark") or "", mecompose.set_code(c["set"]["id"], s)) > 0.5
    if not boxes.any():
        return boxes
    near = dilate(boxes, BOX_REACH)
    ring = dilate(boxes, BOX_REACH + BOX_RING) & ~near
    x0, y0, x1, y1 = meframe.FOOTER
    ring[:y0], ring[y1:], ring[:, :x0], ring[:, x1:] = False, False, False, False
    round_it = np.median(rgb[ring], axis=0)
    solid = near & (np.abs(rgb - round_it).max(-1) > BOX_DIFF)
    solid = drop_small(solid, BOX_SPECK)            # grain; a box's 1 px outline stays
    return fill_holes(solid)


def _box_edge(rgb, boxes):
    """The boxes' outer edge: (band bool HxW, each band pixel's share of box, the ground round it HxWx3). Every
    box is drawn the same on every type (compose._footer): a dark outline, a white ring, a dark or white fill; the
    outline is lost on a dark panel. So an edge pixel mixes the ground with the outline's dark or, inside the box
    found, with white (past it, a pixel lighter than the ground read there is its grain): its share is read off
    whichever line it lies on; a pixel on neither (the outline and the ring blurred together) is
    box if it is in the box's shape, else ground. The ground is the colour printed round the box there."""
    inner = cv2.erode(boxes.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    band = dilate(boxes, 1) & ~inner
    ring = (dilate(boxes, BOX_REACH + BOX_RING) & ~dilate(boxes, BOX_REACH)).astype(np.float32)
    ground, _ = _smooth(rgb, ring, BOX_RING)
    lift = rgb[band] - ground[band]
    best_share, best_off = np.ones(len(lift)), np.full(len(lift), np.inf)
    in_box = boxes[band]
    for colour, where in ((np.array(BOX_DARK, np.float32), True), (compose.WHITE, in_box)):
        towards = colour - ground[band]
        share = np.clip((lift * towards).sum(-1) / np.maximum((towards * towards).sum(-1), 1.0), 0, 1)
        off = np.linalg.norm(lift - share[:, None] * towards, axis=-1)
        better = (off < best_off) & where
        best_share[better], best_off[better] = share[better], off[better]
    out = np.zeros((H, W), np.float32)
    out[band] = np.where(best_off <= BOX_OFF, best_share, in_box)     # off both lines: as found
    return band, out, ground


def recolour(rgba, card, new_type):
    """The card (RGBA float32 733x1024) on the new type's frame. card: modern.read_card's dict of it as printed
    (its type, Stage, costs and weakness row)."""
    old, new = meframe.template(card["type"])[..., :3], meframe.template(new_type)[..., :3]
    dx, dy = offset(rgba[..., :3], old)
    out, typed, share = (_shift(a, dx, dy) for a in _on_template(_shift(rgba[..., :3], -dx, -dy), card, new_type,
                                                                    old, new))
    with layers.kind(WORDS):
        layers.record(share * typed[..., 0])
    result = rgba.copy()
    result[..., :3] = typed * out + (1 - typed) * rgba[..., :3]
    x0, y0, x1, y1 = meframe.ART
    result[y0:y1, x0:x1] = rgba[y0:y1, x0:x1]
    return result


def _on_template(rgb, card, new_type, old, new):
    """recolour on a print lying on its template: (the new print RGB, where the frame takes it (HxWx1), the words'
    ink share)."""
    card_id, old_type, stage_shown = card["id"], card["type"], card["stage"]
    ink_old, ink_new = ink(old_type), ink(new_type)
    # the frames' own print (row labels and rules, Stage tab, name-bar icon) is where both frames print it; a blemish
    # in one template's fill is not
    frames = frame_print(old) & dilate(frame_print(new), FRAME_GROW)
    cx, cy, r = meframe.ICON
    yy, xx = np.mgrid[:H, :W]
    frames |= np.hypot(xx - cx, yy - cy) < r + ICON_CLEAR
    own = frame_print(rgb) & ~dilate(frames, FRAME_GROW)
    pieces = _stage_pieces(rgb, stage_shown, old)
    mark = _trainer_mark(card_id, rgb, old)
    red = _print_red(rgb)
    badge = _badge(red)
    names = red & ~badge & ~dilate(mark, GROW)
    hole = (dilate(own, GROW) | dilate(pieces > 0.02, PIECE_REACH) | dilate(badge, GROW) | dilate(names, KEYLINE_REACH)
            | dilate(mark, GROW))
    ground = _ground(rgb, old, hole)

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
    circles, is_disc = _discs(labels, n, size)
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
    icons = _icons_cover(rgb, circles + _printed_icons(rgb, card))
    boxes = _footer_boxes(card_id, rgb)
    # Stage pieces as Mega Evolution prints them; a Scarlet & Violet card's sit a pixel or two off at their edge
    piece_rim = (dilate(pieces > 0.5, PIECE_REACH) & hole) * contrast_rim
    # whole shapes: their insides count where they match the frame too (a Colorless icon's white on a light panel)
    solid = np.maximum(icons, _soft(mark))
    box_cover = np.zeros((H, W), np.float32)
    if boxes.any():
        band, box_share, box_ground = _box_edge(rgb, boxes)
        box_cover = np.where(band, box_share, boxes.astype(np.float32))
        kept = np.maximum(kept, box_cover)
    kept = np.maximum.reduce([kept, pieces, piece_rim, solid])
    names = dilate(red_names > 0.05, KEYLINE_REACH)       # the red names and their keylines: set below
    kept[names] = 0
    share = np.where(word_px & (kept <= 0.02) & ~names, share, 0)

    # kept print's rim mixes it with the old ground: that share of the new ground replaces it; elsewhere the new frame
    out = np.where((kept > 0)[..., None], rgb + (1 - kept)[..., None] * (new - ground), new)
    # an icon's, box's or medallion's rim: its share of the shape's own colour over the new frame (no card shows the
    # ground under them, so the ground read there is a guess, and its error would show round them)
    rim = (solid > 0) & (solid < 1) & (solid >= kept) & ~names
    if rim.any():
        whole = solid >= 1
        _, nearest = cv2.distanceTransformWithLabels((~whole).astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(whole)                      # the labels number the whole pixels in this order
        if len(ys):
            at = nearest[rim] - 1
            k = solid[rim][:, None]
            out[rim] = k * rgb[ys[at], xs[at]] + (1 - k) * new[rim]
    # the footer boxes' white ring and fill are white on every print: a lossy scan's frame colour bleeding into
    # them goes; their dark outline takes the frame's tint on every type, so it keeps the print's
    at = (box_cover > 0) & ~names
    if at.any():
        k = box_cover[at][:, None]
        colour = (rgb[at] - (1 - k) * box_ground[at]) / np.maximum(k, BOX_MIN_SHARE)
        lum = luminance(colour)[:, None]
        colour = np.where(lum >= BOX_LIGHT, np.clip(lum, 0, 255), colour)
        out[at] = k * colour + (1 - k) * new[at]
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
    return out, typed, share
