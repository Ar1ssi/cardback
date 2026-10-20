"""Scarlet & Violet frame assets, built once from real scans and cached (see refs.py for the sources).

common_frame()   border, art-window bar, weakness-row lines, rule box: the same on every type
name_bar(type)   that type's name bar with the printed name, ex logo and HP removed (LaMa)
profile(type)    that type's energy icon, panel gradient and text colour
ex_logo()        the printed ex logo; ability_badge() the red Ability badge
"""
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import cache, inpaint, refs, tcg
from .imaging import drop_small, fill_holes, hsv, largest_filled, load_rgba, luminance, save_rgba

W, H = 733, 1024
ASSET_VERSION = "1"

INNER_X0, INNER_X1, PANEL_Y1 = 28, 704, 996    # border is x<=27, x>=704, y>=996
BAR_Y0, BAR_Y1, BAR_X0, BAR_X1 = 487, 522, 36, 677
# Lower edge of a Basic's name bar + its silver under-bar (traced on Absol ex and Munkidori ex).
UNDERBAR = [(0, 103), (120, 103), (128, 106), (136, 110), (144, 114), (152, 116), (160, 117), (456, 117),
            (464, 115), (472, 113), (480, 110), (488, 106), (496, 99), (504, 98), (512, 97), (W, 97)]
# Alakazam ex's weakness-row labels/values (removed before its lines and swoosh are lifted).
ALAKAZAM_ROW_TEXT = [(36, 882, 115, 910), (142, 878, 187, 910), (223, 884, 305, 910), (333, 878, 388, 911),
                     (451, 884, 513, 910)]
PANEL_ROWS, PANEL_COLS = slice(560, 850), slice(60, 680)
GRADIENT_STEPS = 33
DARK_INK = (32, 28, 28)


def _asset(name):
    return cache.path("assets", f"v{ASSET_VERSION}", name)


def underbar_bottom():
    xs, ys = zip(*UNDERBAR)
    return np.round(np.interp(np.arange(W), xs, ys)).astype(int)


def is_frame_pixel(rgb):
    hue, sat, val = hsv(rgb)
    return (sat < 0.13) | (val < 0.16) | ((hue > 150) & (hue < 200) & (sat > 0.25))   # silver, black, teal


def alakazam_rowless():
    """Alakazam ex with its weakness-row text inpainted away (the lines and swoosh are lifted from it)."""
    out = _asset("alakazam-rowless.png")
    if out.exists():
        return load_rgba(out)
    scan = tcg.scan(*refs.ALAKAZAM)
    mask = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in ALAKAZAM_ROW_TEXT:
        mask[y0:y1, x0:x1] = True
    clean = inpaint.inpaint(scan, mask)
    save_rgba(clean, out)
    return clean


def common_frame():
    out = _asset("common-frame.png")
    if out.exists():
        return load_rgba(out)
    absol = tcg.scan(refs.ABSOL.card_id, refs.ABSOL.sha)
    alak = tcg.scan(*refs.ALAKAZAM)
    rowless = alakazam_rowless()
    regular = tcg.scan(*refs.HOUNDOUR)
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    card_alpha = absol[..., 3] / 255

    def put(region, src, a=1.0):
        rgb[region] = src[region]
        alpha[region] = a if np.isscalar(a) else a[region]

    # Border: ex art spills over the border on every ex scan, so take a regular Basic's border, darkened to the
    # ex border by a per-channel linear fit on Absol's clean top border (residual ~1.5 levels).
    fit_rows, fit_cols = slice(3, 25), slice(60, 670)
    border_rgb = np.empty((H, W, 3), np.float32)
    for c in range(3):
        gain, offset = np.polyfit(regular[fit_rows, fit_cols, c].ravel(), absol[fit_rows, fit_cols, c].ravel(), 1)
        border_rgb[..., c] = regular[..., c] * gain + offset
    cy0, cy1, cx0, cx1 = 999, 1019, 200, 535           # bridge the regular card's own copyright line
    t = ((np.arange(cy0, cy1) - (cy0 - 1)) / (cy1 - cy0 + 1))[:, None, None]
    border_rgb[cy0:cy1, cx0:cx1] = border_rgb[cy0 - 1, cx0:cx1] * (1 - t) + border_rgb[cy1, cx0:cx1] * t
    border_rgb[PANEL_Y1] = border_rgb[PANEL_Y1 + 1]    # regular cards bevel this row; ex cards do not
    put((xx < INNER_X0) | (xx >= INNER_X1) | (yy >= PANEL_Y1), border_rgb, card_alpha)

    # Art-window bar: per-row median of clean columns, Alakazam's rounded left cap, mirrored for the right.
    band = rowless[BAR_Y0:BAR_Y1]
    band_frame = is_frame_pixel(band[..., :3])
    cap_w = 28
    cap = band[:, BAR_X0:BAR_X0 + cap_w, :3]
    cap_a = fill_holes(band_frame[:, BAR_X0:BAR_X0 + cap_w]).astype(np.float32)
    bar_rgb = np.repeat(np.median(band[:, 64:131, :3], axis=1)[:, None], BAR_X1 - BAR_X0, 1)
    bar_a = np.repeat(np.median(band_frame[:, 64:131].astype(np.float32), axis=1)[:, None], BAR_X1 - BAR_X0, 1)
    bar_rgb[:, :cap_w], bar_a[:, :cap_w] = cap, cap_a
    bar_rgb[:, -cap_w:], bar_a[:, -cap_w:] = cap[:, ::-1], cap_a[:, ::-1]
    full = np.zeros((H, W, 3), np.float32)
    full_a = np.zeros((H, W), np.float32)
    full[BAR_Y0:BAR_Y1, BAR_X0:BAR_X1], full_a[BAR_Y0:BAR_Y1, BAR_X0:BAR_X1] = bar_rgb, bar_a
    put(full_a > 0.5, full)

    # Weakness row: the lines are uniform along their length but art crosses them on every scan, so each is
    # rebuilt from per-row medians of its clean columns; caps, the "|", its hook and the swoosh are keyed.
    row_frame = is_frame_pixel(rowless[..., :3])
    for y0, y1, xa, xb, into_swoosh in [(874, 881, 39, 421, True), (874, 881, 453, 694, False),
                                        (912, 920, 39, 220, False)]:
        cols = [x for x in range(xa + 8, xb - 8) if row_frame[y0:y1, x].all()]
        prof = np.median(rowless[y0:y1, cols, :3], axis=1)
        x1 = xb if into_swoosh else xb - 6
        rgb[y0:y1, xa + 6:x1] = prof[:, None, :]
        alpha[y0:y1, xa + 6:x1] = 1
        if y1 == 881:                       # soft shadow row under the top lines
            rgb[y1, xa + 6:x1] = prof[-1]
            alpha[y1, xa + 6:x1] = 0.45
    keyed = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in [(34, 870, 46, 884), (34, 908, 46, 922), (687, 870, 700, 884),
                           (213, 909, 219, 922), (385, 860, 476, 911)]:
        keyed[y0:y1, x0:x1] = row_frame[y0:y1, x0:x1]
    keyed[874:882, 385:421] = False
    put(keyed, rowless[..., :3])
    sep = np.zeros((H, W), bool)
    sep[887:907, 210:213] = True
    put(sep, luminance(rowless)[..., None].repeat(3, -1))

    # Rule box: Alakazam's printed pill ("Pokemon ex rule" and its text), keyed off the purple art.
    zone = np.zeros((H, W), bool)
    zone[905:996, 245:726] = True
    pill = drop_small(zone & is_frame_pixel(alak[..., :3]), 400)
    pill = cv2.morphologyEx(pill.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    pill = fill_holes(pill)
    rim_x0, rim_x1, rim_rows, foot = 300, 690, (913, 916), (392, 420)
    pill[:rim_rows[1], rim_x0:rim_x1] = False
    put(pill, alak[..., :3], card_alpha)
    alpha[905:rim_rows[0], rim_x0:foot[0]] = 0
    alpha[905:rim_rows[0], foot[1]:rim_x1] = 0
    prof = np.median(alak[rim_rows[0]:rim_rows[1], rim_x0:rim_x1, :3], axis=1)
    rgb[rim_rows[0]:rim_rows[1], rim_x0:rim_x1] = prof[:, None, :]
    alpha[rim_rows[0]:rim_rows[1], rim_x0:rim_x1] = 1
    alpha[rim_rows[0], rim_x0:rim_x1] = np.maximum(0.6, alpha[rim_rows[0] - 1, rim_x0:rim_x1])
    # The rounded left end keys stair-stepped: trace, smooth and re-rasterise its outline.
    ex0, ey0, ex1, ey1 = 243, 900, 300, 996
    _, end_sat, _ = hsv(alak[ey0:ey1, ex0:ex1, :3])
    end_mask = (fill_holes(drop_small(end_sat < 0.25, 400)) | (alpha[ey0:ey1, ex0:ex1] > 0.5)).astype(np.uint8)
    contours, _ = cv2.findContours(end_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    k = 9
    padded = np.concatenate([outline[-k:], outline, outline[:k]])
    smooth = np.stack([np.convolve(padded[:, i], np.ones(k) / k, mode="same")[k:-k] for i in range(2)], -1)
    ss = 4
    poly = Image.new("L", ((ex1 - ex0) * ss, (ey1 - ey0) * ss), 0)
    ImageDraw.Draw(poly).polygon([((x + 0.5) * ss, (y + 0.5) * ss) for x, y in smooth], fill=255)
    end_cov = cv2.resize(np.asarray(poly, np.float32) / 255, (ex1 - ex0, ey1 - ey0), interpolation=cv2.INTER_AREA)
    cut = 292 - ex0
    gained = (end_cov[:, :cut] > 0.02) & (alpha[ey0:ey1, ex0:ex0 + cut] < 0.5)
    lum = luminance(alak[ey0:ey1, ex0:ex0 + cut])[..., None]
    rgb[ey0:ey1, ex0:ex0 + cut] = np.where(gained[..., None], lum, rgb[ey0:ey1, ex0:ex0 + cut])
    alpha[ey0:ey1, ex0:ex0 + cut] = end_cov[:, :cut]
    bottom_rows, bottom_x1 = (983, 988), 676
    starts = {983: 261, 984: 267, 985: 268, 986: 270, 987: 271}    # rows begin diagonally under the end
    prof = np.median(alak[bottom_rows[0]:bottom_rows[1], 292:bottom_x1, :3], axis=1)
    for i, y in enumerate(range(*bottom_rows)):
        rgb[y, starts[y] - 1:bottom_x1] = prof[i]
        alpha[y, starts[y]:bottom_x1] = 1
        alpha[y, starts[y] - 1] = max(alpha[y, starts[y] - 1], 0.5)
    alpha[bottom_rows[1]:PANEL_Y1, 260:bottom_x1] = 0
    solid = np.zeros((H, W), bool)
    solid[rim_rows[0]:PANEL_Y1, 245:726] = alpha[rim_rows[0]:PANEL_Y1, 245:726] > 0.5
    put(fill_holes(solid) & ~solid, alak[..., :3])

    frame = np.dstack([rgb, alpha * 255])
    save_rgba(frame, out)
    return frame


def name_bar(type_name):
    """RGBA: the type's name bar (BASIC tab, bar, under-bar, type icon) with name, ex logo and HP inpainted away;
    alpha marks the name-bar region."""
    out = _asset(f"namebar-{type_name}.png")
    if out.exists():
        return load_rgba(out)
    ref = _ref(type_name)
    scan = tcg.scan(ref.card_id, ref.sha)
    # LaMa alone leaves letter-shaped ghosts on the light bars. The bar's colour changes slowly left to right and
    # has a fixed vertical shading, so its fill is blurred hard along x (ghost strokes are 5-30 px wide) and
    # lightly along y, then LaMa's fine grain is added back; the box edges are feathered into the raw fill.
    boxes = [(136, 36, ref.name_box_right, 93), (512, 34, int(ref.icon[0] - refs.ICON_R - 2), 95)]
    mask = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    clean = inpaint.inpaint(scan, mask)
    rgb = clean[..., :3]
    grain = rgb - cv2.GaussianBlur(rgb, (0, 0), 1.5)
    smooth = cv2.GaussianBlur(rgb, (0, 0), sigmaX=22, sigmaY=1.5)
    feather = cv2.GaussianBlur(cv2.erode(mask.astype(np.float32), np.ones((9, 9), np.uint8)), (0, 0), 3)[..., None]
    rgb[:] = rgb * (1 - feather) + (smooth + 0.5 * grain) * feather    # full grain still traces the old letters
    yy = np.arange(H)[:, None]
    region = yy < underbar_bottom()[None, :]
    bar = np.zeros_like(clean)
    bar[region] = clean[region]
    bar[..., 3] = np.where(region, scan[..., 3], 0)
    save_rgba(bar, out)
    return bar


def profile(type_name):
    """{'icon': (crop, alpha, cx, cy, origin), 'icon_r', 'gradient': Nx3, 'light_text': bool, 'ink': rgb}."""
    out = _asset(f"profile-{type_name}.npz")
    ref = _ref(type_name)
    if not out.exists():
        scan = tcg.scan(ref.card_id, ref.sha)
        cx, cy = ref.icon
        r = refs.ICON_R
        x0, y0 = int(cx - r - 3), int(cy - r - 3)
        crop = scan[y0:int(cy + r + 4), x0:int(cx + r + 4), :3]
        # Panel colours ordered by brightness; text (the extremes) falls outside the 10-90 % band used.
        panel = scan[PANEL_ROWS, PANEL_COLS, :3].reshape(-1, 3)
        lum = luminance(panel)
        order = np.argsort(lum)
        qs = np.linspace(0.1, 0.9, GRADIENT_STEPS)
        idx = (qs * (len(order) - 1)).astype(int)
        half = len(order) // (GRADIENT_STEPS * 2)
        gradient = np.stack([np.median(panel[order[max(0, i - half):i + half + 1]], axis=0) for i in idx])
        light_text = bool(np.median(lum) < 110)
        np.savez(out, crop=crop, origin=np.array([x0, y0]), gradient=gradient, light_text=light_text)
    d = np.load(out)
    x0, y0 = d["origin"]
    cx, cy = ref.icon
    yy, xx = np.mgrid[y0:y0 + d["crop"].shape[0], x0:x0 + d["crop"].shape[1]]
    alpha = np.clip(refs.ICON_R + 0.5 - np.hypot(yy - cy, xx - cx), 0, 1).astype(np.float32)
    light = bool(d["light_text"])
    return {"icon": (d["crop"], alpha, cx, cy, (int(x0), int(y0))), "icon_r": refs.ICON_R,
            "gradient": d["gradient"], "light_text": light,
            "ink": np.array([255, 255, 255] if light else DARK_INK, np.float32)}


def ex_logo():
    """(crop, alpha, left x, right x) of the printed ex logo on Absol's name bar. Its dark outline is not closed in
    the scan, so the silhouette is bright face + warm near-black outline, closed and hole-filled."""
    x0, y0, x1, y1 = 256, 34, 356, 96
    crop = tcg.scan(refs.ABSOL.card_id, refs.ABSOL.sha)[y0:y1, x0:x1, :3]
    lum = luminance(crop)
    outline = (lum < 62) & (crop[..., 0] >= crop[..., 1] + 4)
    shape = cv2.morphologyEx(((lum > 100) | outline).astype(np.uint8), cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    shape = largest_filled(shape.astype(bool))
    alpha = cv2.GaussianBlur(shape.astype(np.float32), (3, 3), 0.5)
    cols = np.nonzero(shape.any(0))[0]
    return (crop, alpha, x0, y0, (x0, y0)), x0 + int(cols.min()), x0 + int(cols.max()) + 1


def ability_badge():
    """Sprite of the red Ability badge (Chien-Pao ex; its blue panel keys out cleanly). Reference point: the
    badge's left/top edge."""
    x0, y0, x1, y1 = 48, 546, 236, 594
    ref = refs.CHIEN_PAO
    crop = tcg.scan(ref.card_id, ref.sha)[y0:y1, x0:x1, :3]
    hue, sat, _ = hsv(crop)
    blue = (hue > 180) & (hue < 230) & (sat > 0.2)
    shape = largest_filled(~blue)
    alpha = cv2.GaussianBlur(shape.astype(np.float32), (3, 3), 0.5)
    ys, xs = np.nonzero(shape)
    return (crop, alpha, x0 + xs.min(), y0 + ys.min(), (x0, y0)), (int(xs.max() - xs.min() + 1),
                                                                  int(ys.max() - ys.min() + 1))


def _ref(type_name):
    if type_name not in refs.TYPES:
        raise ValueError(f"no Scarlet & Violet frame for type {type_name!r} (have: {', '.join(refs.TYPES)})")
    return refs.TYPES[type_name]


def build_all():
    """Build and cache every asset (needs network once, then works offline)."""
    common_frame()
    for t in refs.TYPES:
        name_bar(t)
        profile(t)
    ex_logo()
    ability_badge()
    cache.path("assets", f"v{ASSET_VERSION}", "built.json").write_text(json.dumps(sorted(refs.TYPES)))
