"""Small image helpers shared by the frame builder, the art cleaner and the compositor."""
import cv2
import numpy as np
from PIL import Image

from . import layers


def hsv(rgb):
    """-> hue (degrees), saturation, value (0..1)."""
    h = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    return h[..., 0] * 360 / 256, h[..., 1] / 255, h[..., 2] / 255


def luminance(rgb):
    return (rgb[..., :3] * [0.299, 0.587, 0.114]).sum(-1)


def fill_holes(mask, max_area=None):
    """mask with what it encloses filled; max_area: only holes up to that many px."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    hole = np.ones(n, bool)
    hole[list(set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1]))] = False
    if max_area is not None:
        hole &= stats[:, cv2.CC_STAT_AREA] <= max_area
    return mask | hole[labels]


def drop_small(mask, min_px):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[labels]


def largest_filled(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        raise ValueError("no shape found")
    return fill_holes(labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def disc_kernel(radius):
    k = 2 * int(np.ceil(radius)) + 1
    yy, xx = np.mgrid[:k, :k] - k // 2
    return (np.hypot(yy, xx) <= radius + 0.25).astype(np.uint8)


def dilate(mask, radius):
    """Grow a bool mask by a disc."""
    return cv2.dilate(mask.astype(np.uint8), disc_kernel(radius)) > 0


def grow(coverage, radius):
    """Grey-level dilation of a float coverage map: a keyline that keeps the anti-aliasing."""
    return cv2.dilate(coverage, disc_kernel(radius))


def disc_sprite(img, cx, cy, r):
    """(crop rgb, alpha, cx, cy, origin): an anti-aliased disc of radius r cut from img around (cx, cy)."""
    x0, y0 = int(cx - r - 3), int(cy - r - 3)
    x1, y1 = int(cx + r + 4), int(cy + r + 4)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    alpha = np.clip(r + 0.5 - np.hypot(yy - cy, xx - cx), 0, 1).astype(np.float32)
    return img[y0:y1, x0:x1, :3].copy(), alpha, cx, cy, (x0, y0)


def put_sprite(canvas, sprite, cx, cy, scale=1.0, snap=False):
    """Paste a sprite so its reference point lands on (cx, cy), optionally scaled, keeping sub-pixel position.
    snap: at scale 1, land on the nearest whole pixel instead, so the sprite is not resampled (stays sharp)."""
    crop, alpha, scx, scy, (ox, oy) = sprite
    if snap and scale == 1.0:
        cx, cy = round(cx - (scx - ox)) + (scx - ox), round(cy - (scy - oy)) + (scy - oy)
    if scale != 1.0:
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    tx, ty = cx - (scx - ox) * scale, cy - (scy - oy) * scale
    x0, y0 = int(np.floor(tx)), int(np.floor(ty))
    shift = np.float32([[1, 0, tx - x0], [0, 1, ty - y0]])
    h, w = alpha.shape
    crop = cv2.warpAffine(crop, shift, (w + 1, h + 1), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(alpha, shift, (w + 1, h + 1), flags=cv2.INTER_LINEAR)
    cx0, cy0 = max(0, x0), max(0, y0)
    region = canvas[cy0:y0 + h + 1, cx0:x0 + w + 1]
    a = alpha[cy0 - y0:cy0 - y0 + region.shape[0], cx0 - x0:cx0 - x0 + region.shape[1], None]
    src = crop[cy0 - y0:cy0 - y0 + region.shape[0], cx0 - x0:cx0 - x0 + region.shape[1]]
    region[..., :3] = region[..., :3] * (1 - a) + src * a
    layers.record(a[..., 0], cx0, cy0)


def save_rgba(arr, path):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(path)


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
