"""Marks some older Trainers print over their art that Gen 9 has nothing for: Sword & Shield's Single, Rapid and
Fusion Strike badges, Sun & Moon's TAG TEAM banner and Black & White's Team Plasma badge. Each is found and painted
out (LaMa) before the art moves into the Gen 9 window.

A badge prints the same pixels on every card that has it, where the art behind differs card to card: its mask is
where the prints in its POOL agree (inside the era's art window), its look their median there; both built once and
cached. A card has the badge when its pixels under the mask are close to that look (the prints that have it differ
from it by 23 at most, all other Trainers of their sets by 61 or more: FOUND_BELOW).
"""
import cv2
import numpy as np

from . import cache, inpaint, tcg
from .imaging import drop_small, fill_holes

ASSET_VERSION = "1"
# badge -> (era, prints of it): the clusters of all regular Trainers of their sets whose badge corners agree.
POOL = {
    "single strike": ("swsh", ("swsh5-121", "swsh5-133", "swsh5-134", "swsh5-137", "swsh5-139", "swsh6-139",
                               "swsh6-144", "swsh6-154", "swsh6-156", "swsh7-158")),
    "rapid strike": ("swsh", ("swsh5-127", "swsh5-128", "swsh5-131", "swsh5-132", "swsh5-138", "swsh6-131",
                              "swsh6-136", "swsh6-151", "swsh6-153", "swsh7-153")),
    "fusion strike": ("swsh", ("swsh8-227", "swsh8-230", "swsh8-231", "swsh8-233", "swsh8-236")),
    "tag team": ("sm", ("sm12-186", "sm12-189", "sm12-193", "sm12-198", "sm12-199", "sm12-202")),
    "team plasma": ("bw", ("bw8-118", "bw8-119", "bw8-123", "bw8-124", "bw8-125", "bw9-100", "bw9-101", "bw9-102",
                           "bw9-104", "bw9-105", "bw10-86", "bw10-92", "bw10-93")),
}
AGREE = 12                 # per-pixel spread across the prints (0-255) below which a pixel is the badge's
MIN_PART = 200             # px: smaller agreeing specks are art that happens to match
FOUND_BELOW = 40           # mean difference from the badge's look under its mask
GROW = 4                   # px painted out around the mask: the scans sit a pixel or two apart
PAD = 24                   # px of the art mirrored round the window for LaMa: it sees no frame


def _asset(name):
    return cache.path("assets", f"trainer-v{ASSET_VERSION}", name)


def look(badge, window):
    """(mask HxW bool, median RGB HxWx3) of a badge inside the era's art window (x0, y0, x1, y1); built once."""
    out = _asset(f"badge-{badge.replace(' ', '-')}.npz")
    if out.exists():
        with np.load(out) as f:
            return f["mask"], f["look"]
    prints = np.stack([tcg.scan(i)[..., :3] for i in POOL[badge][1]])
    x0, y0, x1, y1 = window
    inside = np.zeros(prints.shape[1:3], bool)
    inside[y0 + 1:y1 - 1, x0 + 1:x1 - 1] = True
    agree = (prints.std(0).mean(-1) < AGREE) & inside
    agree = cv2.morphologyEx(agree.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    mask = drop_small(fill_holes(agree) & inside, MIN_PART)
    median = np.median(prints, 0).astype(np.float32)
    np.savez_compressed(out, mask=mask, look=median)
    return mask, median


def found(era, scan, window):
    """Names of the badges of this era the scan prints."""
    out = []
    for badge, (badge_era, _) in POOL.items():
        if badge_era != era:
            continue
        mask, median = look(badge, window)
        if mask.any() and float(np.abs(scan[..., :3][mask] - median[mask]).mean()) < FOUND_BELOW:
            out.append(badge)
    return out


def erase(scan, era, window):
    """-> (scan with the era's badges painted out of the art window, their names)."""
    names = found(era, scan, window)
    if not names:
        return scan, names
    x0, y0, x1, y1 = window
    mask = np.zeros(scan.shape[:2], bool)
    for badge in names:
        mask |= look(badge, window)[0]
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * GROW + 1, 2 * GROW + 1), np.uint8)).astype(bool)
    # mirrored, the badge shows in the border too: painted out there as well, or LaMa copies it back in
    art = cv2.copyMakeBorder(np.ascontiguousarray(scan[y0:y1, x0:x1]), PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)
    need = cv2.copyMakeBorder(mask[y0:y1, x0:x1].astype(np.uint8), PAD, PAD, PAD, PAD, cv2.BORDER_REFLECT)
    out = scan.copy()
    out[y0:y1, x0:x1] = inpaint.inpaint(art, need.astype(bool))[PAD:-PAD, PAD:-PAD]
    return out, names
