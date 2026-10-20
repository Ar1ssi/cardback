---
name: sv-restyle
description: Re-make a Sword & Shield Pokemon V card (any type) as a Scarlet & Violet "ex" card - SV frame, name bar, Gill Sans lettering, ex logo, rule box and footer - in the full-art "hybrid" layout or the SV translucent-panel layout, with or without attack text. Use when the user names a SWSH V card (by name or TCGdex id like swsh6-99) and wants it restyled, converted to ex, given the gen 9 / SV look, or wants its text or attacks removed in that style.
---

# SV restyle: Sword & Shield V -> Scarlet & Violet ex

Everything is done by `python -m svrestyle` in this repo. Card text comes from TCGdex, the scan from
pokemontcg.io (733x1024), and the SV frame is built from real SV scans listed in `svrestyle/refs.py`.

## 1. Setup (once per machine/session)
1. `pip install numpy pillow opencv-python-headless` and `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Fonts: Gill Sans is licensed and NOT in the repo. Ask the user for their Gill Sans zip if `SVRESTYLE_FONTS`
   is unset; unzip it into its own empty folder outside the repo and `export SVRESTYLE_FONTS=<folder>`.
   Needed faces (the tool names any that are missing): Gill Sans, Gill Sans Medium, Gill Sans Medium Italic,
   Gill Sans Bold, Gill Sans Bold Italic, Gill Sans Heavy Italic, GillSans Condensed Bold.
3. `python -m svrestyle prepare` - downloads the LaMa model (205 MB, checksum-verified) and the reference scans,
   and builds the frame assets into `~/.cache/svrestyle` (about a minute). Later runs reuse the cache.

## 2. Find the card id
TCGdex ids: `curl -s "https://api.tcgdex.net/v2/en/cards?name=<name>"` and pick the `swsh*` entry with a
regular number (not a TG/SV/full-art number). Confirm with the user when several prints match - the regular
print is the lowest local number in the main set. Only Basic Pokemon **V** are supported (no VMAX/VSTAR).

## 3. Render
```
python -m svrestyle card <id> [<id> ...] --out <dir> [--style hybrid|panel] [--no-attacks] [--debug]
```
- `--style hybrid` (default): the art runs behind the text, white text with a dark outline. `panel`: the SV
  translucent type panel behind the attacks, text in the type's ink.
- `--no-attacks`: no abilities or attacks (names, text, energy costs, damage). Weakness/resistance/retreat stay.
- `--debug`: also writes `<name>-mask.png` (what was removed from the original, in green) and a JSON report.
- The first render of a card runs LaMa (~15 s on CPU); its cleaned art is cached for later style changes.

## 4. Check before handing over
Open the PNG and zoom (2-3x) on: the name bar, the text block, the weakness row and the area right under the
name bar. Things to look for, and what to do:
- Leftover old text or icon ghosts in the art: rerun with `--debug`; the mask overlay shows what was missed
  (detection thresholds are at the top of `svrestyle/swsh.py`).
- A warning that the name was squeezed, or that the text block is too tall: say so to the user.
- A warning that a reference scan differs from the one measured: the frame may be misaligned; check it.
Report failures verbatim. Send the PNG to the user (SendUserFile) rather than just naming its path.

## Limits (tell the user when they apply)
- Fairy has no SV frame (SWSH has no Fairy cards anyway); Dragon uses Raging Bolt ex's bar.
- Energy symbols inside effect text come out as words ("Metal Energy"), as TCGdex spells them.
- The footer is set as an ex would print it: two stars (Double Rare), the card's own regulation mark, set
  code, number and copyright year.
- The Single/Rapid/Fusion Strike stamps are part of the art and are kept.

## Changing the look
Layout constants live at the top of `svrestyle/compose.py` (type sizes, positions) and `svrestyle/frame.py`
(frame geometry); each says which scan it was measured on. Delete `~/.cache/svrestyle/assets` after editing
`frame.py` so the assets rebuild. Run `python -m unittest discover tests` after any change.
