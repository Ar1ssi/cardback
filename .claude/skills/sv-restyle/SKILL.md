---
name: sv-restyle
description: Re-make older Pokemon cards in the current look. A Sword & Shield Pokemon V (any type) becomes a Scarlet & Violet "ex" card (SV frame, Gill Sans lettering, ex logo, rule box, footer; full-art "hybrid" or translucent-panel layout, with or without attack text). A regular Sun & Moon or Sword & Shield Pokemon (Basic/Stage 1/Stage 2) becomes a Mega Evolution regular card (its art in the Mega Evolution frame of its type, evolution portrait, Pokedex line, attacks, flavour text). Use when the user names such cards (by name or TCGdex id like swsh6-99, sm10-19) and wants them restyled, made uniform with Scarlet & Violet / Mega Evolution cards, converted to ex, or their text or attacks removed in that style.
---

# SV restyle: older cards -> Scarlet & Violet ex / Mega Evolution regular

Everything is done by `python -m svrestyle` in this repo. Card text comes from TCGdex, the Pokedex line from
PokeAPI, scans from pokemontcg.io (733x1024). The layout follows the card: Pokemon V -> SV ex (frame built from the SV
scans in `svrestyle/refs.py`); regular Pokemon -> Mega Evolution regular card (frame built from me1/me2 scans listed
in `svrestyle/meframe.py`). Cards already from Scarlet & Violet or Mega Evolution need nothing.

## 1. Setup (once per machine/session)
1. `pip install numpy pillow opencv-python-headless` and `pip install torch --index-url https://download.pytorch.org/whl/cpu`
2. Fonts: Gill Sans is licensed and NOT in the repo. Ask the user for their Gill Sans zip if `SVRESTYLE_FONTS`
   is unset; unzip it into its own empty folder outside the repo and `export SVRESTYLE_FONTS=<folder>`.
   Needed faces (the tool names any that are missing): Gill Sans, Gill Sans Medium, Gill Sans Medium Italic,
   Gill Sans Bold, Gill Sans Bold Italic, Gill Sans Heavy Italic, GillSans Condensed Bold.
3. `python -m svrestyle prepare` - downloads the LaMa model (205 MB, checksum-verified) and the reference scans,
   and builds the frame assets into `~/.cache/svrestyle` (about a minute). Later runs reuse the cache.

## 2. Find the card id
TCGdex ids: `curl -s "https://api.tcgdex.net/v2/en/cards?name=<name>"` and pick the `swsh*`/`sm*` entry with a
regular number (not a TG/SV/full-art number). When the user gives a deck or custom-set export (JSON with card ids),
use exactly those ids - a promo there has different art from the set print. Otherwise the regular print is the
lowest local number in the main set. Supported: Basic Pokemon **V** (no VMAX/VSTAR) and regular Basic/Stage 1/
Stage 2 Pokemon; GX, ex, BREAK and Trainer/Energy cards are refused.

## 3. Render
```
python -m svrestyle card <id> [<id> ...] --out <dir> [--style hybrid|panel] [--no-attacks] [--debug]
```
- Output names end `-ex-<style>.png` (V cards) or `-me.png` (regular cards).
- `--style hybrid` (default, ex only): the art runs behind the text, white text with a dark outline. `panel`: the SV
  translucent type panel behind the attacks, text in the type's ink.
- `--no-attacks`: no abilities or attacks (names, text, energy costs, damage). Weakness/resistance/retreat stay.
- `--debug` (ex only): also writes `<name>-mask.png` (what was removed from the original, in green) and a JSON report.
- The first render of a card runs LaMa (~15 s on CPU); its cleaned art is cached for later style changes.

## 3b. Editor
When the user wants to change a card's attacks, abilities, energy costs, damage, weakness, resistance, retreat, name
or HP, or wants a point-and-click tool: `python -m svrestyle gui` (serves http://127.0.0.1:8765/; `--no-browser` in a
headless session). The same edits go through `svrestyle.edit.apply` if you script them instead:
`compose.compose(id, opts, card=edit.apply(compose.read_card(id), edits))` (ex) or
`mecompose.compose_card(id, card=edit.apply(mecompose.read_card(id), edits))` (Mega Evolution).

## 4. Check before handing over
Open the PNG and zoom (2-3x) on: the name bar, the text block, the weakness row and the area right under the
name bar. Things to look for, and what to do:
- Leftover old text or icon ghosts in the art: rerun with `--debug`; the mask overlay shows what was missed
  (detection thresholds are at the top of `svrestyle/swsh.py`).
- A warning that the name was squeezed, or that the text block is too tall: say so to the user.
- A warning that a reference scan differs from the one measured: the frame may be misaligned; check it.
Report failures verbatim. Send the PNG to the user (SendUserFile) rather than just naming its path.

## Limits (tell the user when they apply)
- Mega Evolution layout: regular cards keep their attack text (as real regular cards print it) unless asked;
  `--no-attacks` drops it. Old art and its stamps (Ultra Beast banner, Rapid/Single Strike) are kept as printed.
  Dragon has one Mega Evolution reference (Latios), so its blank frame relies more on inpainting.
  Promos print no set code. Sub-sets without their own code (Shiny Vault) use the main set's (`SUBSET_CODES`).
- Fairy has no SV frame (SWSH has no Fairy cards anyway); Dragon uses Raging Bolt ex's bar.
- Energy symbols inside effect text come out as words ("Metal Energy"), as TCGdex spells them.
- The footer is set as an ex would print it: two stars (Double Rare), the card's own regulation mark, set
  code, number and copyright year.
- The Single/Rapid/Fusion Strike stamps are part of the art and are kept.

## Changing the look
Layout constants live at the top of `svrestyle/compose.py` (type sizes, positions), `svrestyle/frame.py`
(SV frame geometry), `svrestyle/mecompose.py` and `svrestyle/meframe.py` (Mega Evolution); each says which scan it
was measured on. Delete `~/.cache/svrestyle/assets` after editing a frame module so the assets rebuild, and
`~/.cache/svrestyle/art` after changing how art is cleaned. Run `python -m unittest discover tests` after any change.
