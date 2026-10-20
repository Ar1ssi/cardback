# textless-card-art

Tools for re-making Pokemon TCG card scans.

## svrestyle: older cards -> Scarlet & Violet / Mega Evolution look

Give it TCGdex card ids; it picks the layout from the card:
- **Pokemon V** (Sword & Shield Basic V) -> **Scarlet & Violet ex**: the old frame and text are removed and the art
  rebuilt (LaMa inpainting), then the SV frame for the card's type, Gill Sans lettering, ex logo, rule box and footer
  are set from the card's real data.
- **Regular Pokemon** (Sun & Moon, Sword & Shield; Basic, Stage 1, Stage 2) -> **Mega Evolution regular card**: the art
  window moves over as is (it sits in the same place in all three eras), a Stage card's old evolution portrait is
  inpainted out and its pre-evolution picture moved into the Mega Evolution portrait ring, and the card is set on a
  blank Mega Evolution frame of its type with name, HP, Pokedex line, abilities/attacks, weakness row, flavour text
  and footer.

```sh
pip install -r requirements.txt            # torch: --index-url https://download.pytorch.org/whl/cpu
export SVRESTYLE_FONTS=/path/to/gill-sans  # licensed fonts, not in this repo
export SVRESTYLE_TCGLIVE=/path/to/TCGLive_Extract   # optional: scans from Pokemon TCG Live (see below)
python -m svrestyle prepare                # once: model + reference scans + frame assets (~/.cache/svrestyle)
python -m svrestyle card swsh6-99 --no-attacks          # Galarian Slowking V -> ex, full art, no attacks
python -m svrestyle card swsh9-040 swsh1-138 --style panel
python -m svrestyle card swsh3-88 sm10-19 sma-SV33      # Tyranitar, Kartana, Shiny Vault Kartana -> Mega Evolution
python -m unittest discover tests
```

### Scans from Pokemon TCG Live

By default scans come from pokemontcg.io. With `--tcglive DIR` (any command) or `SVRESTYLE_TCGLIVE=DIR`, they are
taken from a Pokemon TCG Live extract instead: the game's 1024x1024 card fronts, decoded from
`DIR/cardart/bundles/<set>_en_<NNN>` (needs `pip install UnityPy`) or read from `DIR/WebApp/assets/cards/<set>_<N>.webp`.
A front is the 733x1024 card at the same scale plus 145 px of bleed on each side, so it is cropped, given the printed
card's rounded corners and used with the same layout constants; the reference cards the frames are built from come
from it too. TCGdex ids map to TCG Live ids by rule (`swsh6-99` -> `swsh6_99`, `sv03.5-065` -> `sv3-5_65`) plus
`tcg.TCGLIVE_SETS` (`sma-SV33` -> `sm11-5_102`, `swshp-SWSH055` -> `swshbsp_55`). A card the extract lacks falls back
to pokemontcg.io with a warning. Everything built from these scans is cached under `<cache>/tcglive/`.

### Editor (GUI)

```sh
python -m svrestyle gui            # opens http://127.0.0.1:8765/ in your browser (--port, --no-browser)
```

Load a card by its TCGdex id, or import a ptcg-sim custom-set export to get every older card in it listed. For each
card you can change the name (and regional prefix), HP, "Evolves from", illustrator and flavour text, add, remove and
reorder abilities and attacks, pick each attack's energy cost from the type icons, set its damage (a number with an
optional +, × or −) and text, and set weakness (×2 or +20), resistance and retreat. The preview re-renders as you
edit, in exactly the card layout the command line produces; download one PNG, or every card of an imported set as a
zip (your edits included). The server only listens on 127.0.0.1 and needs no packages beyond requirements.txt.

Options: `--style hybrid` (default, art behind the text) or `panel` (SV translucent type panel) - ex only;
`--no-attacks`; `--debug` (ex: mask overlay + JSON report); `--out DIR`.

Supported types: Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal, Dragon, Colorless (no Fairy:
neither target era has a Fairy frame). Tested on the ex side with swsh6-99, swsh9-040, swsh1-138, swsh4-43, swsh3-19,
swsh7-110, swsh8-113, swsh1-1, swsh1-141, swsh10-078, swsh3-99, swsh12.5-065, swshp-SWSH055, swsh10-101, swsh6-58,
swsh6-80, swsh6-97; on the Mega Evolution side with 22 Sun & Moon / Sword & Shield cards (all types but Lightning),
and checked against real me1 cards re-set by the same code (within 1-4 px).

How it works, module by module:
- `swsh.py` - masks the V frame regions; finds the text block by its white glyphs with a black keyline, the
  Ability badge by colour, and cost icons by circle detection on their fixed slots; LaMa rebuilds the art.
- `frame.py` - builds the SV frame from real scans (`refs.py`): border, window bar, weakness row and rule box
  (type-independent), and per type the name bar (name/HP removed), energy icon, panel palette and ink colour.
- `compose.py` / `typeset.py` - lays out name, HP, abilities, attacks (justified effect text, italic reminder
  text, damage), weakness row and footer, with constants measured on SV scans.
- Energy icons (attack costs, weakness, resistance, retreat, and the ex name-bar type icon) are the Mega Evolution
  ones on both layouts, lifted from real me1 scans (`meframe.COST_ICONS`, `meframe.ICON`); on the Mega Evolution
  layout they also take Mega Evolution's sizes and positions.
- `meframe.py` - blank Mega Evolution frames: Mega Evolution cards of one type share the same bar, panel and footer
  texture, so each printed pixel of a base card is taken from another card of the type that has nothing printed
  there (LaMa only for what every card covers); plus the silver Stage 1 / Stage 2 pieces from Kadabra and Alakazam.
- `edit.py` / `gui.py` / `gui.html` - the editor: which fields can change and the checks on them, the local HTTP
  server, and the page.
- `mecompose.py` - the Mega Evolution layout: art window, portrait, Pokedex line (PokeAPI), the vertical spacing of
  abilities and attacks fitted on 45 me1 cards, flavour text; text drawing is shared with `compose.py`.

For Claude Code sessions on this repo the `sv-restyle` skill (`.claude/skills/sv-restyle/SKILL.md`) runs this.

## Older one-off scripts
`mask-alakazam-ex-mew-065.py` + `lama-inpaint.py`: text removal for Alakazam ex (151 #065); see `HANDOFF.md`.
