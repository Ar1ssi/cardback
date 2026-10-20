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

### Holofoil (`--foil`)

With a TCG Live extract, `card ... --foil` also writes the card's holofoil the way TCG Live draws it: a 1024x1024
`<card>-front.png` (the card plus bleed, the game's texture layout), white plates `<card>-wp.png` (base printing)
and `<card>-wp-ph.png` (reverse holo), and `<card>-foil.json` naming the shader, material and pattern for each
(`foil.py`). The foil is the new era's: ex cards get Scarlet & Violet ex foil (SunPillar, TEX_SouthernCross); Mega
Evolution cards get SvHolo on the art window if the old card's base printing was foil, and a FlatSilver reverse holo
if it had one. Plates start from the game's own: the old card's plate for the art (the Pokemon stays out of the
foil), Absol ex's or me1 Abra's / Meganium's for the frame; the composers record every stamp and sprite by kind
(`layers.py`), so the HP, icons, badges and footer drawn on the card are cut out of the plate exactly.

### Scarlet & Violet / Mega Evolution cards: edited in place

In the editor, a Scarlet & Violet or Mega Evolution Pokemon (regular or ex) keeps its own card: only what you edit is
taken off and set again (`modern.py`). Name/HP and the weakness row are found by setting the printed values again
with the composers (they match these prints within a pixel or two) and taking what stands out there; abilities and
attacks by what stands out from the panel in the text zone. LaMa rebuilds what was taken off, except the weakness
row, which is blended along its pixel rows (LaMa copies the labels into it). New blocks follow the Mega Evolution
layout (regular cards; Scarlet & Violet regulars share it) or the Absol ex layout (ex), in the card's own text colour;
an edited ex name keeps the card's own ex logo. With holofoil on, the card's own TCG Live printings are re-made: the
same shaders, plates with what was taken off filled and what was set cut out, the etch carried over, and the front
keeps the game's real bleed. In effect text, {G} {R} {W} {L} {P} {F} {D} {M} {N} {C} print as energy symbols (any
layout).

### Stage: Basic, Stage 1, Stage 2

Every card in the editor (restyled ex, restyled Mega Evolution, edited in place) can be made Basic, Stage 1 or
Stage 2 (`stage.py`). A Stage card gets the modern pieces top left: the STAGE tab, the silver ring with the
pre-evolution's portrait, and the "Evolves from" strip (regular cards also the swoosh in the name bar); they sit at the
same place on every SV / ME card, regular and ex. The portrait is lifted whole from a SV / ME card TCGdex lists as
evolving from that Pokemon (they all print the same picture on the same disc); name a card id under "Portrait from
card" to choose: a SV / ME Stage card gives its portrait, any other card the middle of its art. A card made Basic
gets the BASIC tab and the pieces taken off (LaMa; the plain left border is blended top to bottom). A Stage card
keeps its own strip (Mega ex cards end it under their "Mega-Evolved form of" badge): only the old line is blended
away along the strip's rows. The line is set like the print, which is not Gill Sans: Segoe UI Bold Italic narrowed
and tightened matches it glyph for glyph (`segoeuiz.ttf`, looked up in the fonts folder and the system font folders;
without it, Gill Sans Bold Italic). Only the tab's
lettering is swapped, so each card keeps its own tab and border. On the holofoil plates the ring and strip take the
SV ex frame's silver value on ex cards and are left out elsewhere; the portrait, lettering and "Evolves from" line
are cut out.

### Editor (GUI)

```sh
python -m svrestyle gui            # opens http://127.0.0.1:8765/ in your browser (--port, --no-browser)
```

Load a card by its TCGdex id, or import a ptcg-sim custom-set export to get every older card in it listed. For each
card you can change the name (and regional prefix), HP, stage, "Evolves from" and portrait, illustrator and flavour
text, add, remove and
reorder abilities and attacks, pick each attack's energy cost from the type icons, set its damage (a number with an
optional +, × or −) and text, and set weakness (type and ×2 / +10…+40), resistance (type and −10…−40) and
retreat (click + / −, or a retreat symbol to remove it). With `--tcglive DIR`, tick Holofoil to re-make the card's
TCG Live foil with every render: the plates follow the edits (costs, Ability badge, weakness, resistance and retreat
icons are cut out where they are drawn); preview the foil on the card (a rough stand-in for the game's shaders that
follows the pointer) or the white plate itself, per printing (holo / reverse holo), and download the card with its
foil textures as a zip (the custom-set zip includes them too). The preview re-renders as you
edit, in exactly the card layout the command line produces; download one PNG, or every card of an imported set as a
zip (your edits included). The server only listens on 127.0.0.1 and needs no packages beyond requirements.txt.

Changes: every render lists what differs from the printed card (the restyle itself, name, HP, stage, "Evolves
from", portrait, each ability and attack by name: cost, damage, text, added, removed, reordered; weakness,
resistance, retreat, illustrator, flavour text); the card list counts them per card. View the card edited, as
printed, or both side by side. "Changelog (.md)" writes every card's changes as Markdown, and each zip download
holds the same CHANGELOG.md.

Projects: every card you load joins the card list. Name the project and Save (top right) to keep the list, every
card's edits and options, the card open and the holofoil switch; after that it saves itself 1.5 s after each change,
and the page opens it again on its own next time. Saved projects (`SVRESTYLE_CACHE/projects/*.json`) can be opened
or deleted from the same bar, from any browser that reaches the server.

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
- `stage.py` - Stage pieces, evolution portrait (found on TCGdex) and "Evolves from" line, for every layout.
- `edit.py` / `gui.py` / `gui.html` - the editor: which fields can change and the checks on them, the local HTTP
  server, and the page.
- `mecompose.py` - the Mega Evolution layout: art window, portrait, Pokedex line (PokeAPI), the vertical spacing of
  abilities and attacks fitted on 45 me1 cards, flavour text; text drawing is shared with `compose.py`.

For Claude Code sessions on this repo the `sv-restyle` skill (`.claude/skills/sv-restyle/SKILL.md`) runs this.

## Older one-off scripts
`mask-alakazam-ex-mew-065.py` + `lama-inpaint.py`: text removal for Alakazam ex (151 #065); see `HANDOFF.md`.
