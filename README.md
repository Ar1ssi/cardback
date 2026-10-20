# textless-card-art

Tools for re-making Pokemon TCG card scans.

## svrestyle: Sword & Shield V -> Scarlet & Violet ex

Takes any Sword & Shield Basic Pokemon V by its TCGdex id and rebuilds it as a Scarlet & Violet ex card: the
old frame and text are removed and the art rebuilt (LaMa inpainting), then the SV frame for the card's type,
Gill Sans lettering, ex logo, rule box and footer are set from the card's real data.

```sh
pip install -r requirements.txt            # torch: --index-url https://download.pytorch.org/whl/cpu
export SVRESTYLE_FONTS=/path/to/gill-sans  # licensed fonts, not in this repo
python -m svrestyle prepare                # once: model + reference scans + frame assets (~/.cache/svrestyle)
python -m svrestyle card swsh6-99 --no-attacks          # Galarian Slowking ex, full art, no attacks
python -m svrestyle card swsh9-040 swsh1-138 --style panel
python -m unittest discover tests
```

Options: `--style hybrid` (default, art behind the text) or `panel` (SV translucent type panel);
`--no-attacks`; `--debug` (mask overlay + JSON report); `--out DIR`.

Supported: Basic V of every SWSH type (Grass, Fire, Water, Lightning, Psychic, Fighting, Darkness, Metal,
Dragon, Colorless). Tested on swsh6-99, swsh9-040, swsh1-138, swsh4-43, swsh3-19, swsh7-110, swsh8-113, swsh1-1,
swsh1-141, swsh10-078.

How it works, module by module:
- `swsh.py` - masks the V frame regions; finds the text block by its white glyphs with a black keyline, the
  Ability badge by colour, and cost icons by circle detection on their fixed slots; LaMa rebuilds the art.
- `frame.py` - builds the SV frame from real scans (`refs.py`): border, window bar, weakness row and rule box
  (type-independent), and per type the name bar (name/HP removed), energy icon, panel palette and ink colour.
- `compose.py` / `typeset.py` - lays out name, HP, abilities, attacks (justified effect text, italic reminder
  text, damage), weakness row and footer, with constants measured on SV scans.

For Claude Code sessions on this repo the `sv-restyle` skill (`.claude/skills/sv-restyle/SKILL.md`) runs this.

## Older one-off scripts
`mask-alakazam-ex-mew-065.py` + `lama-inpaint.py`: text removal for Alakazam ex (151 #065); see `HANDOFF.md`.
