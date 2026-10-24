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
- **Trainers** (Item, Supporter, Stadium, Pokemon Tool; Sword & Shield back to Base Set) -> **Scarlet & Violet
  Trainer**: the old art moves into the Gen 9 window and the card is set on the Gen 9 frame of its kind (see
  Trainers below).

```sh
pip install -r requirements.txt            # torch: --index-url https://download.pytorch.org/whl/cpu
export SVRESTYLE_FONTS=/path/to/gill-sans  # licensed fonts, not in this repo
export SVRESTYLE_TCGLIVE=/path/to/TCGLive_Extract   # optional: scans from Pokemon TCG Live (see below)
python -m svrestyle prepare                # once: model + reference scans + frame assets (~/.cache/svrestyle)
python -m svrestyle card swsh6-99 --no-attacks          # Galarian Slowking V -> ex, full art, no attacks
python -m svrestyle card swsh9-040 swsh1-138 --style panel
python -m svrestyle card swsh3-88 sm10-19 sma-SV33      # Tyranitar, Kartana, Shiny Vault Kartana -> Mega Evolution
python -m svrestyle card base1-91 ex3-87 dp1-114 swsh1-156   # Bill, Mr. Briney's Compassion, Speed Stadium, Air Balloon
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

Every card's foil starts from its own plate in the extract. `foil.field` takes what is printed off it (text, icons,
figure): the same set's printings of the same kind with the same foil shapes - a Sword & Shield Darkness reverse
holo's symbols on chevrons, a Sun & Moon Grass reverse's leaves - show the foil under this card's text, and each is
fitted to this card's own tone around every gap. What no card shows (the art window) is bridged smoothly. A
restyled reverse holo is that field wherever the old card had foil, shaped by the Mega Evolution template (art
window, cut-outs); where the old card had none (a plain border) it takes the template's.

### Scarlet & Violet / Mega Evolution cards: edited in place

In the editor, a Scarlet & Violet or Mega Evolution Pokemon (regular or ex) keeps its own card: only what you edit is
taken off and set again (`modern.py`). Name/HP and the weakness row are found by setting the printed values again
with the composers (they match these prints within a few pixels) and taking what stands out there, each printed
shape whole; abilities and attacks by what stands out from the panel in the text zone. LaMa rebuilds what was taken
off, except the weakness row, which is blended along its pixel rows (LaMa copies the labels into it). On a plain
panel, a blot LaMa makes up in a hole is covered with the panel beside it. New blocks follow the Mega Evolution
layout (regular cards; Scarlet & Violet regulars share it) or the Absol ex layout (ex), in the ink the old block
was printed in, read from its lines of glyphs (dark or white, and its keyline when outlined over full art);
an edited ex name keeps the card's own ex logo. With holofoil on, the card's own TCG Live printings are re-made: the
same shaders and the card's own plates, edited the way each plate prints that kind of ink: ink the plate shows (a
cut-out name, an ex name's lit letters, the Stage tab's letters) is taken off and set again at the printed value;
ink the plate runs through (most reverse holos' names and "Evolves from") leaves it as printed. In abilities,
attacks and the weakness row, the old cut-outs (energy circles, the badge) take the card's own foil field, so a
dropped cost leaves no circle behind, and new icons are cut like the printed ones; what every printing cuts (the
row's lines) stays cut. The etch is
carried over, and the front keeps the game's real bleed. In effect text, {G} {R} {W} {L} {P} {F} {D} {M} {N} {C} print as energy symbols (any
layout).

### Stage: Basic, Stage 1, Stage 2, Mega

Every card in the editor (restyled ex, restyled Mega Evolution, edited in place) can be made Basic, Stage 1 or
Stage 2 (`stage.py`). A Stage card gets the modern pieces top left: the STAGE tab, the silver ring with the
pre-evolution's portrait, and the "Evolves from" strip (regular cards also the swoosh in the name bar); they sit at the
same place on every SV / ME card, regular and ex. The portrait is lifted whole from a SV / ME card TCGdex lists as
evolving from that Pokemon (they all print the same picture on the same disc); name a card id under "Portrait from
card" to choose: a SV / ME Stage card gives its portrait, any other card the middle of its art. Mega is a custom stage: the
Stage 1 pieces (ring, portrait, "Evolves from") with the tab reading MEGA, set in Verdana Bold widened and slanted
forward to match the printed BASIC / STAGE lettering (`verdanab.ttf`; without it Tahoma Bold, then Gill Sans Bold),
letter by letter with the printed tabs' 2 px gaps and centred where they centre theirs, on the card's own tab with its
lettering taken off. A Mega card's portrait comes first from
the XY "M ... EX" card of the Pokemon (the figure keyed off its gold disc). Every stage then falls back, in order,
to: an SV / ME card evolving from the Pokemon, an XY Mega card, the Pokemon's official artwork (PokeAPI), and the
middle of one of its cards' art. A card made Basic
gets the BASIC tab and the pieces taken off (LaMa; the plain left border is blended top to bottom). A Stage card
keeps its own strip (Mega ex cards end it under their "Mega-Evolved form of" badge): only the old line is blended
away along the strip's rows. The line is set like the print, which is not Gill Sans: Segoe UI Bold Italic narrowed
and tightened matches it glyph for glyph (`segoeuiz.ttf`, looked up in the fonts folder and the system font folders;
without it, Gill Sans Bold Italic). Only the tab's
lettering is swapped, so each card keeps its own tab and border. On the holofoil plates the ring and strip take the
SV ex frame's silver value on ex cards and are left out elsewhere; the portrait, lettering and "Evolves from" line
are cut out.

### Type (regular Pokemon)

A regular Pokemon (one Prize: no ex, V or other rule) can change type in the editor. A card restyled as a Mega
Evolution card is simply set on the new type's frame. A Scarlet & Violet or Mega Evolution card keeps its own print
(`retype.py`): every regular card of a type prints the same frame (`meframe.template`), so what the card prints that
its type's frame does not (words, numbers, icons, the footer's boxes, the Stage pieces, a Trainer's medallion) is
found against its own background and set on the new type's frame. Words take the new type's ink, each pixel at its
share of ink (Darkness prints white, the others dark); everything else keeps its colours, its rim blended onto the new
ground by its contrast. Energy icons are found from the card's data (its costs matched in the cost column, the
weakness row's icons where the composers set them); the Ability badge and red Ability names by their red, the names
in a white keyline on a dark panel and none on a light one, as the Mega Evolution layout sets them. Silver parts
(border, Pokedex strip, the row's rules, the Stage tab) stay as printed. Attack costs, weakness and resistance are
left as they are: change them too if the new type needs it. With holofoil on, the reverse holo follows the ink:
TCG Live's plates cut white lettering out (Darkness) and run the foil through dark lettering (every other type),
so a card going to or from Darkness has its name, HP, text and the row's labels cut out, or filled with the foil
around them (`foil.lettering_cutouts`); energy symbols, the badge, the rules and the footer stay as printed.

Fit to the print: a Scarlet & Violet print sits up to a pixel off the Mega Evolution templates, so it is moved onto
its template first (matched on the borders and the row's rules) and back after, and each Stage piece is matched where
the print has it (a Scarlet & Violet swoosh sits a few pixels right). The background under the print is the old
template, shaded like the print round it. Energy icons are discs fitted to the print's own rim, covered whole (a
Colorless icon's white on a light panel too) with a geometric anti-aliased edge; the footer boxes are found on the
print and their edges unmixed against the box's dark outline or white ring, the white kept neutral.

The Mega Evolution frame templates take the weakness row's lower rule and the panel's bottom rim from the median of
the type's prints (every card's print reaches over them, and the fills blurred them away), the Pokedex strip and its
shadow from the prints too (each pixel from the prints with no dex text there), and fill the weakness row's value
boxes with the row's own grain from further along it; the rebuilt templates are cached as `me-v4`.

### Trainers

Older Items, Supporters, Stadiums and Pokemon Tools become Scarlet & Violet / Mega Evolution Trainers
(`trainer.py`). Technical Machines are Tools, as they are from Sword & Shield on; a card TCGdex types Item but whose
text is a Tool's (attaching itself, or lending an attack: Curse Powder, the e-Card Cubes) is a Tool too. The first sets
print no type: the kind comes from the rule the text holds, else Item (with a warning).

- **Frame**: every SV / ME Trainer of a kind prints the same frame, so `trainerframe.py` builds each kind's from the
  median of real prints (name, text and footer taken off; the rule pill and tabs kept). `prepare` builds them.
- **Art**: Sun & Moon to HeartGold SoulSilver cards have one art window per era (`trainer.ERA_ART`). Older scans are
  photographs that sit a few px off and a few percent larger or smaller, so `oldwindow.py` registers each card's frame
  to its layout's mean frame (Base Set, e-Card, EX, Diamond & Pearl, and XY Evolutions' redrawn Base Set layout) and
  carries the layout's window over. The art is scaled to cover the Gen 9 window. Era marks over the art (Single /
  Rapid / Fusion Strike, TAG TEAM, Team Plasma) are found by their learned masks and painted out with LaMa
  (`badges.py`).
- **Text** from TCGdex: the old rule sentences the Gen 9 pill prints instead are taken out (a Supporter whose text
  needs its place beside the Active Pokemon keeps it); spelled-out types print as energy symbols; a choice's options
  print as hanging bullets; straight quotes become curly. The effect is justified by letter-spacing and broken for the
  paragraph as a whole, its first baseline on the prints' grid. Fossils and dolls get their HP in the bar and "a N-HP
  Basic {C} Pokemon" in the text; Technical Machines their attacks, set like the Mega Evolution cards'.
- **Refused**: full-art, gold and other special prints (no art window to move), ACE SPEC and Prism Star cards, GX-attack
  Z-Crystals and Team Flare Hyper Gear (rules no Gen 9 frame prints), Chaos Tower (played either way up), Classic
  Collection reprints, TCG Pocket cards, and sets of no known layout.

Gill Sans has no Greek letters: a name like "Miracle Sphere α" renders with a warning and an empty box for the letter;
edit the text in the editor.

### Editor (GUI)

```sh
python -m svrestyle gui            # opens http://127.0.0.1:8765/ in your browser (--port, --no-browser)
```

On Windows, `start-site.bat` starts the editor (in its own "svrestyle server" window, unless something already
serves the port) and shares it through a Cloudflare quick tunnel (needs `cloudflared`). The
`https://….trycloudflare.com` address prints in its window and is written to `SVRESTYLE_CACHE\tunnel.log`; it
changes every time the tunnel starts.

Load a card by its TCGdex id, or import a ptcg-sim custom-set export to get every older card in it listed. For each
card you can change the name (and regional prefix), HP, type (regular Pokemon), stage, "Evolves from" and portrait, illustrator and flavour
text, add, remove and
reorder abilities and attacks, pick each attack's energy cost from the type icons, set its damage (a number with an
optional +, × or −) and text, and set weakness (type and ×2 / +10…+40), resistance (type and −10…−40) and
retreat (click + / −, or a retreat symbol to remove it). With `--tcglive DIR`, tick Holofoil to re-make the card's
TCG Live foil with every render: the plates follow the edits (costs, Ability badge, weakness, resistance and retreat
icons are cut out where they are drawn); preview the foil on the card or the white plate itself, per printing (holo /
reverse holo), and download the card with its foil textures as a zip (the custom-set zip includes them too). The preview re-renders as you
edit, in exactly the card layout the command line produces; download one PNG, or every card of an imported set as a
zip (your edits included). The server only listens on 127.0.0.1 and needs no packages beyond requirements.txt.
A Trainer has its own fields: name, a Supporter's character, kind (Item, Supporter, Stadium or Tool), HP (fossils and
dolls), illustrator and effect text (a paragraph per line), and its Abilities or attacks; holofoil is for Pokemon
only. A Scarlet & Violet or Mega Evolution Trainer is edited too: it keeps its art and is set again, text and
all, on its kind's Gen 9 frame (an inline ex logo in its text prints as plain "ex").

The foil preview is TCG Live's own: its TPCi/Cards3D shaders (the extract's WebApp translation to WebGL,
`WebApp/web/data/foil/`) and MAT_Cards3D_* materials, with their spectrum, shine, glitter and pattern textures
from the extract's shaders bundle (`pokemon_Pokemon TCG Live/shadersbundle/Texture2D/`, lossless), drawing the
render's front, plates, etch and cold foil (`svrestyle/static/foil.js`, the same port the ptcg-sim client runs).
The card turns with the pointer for the light, else sways slowly. three.js 0.185.0 is vendored in
`svrestyle/static/three/` (MIT, LICENSE.txt there); the page needs WebGL2.

Changes: every render lists what differs from the printed card (the restyle itself, name, HP, type, stage, "Evolves
from", portrait, each ability and attack by name: cost, damage, text, added, removed, reordered; weakness,
resistance, retreat, illustrator, flavour text); the card list counts them per card. View the card edited, as
printed, or both side by side. "Changelog (.md)" writes every card's changes as Markdown, and each zip download
holds the same CHANGELOG.md.

Projects: every card you load joins the card list; the Cards panel adds several by id at once (commas or spaces;
those TCGdex does not know, or that are not Pokémon, stay in the box with the reason) and × takes one out with its
edits (Undo puts it back); the find box above the list narrows it to the cards whose name or id holds what you type
(Enter opens the first). Drag a row to move it, or right click it to send it to the top or bottom; the list's
order is the order Render all, the zip and the change log use. The page saves itself 0.3 s after each change - the list, every
card's edits and options, the card open and the holofoil switch - and opens it again on its own next time. Work
with no name yet is saved as "<set> <date time>" (or "Untitled <date time>"); type a name and Save to rename it. New starts an empty project (the open one is saved first and stays in the list);
it saves itself with its first change, or under the name typed and Save. Saved projects (`SVRESTYLE_CACHE/projects/*.json`) can be opened
or deleted from the same bar, from any browser that reaches the server.

Wording help: effect text (abilities, attacks, Trainer text) gets the TCG's fixed terms in their printed capitals as
you type - "active pokemon" becomes "Active Pokémon", "knocked out" "Knocked Out", "energy card" "Energy card" - and
the server does the same to every saved edit (`svrestyle/wording.py`; the list is the terms the ptcg-sim card corpora
print capitalised, counted by `scripts/wording_terms.py`). Names and flavour text are left as typed. "Suggest wording"
under each effect field asks Claude (model `claude-haiku-5-5`) for the same effect in printed-card wording; the
suggestion shows with its notes and nothing changes until you click Use. With `ANTHROPIC_API_KEY` in the server's
environment it calls the Anthropic Messages API (the key never reaches the page). Without a key it runs the server
machine's Claude Code CLI headless (`claude -p`, on that machine's Claude Code login): no tools, no MCP servers, and
`--safe-mode`, so the machine's CLAUDE.md, hooks and plugins stay out; at most 2 at once and 20 a minute, since the
tunnel makes the site public. `SVRESTYLE_CLAUDE` names the `claude` executable if it is not on `PATH`. With neither,
the button is disabled.

Working together: everyone with the same project open edits one copy. Each change goes to the server as a patch of
only the cards and fields that changed, and the others' pages take it in about half a second (long polls, which
pass through Cloudflare quick tunnels; event streams do not). Two people editing different fields, or different
cards, both keep their work; the same field at once, the later one wins. Text being typed is never overwritten
while it is unsent. The header shows how many people have the project open.

### Custom sets for ptcg-sim

The list is also a custom set the simulator (Ar1ssi/vibecodepokemontcg2) plays. Type a set code (2-5 capitals or
digits, the code a decklist prints: `4 Tyranitar HBW 1`) above the list and click "Download for ptcg-sim (.zip)". The
zip holds one folder, `custom<code>/` (`customhbw/`): `set.json` with every card in TCGdex's record shape (name, HP,
type, stage, attacks, Abilities, weakness, resistance, retreat, Trainer text; your edits applied), each card as
`<nnn>/high.webp` (733x1024) and `low.webp`, numbered in list order, and the CHANGELOG. Unzip it into the simulator's
`client/src/assets/custom-sets/` (the running simulator picks it up, no restart): the set shows under the deck builder's Custom pill, its cards
import from a decklist by that code, and the rules engine reads their text like any printed card's (the simulator's
TCGdex proxy serves the folder as TCGdex records). Each card in `set.json` also keeps the TCGdex card it was made from
with your edits (`custom`), so "Import custom set…" opens the `set.json` again with everything restored. A Mega card's
stage is the slot it evolves into (one above its pre-evolution, looked up on TCGdex; Stage 1 when that fails, noted in
the CHANGELOG). Foil textures are not part of this export. `svrestyle/export.py` is the code.

With the simulator on the same machine, skip the zip: set `SVRESTYLE_PTCG_SIM` to its checkout (or start the GUI with
`--ptcg-sim DIR`; `start-site.bat` sets it when the simulator is in `Downloads\vibecodepokemontcg2-main`) and "Send to
ptcg-sim" appears under the set code. It writes the same files straight into the simulator's
`client/src/assets/custom-sets/<set id>/`; the simulator watches that folder, reloads the set while it runs and
refreshes open deck builders. Only cards whose edits or options changed since the last send are rendered (Shift+click
renders all of them: do that once after updating the editor, whose new rendering a kept card would not show), only
changed files are written, and card folders past the end of the list are removed. "Keep ptcg-sim in sync" sends after
every render of a card in the list. Games already running keep the cards they loaded. The button shows, and the route
answers, only in a browser on the machine running the editor: visitors through the Cloudflare tunnel cannot write
into the simulator.

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
- `lifted.py` - HP, damage and weakness-row values, and the energy symbols in effect text, in the print's own
  glyphs: SV / ME figures are proportional and unlike any Gill Sans face, and the symbols print bare in the text's
  ink, so each glyph is lifted off scans (cut out, aligned, averaged at 4x) and placed as fitted on them (figures by
  their advances; symbols from their gap, found by matching the words either side to our layout of the card's own
  text). Built once by `prepare` or the first render (about a minute from the TCG Live extract); a value with a glyph
  never printed (+20) falls back to Gill Sans, a symbol to the icon's symbol in a disc.
- Energy icons (attack costs, weakness, resistance, retreat, and the ex name-bar type icon) are the Mega Evolution
  ones on both layouts, lifted from real me1 scans (`meframe.COST_ICONS`, `meframe.ICON`); on the Mega Evolution
  layout they also take Mega Evolution's sizes and positions.
- `meframe.py` - blank Mega Evolution frames: Mega Evolution cards of one type share the same bar, panel and footer
  texture, so each printed pixel of a base card is taken from another card of the type that has nothing printed
  there (LaMa only for what every card covers); plus the silver Stage 1 / Stage 2 pieces from Kadabra and Alakazam.
- `retype.py` - a Scarlet & Violet / Mega Evolution regular card's print lifted off its type's frame and set on
  another's (see Type above).
- `stage.py` - Stage pieces, evolution portrait (found on TCGdex) and "Evolves from" line, for every layout.
- `edit.py` / `gui.py` / `gui.html` - the editor: which fields can change and the checks on them, the local HTTP
  server, and the page.
- `mecompose.py` - the Mega Evolution layout: art window, portrait, Pokedex line (PokeAPI), the vertical spacing of
  abilities and attacks fitted on 45 me1 cards, flavour text; text drawing is shared with `compose.py`.

For Claude Code sessions on this repo the `sv-restyle` skill (`.claude/skills/sv-restyle/SKILL.md`) runs this.

## Older one-off scripts
`mask-alakazam-ex-mew-065.py` + `lama-inpaint.py`: text removal for Alakazam ex (151 #065); see `HANDOFF.md`.
