"""Printed-card wording for the editor's text fields.

Two helpers:
- fix_terms(text): the TCG's fixed terms in their printed capitals ("active pokémon" -> "Active Pokémon"), the
  rest of the text untouched. The list below is the terms that print capitalised in the ptcg-sim corpora
  (out/pkmn-pokemon-cards.json + out/pkmn-trainer-cards.json, 9008 cards, counted 2026-10-23 by
  scripts/wording_terms.py: every term here prints capitalised in 99% or more of its uses, except "Basic", which
  older sets wrote "basic Energy" and every Scarlet & Violet / Mega Evolution card prints "Basic" (437 to 0);
  "damage counter", "attack", "heads" and "tails" print lower-case and are left alone).
- suggest(text, kind, card): a rewrite in printed wording from Claude (model claude-haiku-5-5). With
  ANTHROPIC_API_KEY set it calls the Anthropic Messages API (standard library only); without it, it runs the local
  Claude Code CLI headless (`claude -p`, the machine's Claude Code login, no key needed). available() says whether
  either works.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

# (canonical, plural suffix allowed). Longer phrases are tried first so "Special Condition" wins over "Special".
TERMS = (
    ("Weakness and Resistance", False),
    ("Weakness or Resistance", False),
    ("Mega Evolution Pokémon", False),
    ("Fusion Strike Pokémon", False),
    ("Active Pokémon", False),
    ("Benched Pokémon", False),
    ("Defending Pokémon", False),
    ("Attacking Pokémon", False),
    ("Basic Pokémon", False),
    ("Evolved Pokémon", False),
    ("Evolution Pokémon", False),
    ("Tera Pokémon", False),
    ("Pokémon Checkup", False),
    ("Pokémon Tool", True),
    ("Pokémon Power", True),
    ("Special Condition", True),
    ("Special Energy", False),
    ("Basic Energy", False),
    ("Energy card", True),
    ("Prize card", True),
    ("Trainer card", True),
    ("Supporter card", True),
    ("Item card", True),
    ("Stadium card", True),
    ("Evolution card", True),
    ("Knocked Out", False),
    ("Knock Out", False),
    ("Active Spot", False),
    ("Retreat Cost", False),
    ("Lost Zone", False),
    ("Rule Box", True),
    ("VSTAR Power", False),
    ("GX attack", False),
    ("Ancient Trait", False),
    ("Stage 1", False),
    ("Stage 2", False),
    ("Pokémon-GX", False),
    ("Pokémon-EX", False),
    ("Pokémon", False),
    ("Basic", False),
    ("Energy", False),
    ("Ability", False),
    ("Abilities", False),
    ("Bench", False),
    ("Benched", False),
    ("Active", False),
    ("Prize", True),
    ("Weakness", False),
    ("Resistance", False),
    ("Asleep", False),
    ("Burned", False),
    ("Confused", False),
    ("Paralyzed", False),
    ("Poisoned", False),
    ("Stadium", True),
    ("Supporter", True),
    ("Trainer", True),
    ("Item", True),
    ("Tool", True),
    ("HP", False),
    ("VSTAR", False),
    ("VMAX", False),
)

# fix_terms never looks inside energy symbols: {R} is an icon, not a word.
_SYMBOL = r"\{[A-Za-z]\}"


def _pattern(term, plural):
    words = [re.escape(w).replace("é", "[eé]").replace("É", "[eé]") for w in term.split(" ")]
    body = r"\s+".join(words)
    tail = r"(e?s)?" if plural else "()"
    # a term is a whole word: no letter, digit or hyphen before it, no letter or digit after it (a possessive
    # "Pokémon's" and a hyphenated "Pokémon-GX" still count)
    return re.compile(rf"(?<![\w\-]){body}{tail}(?!\w)", re.IGNORECASE)


_COMPILED = tuple((canonical, _pattern(canonical, plural)) for canonical, plural in TERMS)
_SKIP = re.compile(_SYMBOL)


def fix_terms(text):
    """The text with every fixed term in its printed capitals. Energy symbols and ALL-CAPS words are left as
    typed; the result has the same length as the input, so a caret position stays valid."""
    if not text:
        return text or ""
    pieces, last = [], 0
    for m in _SKIP.finditer(text):
        pieces.append(_fix_span(text[last:m.start()]))
        pieces.append(m.group(0))
        last = m.end()
    pieces.append(_fix_span(text[last:]))
    return "".join(pieces)


def _fix_span(span):
    for canonical, pattern in _COMPILED:
        def swap(m, canonical=canonical):
            found = m.group(0)
            if found.isupper() and len(found) > 2:        # "KNOCKED OUT" in a shout stays a shout
                return found
            fixed = canonical + (m.group(1) or "")
            return fixed if len(fixed) == len(found) else found
        span = pattern.sub(swap, span)
    return span


# ---------------------------------------------------------------- suggestions
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-5-5"
TIMEOUT = 25
MAX_TEXT = 1200

# Printed texts the model is shown as the target register (ptcg-sim corpus rows, by name [set number]).
EXAMPLES = (
    ("attack", "Burning Darkness, Charizard ex [Obsidian Flames 228]",
     "This attack does 30 more damage for each Prize card your opponent has taken."),
    ("attack", "Tail Smash, Bibarel [Brilliant Stars 121]", "Flip a coin. If tails, this attack does nothing."),
    ("attack", "Phantom Dive, Dragapult ex [Twilight Masquerade 200]",
     "Put 6 damage counters on your opponent's Benched Pokémon in any way you like."),
    ("attack", "Bellowing Thunder, Raging Bolt ex [Temporal Forces 218]",
     "You may discard any amount of Basic Energy from your Pokémon. This attack does 70 damage for each card "
     "you discarded in this way."),
    ("attack", "Moonlight Shuriken, Radiant Greninja [Astral Radiance 046]",
     "Discard 2 Energy from this Pokémon. This attack does 90 damage to 2 of your opponent's Pokémon. (Don't "
     "apply Weakness and Resistance for Benched Pokémon.)"),
    ("ability", "Quick Search, Pidgeot ex [Obsidian Flames 225]",
     "Once during your turn, you may search your deck for a card and put it into your hand. Then, shuffle your "
     "deck. You can't use more than 1 Quick Search Ability each turn."),
    ("ability", "Infernal Reign, Charizard ex [Obsidian Flames 228]",
     "When you play this Pokémon from your hand to evolve 1 of your Pokémon during your turn, you may search "
     "your deck for up to 3 Basic {R} Energy cards and attach them to your Pokémon in any way you like. Then, "
     "shuffle your deck."),
    ("ability", "Concealed Cards, Radiant Greninja [Astral Radiance 046]",
     "You must discard an Energy card from your hand in order to use this Ability. Once during your turn, you "
     "may draw 2 cards."),
    ("trainer", "Ultra Ball [Scarlet & Violet 196]",
     "You can use this card only if you discard 2 other cards from your hand.\n\nSearch your deck for a Pokémon, "
     "reveal it, and put it into your hand. Then, shuffle your deck."),
    ("trainer", "Boss's Orders [Paldea Evolved 265]",
     "Switch in 1 of your opponent's Benched Pokémon to the Active Spot."),
    ("trainer", "Rare Candy [Scarlet & Violet 256]",
     "Choose 1 of your Basic Pokémon in play. If you have a Stage 2 card in your hand that evolves from that "
     "Pokémon, put that card onto the Basic Pokémon to evolve it, skipping the Stage 1. You can't use this card "
     "during your first turn or on a Basic Pokémon that was put into play this turn."),
    ("trainer", "Path to the Peak [Astral Radiance 213]",
     "Pokémon with a Rule Box in play (both yours and your opponent's) have no Abilities. (Pokémon V, "
     "Pokémon-GX, etc. have Rule Boxes.)"),
)

SYSTEM = """You rewrite fan-made Pokémon Trading Card Game card text into the exact register of printed cards.
Keep the author's effect: never add, drop or change what the card does, its numbers or its timing. Only the
wording changes. If the text is already printed-style, return it unchanged.

House rules of printed text:
- Fixed terms in capitals: Active Pokémon, Benched Pokémon, Bench, Active Spot, Energy, Energy card, Basic Energy,
  Special Energy, Prize card, Knocked Out, Special Condition, Asleep/Burned/Confused/Paralyzed/Poisoned, Ability,
  Pokémon Tool, Supporter, Item, Stadium, Trainer card, Evolution card, Stage 1, Stage 2, Lost Zone, Weakness,
  Resistance, Retreat Cost, Pokémon Checkup. Lower-case: attack, damage counter, heads, tails, deck, hand, discard pile.
- Energy types are written as symbols in braces: {G} {R} {W} {L} {P} {F} {D} {M} {N} {C} ("a Basic {R} Energy card").
- The card refers to itself as "this Pokémon" and names its own attack or Ability only when limiting it
  ("You can't use more than 1 Quick Search Ability each turn").
- Templates: "Flip a coin. If heads, …" / "If tails, this attack does nothing." · "Search your deck for …, reveal
  it, and put it into your hand. Then, shuffle your deck." · "This attack does 30 more damage for each …" ·
  "This attack does 20 damage to 1 of your opponent's Benched Pokémon. (Don't apply Weakness and Resistance for
  Benched Pokémon.)" · "Put 2 damage counters on …" · "Your opponent's Active Pokémon is now Paralyzed." ·
  "Once during your turn, you may …" · "Switch your Active Pokémon with 1 of your Benched Pokémon." ·
  "Discard … from this Pokémon." · "Heal 30 damage from this Pokémon." · "You may …" for optional steps ·
  "in any way you like" · "Then, shuffle your deck." · "during your next turn" · "can't attack during your next turn".
- Numbers as digits ("2 cards", "1 of your"); "a" only for a single unspecified card ("draw a card").
- Trainer text: one paragraph per rule; a play condition first ("You can use this card only if …").
- Curly apostrophes are fine as straight ones; the editor fixes typography itself.

Answer with JSON only: {"text": "<rewritten text>", "notes": ["<one short note per change, or empty>"]}."""


class SuggestionError(RuntimeError):
    """No suggestion could be had (no key or CLI, network, upstream error, busy, unusable answer)."""


# The site can be public (start-site.bat's tunnel), so the CLI runs on the machine's own Claude Code login with a
# cap: two sessions at once, CLI_PER_MINUTE starts a minute.
CLI_TIMEOUT = 90
CLI_PER_MINUTE = 20
_cli_slots = threading.BoundedSemaphore(2)
_cli_starts = []
_cli_starts_lock = threading.Lock()


def claude_cli():
    """The Claude Code executable, or None. SVRESTYLE_CLAUDE names it explicitly. npm's Windows claude.cmd shim is
    swapped for the claude.exe it wraps: a batch file would pass the arguments through cmd.exe."""
    path = os.environ.get("SVRESTYLE_CLAUDE") or shutil.which("claude")
    if not path:
        return None
    if path.lower().endswith((".cmd", ".bat")):
        exe = os.path.join(os.path.dirname(path), "node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe")
        return exe if os.path.isfile(exe) else None
    return path


def backend():
    """"api" (ANTHROPIC_API_KEY set), "claude-code" (the CLI is installed) or None."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    return "claude-code" if claude_cli() else None


def available():
    return backend() is not None


def _examples_block():
    return "\n\n".join(f"[{kind}] {source}\n{text}" for kind, source, text in EXAMPLES)


def suggest(text, kind="attack", card=None, *, urlopen=urllib.request.urlopen, api_key=None, run=subprocess.run):
    """-> {"text": str, "notes": [str]}. kind: attack | ability | trainer; card: {name, type, stage, hp} context."""
    if kind not in ("attack", "ability", "trainer"):
        raise ValueError("kind: attack, ability or trainer")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text: nothing to rewrite")
    if len(text) > MAX_TEXT:
        raise ValueError(f"text: at most {MAX_TEXT} characters")
    context = {k: v for k, v in (card or {}).items() if k in ("name", "type", "stage", "hp", "kind") and v}
    user = (f"Printed examples of the register:\n\n{_examples_block()}\n\n"
            f"Rewrite this {kind} text. Card: {json.dumps(context, ensure_ascii=False)}\n\n{text}")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return _api_suggest(user, key, urlopen)
    cli = claude_cli()
    if not cli:
        raise SuggestionError("no ANTHROPIC_API_KEY and no Claude Code CLI: install Claude Code (claude) and log in, "
                              "or start the GUI with the key in the environment")
    return parse_reply(_cli_reply(user, cli, run))


def _take_cli_start(now=time.monotonic):
    with _cli_starts_lock:
        t = now()
        _cli_starts[:] = [s for s in _cli_starts if t - s < 60]
        if len(_cli_starts) >= CLI_PER_MINUTE:
            raise SuggestionError(f"at most {CLI_PER_MINUTE} suggestions a minute: try again shortly")
        _cli_starts.append(t)


def _cli_reply(user, cli, run):
    """The model's reply from a headless Claude Code session. The text comes from whoever opens the site, so the
    session gets no tools and no MCP servers, and --safe-mode keeps the machine's CLAUDE.md, hooks, plugins and
    skills out of it. The prompt goes through stdin, never the command line."""
    _take_cli_start()
    if not _cli_slots.acquire(timeout=CLI_TIMEOUT):
        raise SuggestionError("suggestions are busy: try again in a moment")
    args = [cli, "-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--model", MODEL,
            "--no-session-persistence", "--output-format", "json", "--system-prompt", SYSTEM]
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cwd:
            done = run(args, input=user, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=CLI_TIMEOUT, cwd=cwd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as e:
        raise SuggestionError(f"Claude Code took longer than {CLI_TIMEOUT} s") from e
    except OSError as e:
        raise SuggestionError(f"could not start Claude Code: {e}") from e
    finally:
        _cli_slots.release()
    try:
        out = json.loads(done.stdout or "")
    except json.JSONDecodeError:
        out = None
    if not isinstance(out, dict):
        detail = (done.stderr or done.stdout or "").strip()[:300]
        raise SuggestionError(f"Claude Code failed (exit {done.returncode}): {detail or 'no output'}")
    if done.returncode or out.get("is_error"):
        raise SuggestionError(f"Claude Code error: {str(out.get('result') or out.get('subtype') or 'unknown')[:300]}")
    return out.get("result") if isinstance(out.get("result"), str) else ""


def _api_suggest(user, key, urlopen):
    body = json.dumps({"model": MODEL, "max_tokens": 600, "temperature": 0, "system": SYSTEM,
                       "messages": [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json", "x-api-key": key,
                                                 "anthropic-version": "2023-06-01"})
    try:
        with urlopen(req, timeout=TIMEOUT) as r:
            answer = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf8", "replace")[:300]
        raise SuggestionError(f"Anthropic API error {e.code}: {detail}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SuggestionError(f"could not reach the Anthropic API: {e}") from e
    except json.JSONDecodeError as e:
        raise SuggestionError("the Anthropic API answered with something other than JSON") from e
    return parse_answer(answer)


def parse_answer(answer):
    """The model's {"text", "notes"} out of a Messages API response."""
    parts = answer.get("content") if isinstance(answer, dict) else None
    return parse_reply("".join(p.get("text", "") for p in parts or [] if isinstance(p, dict) and p.get("type") == "text"))


def parse_reply(reply):
    """{"text", "notes"} out of the model's reply text."""
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise SuggestionError("the suggestion was not JSON with a text field")
    notes = [str(n) for n in data.get("notes", []) if isinstance(n, (str, int, float))] if isinstance(data.get("notes"), list) else []
    return {"text": fix_terms(data["text"].strip()), "notes": notes}
