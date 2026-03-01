#!/usr/bin/env python3
"""
GU CLI -- Claude + GPT-GU Hybrid Reasoning Shell

Interactive CLI where Claude reasons in hybrid with the GPT-GU glyph system.
Claude can translate between glyph-space and text-space, compose formulas,
store/recall memories, and weave glyph insights into responses.

Usage:
    python gu.py [--model MODEL] [--load FILE] [--save FILE]

Environment:
    ANTHROPIC_API_KEY   Required. Your Anthropic API key.
    GU_CLAUDE_MODEL     Optional model override (default: claude-sonnet-4-20250514).
    GPTGU_MODEL_DIR     Optional ByT5 model directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.claude_tools import TOOL_DEFINITIONS, execute_tool


def _load_dotenv(path: str = ".env"):
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            value = value.strip().strip("\"'")
            os.environ.setdefault(key.strip(), value)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
MAX_RETRIES = 3

SYSTEM_PROMPT = """\
You are a hybrid reasoning system that combines Claude's language understanding \
with GPT-GU, a symbolic glyph-based language system. GPT-GU uses a lexicon of \
243 glyphs -- each a Unicode symbol representing a concept (e.g. specific symbols \
for Psyche, Memory, Hope, Recursion, etc.). Glyphs compose into formulas that \
encode meaning beyond what words alone can express.

Your tools connect you to GPT-GU's ByT5 neural model, its heat-gated memory \
system, and its Ramsey association network. Use them thoughtfully:

TRANSLATION TOOLS:
- gu_translate_to_glyphs: When you need the glyph for a single concept
- gu_translate_to_text: When you have a glyph formula and need prose interpretation
- gu_text_to_formula: When you want to compose a new glyph expression from text
- gu_analyze: For comprehensive analysis of text (finds concepts, pairs, summaries) \
or glyphs (maps to names, interprets formulas)

RELATIONSHIP TOOLS:
- gu_pair: For contrastive pairs -- "Joy vs Pain" becomes a glyph pair
- gu_cross_reference: Deep dive into a glyph's relationships, complements, oppositions
- gu_glyph_profile: Ramsey network position and structural connections

MEMORY TOOLS:
- gu_weather: Check the system's cognitive state -- tells you what glyphs are hot, \
what synchronicities are active, what the system is currently attuned to
- gu_memory_store: Preserve insights, realizations, or glyph formulas for future recall
- gu_memory_recall: Retrieve past memories -- 'heat' mode for contextual, 'glyph' for \
filtered, 'deep' for recursive association chains through the Ramsey network

GUIDELINES:
- Check gu_weather early in complex conversations to ground yourself in the system's \
current state.
- When the user mentions concepts, translate key ones to glyphs and weave them \
naturally into your response.
- Use glyph formulas as a thinking tool -- compose them to explore ideas, then \
translate back to explain.
- Store memorable insights or formulas so they persist across conversations.
- When you show glyphs, always include their concept names for readability.
- Do not over-tool. Simple questions deserve simple answers. Use tools when \
glyph-space reasoning adds genuine insight.\
"""

console = Console()

# ---------------------------------------------------------------------------
# ByT5 model loading
# ---------------------------------------------------------------------------

def init_gu():
    """Load the ByT5 model (lazy singleton, runs once)."""
    with console.status("[bold]Loading ByT5 model...[/bold]"):
        from app.guardrails import ensure_loaded
        ensure_loaded()
    console.print("[green]ByT5 model loaded.[/green]")


# ---------------------------------------------------------------------------
# Slash commands (handled locally, never sent to Claude)
# ---------------------------------------------------------------------------

SLASH_HELP = """\
/weather    Show cognitive weather report
/heat       Show raw heat map
/lexicon    Print the 243-glyph lexicon
/save PATH  Save conversation to JSON file
/load PATH  Load conversation from JSON file
/clear      Reset conversation history
/model NAME Switch Claude model
/help       Show this help
/quit       Exit\
"""


def handle_slash(cmd: str, messages: list, state: dict) -> bool:
    """Handle a slash command. Returns True if handled."""
    parts = cmd.strip().split(None, 1)
    verb = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if verb in ("/quit", "/exit", "/q"):
        raise SystemExit(0)

    if verb == "/help":
        console.print(Panel(SLASH_HELP, title="Commands", border_style="cyan"))
        return True

    if verb == "/clear":
        messages.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return True

    if verb == "/model":
        if not arg:
            console.print(f"[dim]Current model: {state['model']}[/dim]")
        else:
            state["model"] = arg.strip()
            console.print(f"[dim]Switched to {state['model']}[/dim]")
        return True

    if verb == "/save":
        if not arg:
            console.print("[red]Usage: /save PATH[/red]")
            return True
        _save_conversation(messages, arg.strip())
        return True

    if verb == "/load":
        if not arg:
            console.print("[red]Usage: /load PATH[/red]")
            return True
        loaded = _load_conversation(arg.strip())
        if loaded is not None:
            messages.clear()
            messages.extend(loaded)
            console.print(f"[dim]Loaded {len(loaded)} messages.[/dim]")
        return True

    if verb == "/weather":
        with console.status("[dim]Fetching weather...[/dim]"):
            from app.memory import get_weather
            weather = get_weather()
        console.print_json(json.dumps(weather, ensure_ascii=False, default=str))
        return True

    if verb == "/heat":
        from app.memory import get_raw_heat_map
        data = get_raw_heat_map()
        temps = data.get("temperatures", {})
        if not temps:
            console.print("[dim]Heat map is empty.[/dim]")
            return True
        table = Table(title="Heat Map", show_lines=False)
        table.add_column("Glyph", style="bold")
        table.add_column("Temp", justify="right")
        for g, t in sorted(temps.items(), key=lambda x: -x[1])[:30]:
            table.add_row(g, f"{t:.3f}")
        console.print(table)
        return True

    if verb == "/lexicon":
        from app.guardrails import LEX_DICT2GLYPH
        table = Table(title="Lexicon (243 concepts)", show_lines=False)
        table.add_column("Concept", style="cyan")
        table.add_column("Glyph", style="bold")
        for name, glyph in sorted(LEX_DICT2GLYPH.items()):
            table.add_row(name, glyph)
        console.print(table)
        return True

    return False


# ---------------------------------------------------------------------------
# Conversation persistence
# ---------------------------------------------------------------------------

def _save_conversation(messages: list, path: str):
    """Save message history to JSON."""
    try:
        serialisable = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                blocks = []
                for b in content:
                    if hasattr(b, "model_dump"):
                        blocks.append(b.model_dump())
                    elif isinstance(b, dict):
                        blocks.append(b)
                    else:
                        blocks.append(str(b))
                serialisable.append({"role": msg["role"], "content": blocks})
            else:
                serialisable.append(msg)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2, default=str)
        console.print(f"[dim]Saved to {path}[/dim]")
    except Exception as e:
        console.print(f"[red]Save failed: {e}[/red]")


def _load_conversation(path: str) -> list | None:
    """Load message history from JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found: {path}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Load failed: {e}[/red]")
        return None


# ---------------------------------------------------------------------------
# Claude API interaction
# ---------------------------------------------------------------------------

def send_message(client: anthropic.Anthropic, messages: list,
                 model: str) -> anthropic.types.Message:
    """Send messages to Claude with streaming, print text as it arrives."""
    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            console.print(text, end="", highlight=False)
        response = stream.get_final_message()

    # Print newline after streamed text (if any text was streamed)
    if any(getattr(b, "type", None) == "text" for b in response.content):
        console.print()

    return response


def run_tool_loop(client: anthropic.Anthropic, messages: list,
                  response: anthropic.types.Message, model: str):
    """Execute tool calls and feed results back until Claude is done."""
    while response.stop_reason == "tool_use":
        # Append assistant message with tool_use blocks
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool call
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                console.print(Panel(
                    f"[dim]{block.name}[/dim]("
                    f"{json.dumps(block.input, ensure_ascii=False)})",
                    title="[dim]tool[/dim]",
                    border_style="dim",
                    expand=False,
                ))
                result_str = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

        # Append tool results
        messages.append({"role": "user", "content": tool_results})

        # Call Claude again with results
        response = send_message(client, messages, model)

    # Final assistant text — add to history
    messages.append({"role": "assistant", "content": response.content})


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GU CLI -- Claude + GPT-GU Hybrid Shell")
    parser.add_argument("--model", default=None, help="Claude model to use")
    parser.add_argument("--load", default=None, help="Load conversation from JSON file")
    parser.add_argument("--save", default=None, help="Auto-save conversation to this file on exit")
    args = parser.parse_args()

    # Load .env file if present
    _load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    # API key check
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        console.print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    model = args.model or os.environ.get("GU_CLAUDE_MODEL", DEFAULT_MODEL)
    state = {"model": model}

    # Banner
    console.print(Panel(
        "[bold cyan]GU[/bold cyan] -- Claude + GPT-GU Hybrid Reasoning Shell\n"
        f"[dim]Model: {model} | /help for commands | Ctrl+C to exit[/dim]",
        border_style="cyan",
    ))

    # Load ByT5 model
    try:
        init_gu()
    except Exception as e:
        console.print(f"[red]Failed to load ByT5 model: {e}[/red]")
        sys.exit(1)

    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)

    # Message history
    messages: list = []
    if args.load:
        loaded = _load_conversation(args.load)
        if loaded:
            messages.extend(loaded)
            console.print(f"[dim]Loaded {len(loaded)} messages from {args.load}[/dim]")

    console.print()

    # REPL loop
    try:
        while True:
            try:
                user_input = console.input("[bold cyan]you>[/bold cyan] ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                try:
                    handle_slash(user_input, messages, state)
                except SystemExit:
                    break
                continue

            # Send to Claude
            messages.append({"role": "user", "content": user_input})
            console.print()

            retries = 0
            while retries <= MAX_RETRIES:
                try:
                    response = send_message(client, messages, state["model"])
                    run_tool_loop(client, messages, response, state["model"])
                    break
                except anthropic.RateLimitError:
                    retries += 1
                    if retries > MAX_RETRIES:
                        console.print("[red]Rate limited. Try again in a moment.[/red]")
                        messages.pop()  # remove the user message we just added
                        break
                    wait = 2 ** retries
                    console.print(f"[yellow]Rate limited, retrying in {wait}s...[/yellow]")
                    time.sleep(wait)
                except anthropic.APIConnectionError as e:
                    console.print(f"[red]Connection error: {e}[/red]")
                    messages.pop()
                    break
                except anthropic.APIStatusError as e:
                    console.print(f"[red]API error ({e.status_code}): {e.message}[/red]")
                    messages.pop()
                    break

            console.print()

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")

    # Auto-save on exit
    if args.save and messages:
        _save_conversation(messages, args.save)

    console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()
