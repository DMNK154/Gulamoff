"""
Claude API tool definitions and dispatch registry for GPT-GU.

Defines 10 tools that Claude can invoke to interact with the glyph system:
  Tier 1 (Translation): translate_to_glyphs, translate_to_text, text_to_formula, analyze
  Tier 2 (Relationships): pair, cross_reference, glyph_profile
  Tier 3 (Memory): memory_store, memory_recall, weather
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# JSON-safe serialisation helper (mirrors server.py pattern)
# ---------------------------------------------------------------------------

def _serialise(data: Any) -> Any:
    if data is None:
        return None
    if isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, (list, tuple)):
        return [_serialise(item) for item in data]
    if isinstance(data, dict):
        return {str(k): _serialise(v) for k, v in data.items()}
    return str(data)


def _json_result(data: Any) -> str:
    return json.dumps(_serialise(data), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool Schemas  (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[dict] = [
    # ---- Tier 1: Core Translation ----
    {
        "name": "gu_translate_to_glyphs",
        "description": (
            "Translate a single concept or word into its glyph symbol. "
            "The GPT-GU lexicon contains 243 concept-to-glyph mappings. "
            "If the concept is not in the lexicon, the ByT5 model will "
            "attempt to generate a glyph. Returns a single Unicode glyph."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "The concept to translate, e.g. 'Hope', 'Recursion', 'Memory'",
                },
            },
            "required": ["word"],
        },
    },
    {
        "name": "gu_translate_to_text",
        "description": (
            "Translate a glyph formula (one or more glyph symbols, optionally "
            "connected with operators) into natural language prose. Uses the "
            "ByT5 model with sampling for creative interpretation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "glyphs": {
                    "type": "string",
                    "description": "Glyph formula string, e.g. raw glyph symbols or glyphs with operators",
                },
            },
            "required": ["glyphs"],
        },
    },
    {
        "name": "gu_text_to_formula",
        "description": (
            "Convert natural language text into a glyph formula. The ByT5 "
            "model generates a sequence of glyphs that encode the meaning "
            "of the input text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Natural language text to encode as glyphs",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "gu_analyze",
        "description": (
            "Full analysis of text or glyphs. For text: identifies matching "
            "concepts from the 243-glyph lexicon, detects contrastive pairs, "
            "produces a glyph-enhanced summary. For glyphs: maps each glyph "
            "to its concept name, detects pair relationships, interprets "
            "formulas, provides complete text translation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Text or glyph string to analyze"},
                "mode": {
                    "type": "string",
                    "enum": ["text", "glyphs"],
                    "description": "Whether input is natural language text or glyph symbols",
                },
            },
            "required": ["input", "mode"],
        },
    },
    # ---- Tier 2: Relationships ----
    {
        "name": "gu_pair",
        "description": (
            "Work with glyph pairs (contrastive or complementary relationships). "
            "In 'create' mode: takes a contrast description like 'Joy vs Pain' "
            "and returns the corresponding glyph pair. In 'interpret' mode: "
            "takes a glyph pair like 'A / B' and returns the text description. "
            "The system has 180+ predefined pair relationships."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Contrast text or glyph pair"},
                "mode": {
                    "type": "string",
                    "enum": ["create", "interpret"],
                    "description": "'create' for text->pair, 'interpret' for pair->text",
                },
            },
            "required": ["input", "mode"],
        },
    },
    {
        "name": "gu_cross_reference",
        "description": (
            "Get the complete relationship profile for a glyph: complementary "
            "glyphs (frequently co-occurring), oppositional glyphs (contrastive "
            "pairs), transformation targets (A->B patterns), formula examples, "
            "and scroll contexts. Accepts a glyph symbol or concept name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "glyph_or_name": {
                    "type": "string",
                    "description": "A glyph symbol or concept name, e.g. 'Memory' or a glyph character",
                },
            },
            "required": ["glyph_or_name"],
        },
    },
    {
        "name": "gu_glyph_profile",
        "description": (
            "Get the full Ramsey network profile for a glyph: its angular "
            "position on the lattice, prime slot assignments, strongest edges "
            "to other glyphs with connection types (co-occurrence, semantic, "
            "synchronicity, semiotic), and associated memory count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "glyph": {
                    "type": "string",
                    "description": "A single glyph symbol",
                },
            },
            "required": ["glyph"],
        },
    },
    # ---- Tier 3: Memory ----
    {
        "name": "gu_memory_store",
        "description": (
            "Store a new memory in the glyph memory system. Text is "
            "automatically converted to a glyph formula. You can also store "
            "a pre-composed glyph formula directly. Each memory captures a "
            "heat snapshot which determines future recall eligibility. Returns "
            "the stored record with formula, glyphs, salience, and ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text to store, or a glyph formula if mode is 'glyph'",
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "glyph"],
                    "default": "text",
                    "description": "'text' to auto-convert, 'glyph' for pre-composed formula",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "gu_memory_recall",
        "description": (
            "Recall memories. Three modes: 'heat' returns memories matching "
            "the current activation landscape (default, most contextual). "
            "'glyph' filters memories containing a specific glyph. 'deep' "
            "performs recursive Ramsey network association-chain retrieval "
            "starting from seed glyphs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["heat", "glyph", "deep"],
                    "default": "heat",
                    "description": "Retrieval mode",
                },
                "glyph": {
                    "type": "string",
                    "description": "Required for 'glyph' mode: the glyph to filter by",
                },
                "seed_glyphs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required for 'deep' mode: seed glyph symbols",
                },
                "top_n": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum results to return",
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "gu_weather",
        "description": (
            "Get the cognitive weather report: hottest glyphs and temperatures, "
            "total heat intensity, thermally-reachable memory count, "
            "synchronicity events, trending glyphs, phase-space distribution, "
            "Ramsey lattice resonances, and meta-patterns. Call this to "
            "understand what the glyph system is currently attuned to."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch functions  (tool name -> callable(input_dict) -> str)
# ---------------------------------------------------------------------------

def _dispatch_translate_to_glyphs(inp: dict) -> str:
    from app.guardrails import dict2glyph
    result = dict2glyph(inp["word"])
    return _json_result({"glyph": result, "word": inp["word"]})


def _dispatch_translate_to_text(inp: dict) -> str:
    from app.guardrails import formula_to_text
    result = formula_to_text(inp["glyphs"])
    return _json_result({"text": result, "glyphs": inp["glyphs"]})


def _dispatch_text_to_formula(inp: dict) -> str:
    from app.guardrails import formula_back_to_glyphs
    result = formula_back_to_glyphs(inp["text"])
    return _json_result({"formula": result, "text": inp["text"]})


def _dispatch_analyze(inp: dict) -> str:
    mode = inp["mode"]
    text = inp["input"]
    if mode == "text":
        from app.guardrails import analyze_text
        return _json_result(analyze_text(text))
    else:
        from app.guardrails import analyze_glyphs
        return _json_result(analyze_glyphs(text))


def _dispatch_pair(inp: dict) -> str:
    mode = inp["mode"]
    text = inp["input"]
    if mode == "create":
        from app.guardrails import pair_from_text
        result = pair_from_text(text)
        return _json_result({"pair": result, "input": text})
    else:
        from app.guardrails import text_from_pair
        result = text_from_pair(text)
        return _json_result({"text": result, "pair": text})


def _dispatch_cross_reference(inp: dict) -> str:
    from app.cross_reference import analyze_glyph
    return _json_result(analyze_glyph(inp["glyph_or_name"]))


def _dispatch_glyph_profile(inp: dict) -> str:
    from app.memory import get_glyph_profile
    return _json_result(get_glyph_profile(inp["glyph"]))


def _dispatch_memory_store(inp: dict) -> str:
    mode = inp.get("mode", "text")
    content = inp["content"]
    if mode == "glyph":
        from app.memory import store_glyph_memory
        return _json_result(store_glyph_memory(content))
    else:
        from app.memory import store_memory
        return _json_result(store_memory(content))


def _dispatch_memory_recall(inp: dict) -> str:
    mode = inp.get("mode", "heat")
    top_n = inp.get("top_n", 5)

    if mode == "glyph":
        from app.memory import recall_by_glyph
        glyph = inp.get("glyph", "")
        return _json_result(recall_by_glyph(glyph, top_n))
    elif mode == "deep":
        from app.memory import deep_recall
        seeds = inp.get("seed_glyphs", [])
        return _json_result(deep_recall(seeds))
    else:
        from app.memory import recall_memories
        return _json_result(recall_memories(top_n))


def _dispatch_weather(inp: dict) -> str:
    from app.memory import get_weather
    return _json_result(get_weather())


TOOL_DISPATCH: Dict[str, Callable[[dict], str]] = {
    "gu_translate_to_glyphs": _dispatch_translate_to_glyphs,
    "gu_translate_to_text": _dispatch_translate_to_text,
    "gu_text_to_formula": _dispatch_text_to_formula,
    "gu_analyze": _dispatch_analyze,
    "gu_pair": _dispatch_pair,
    "gu_cross_reference": _dispatch_cross_reference,
    "gu_glyph_profile": _dispatch_glyph_profile,
    "gu_memory_store": _dispatch_memory_store,
    "gu_memory_recall": _dispatch_memory_recall,
    "gu_weather": _dispatch_weather,
}


def execute_tool(name: str, input_data: dict) -> str:
    """Execute a tool by name. Returns JSON string result."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(input_data)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
