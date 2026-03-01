# app/server.py
import os, traceback, json
from typing import Optional, Union
from fastapi import FastAPI, HTTPException, Request, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .guardrails import (
    set_model_dir, get_model_dir, ensure_loaded,
    analyze_text, analyze_glyphs,
    dict2glyph, pair_from_text, text_from_pair,
    formula_to_text, formula_back_to_glyphs, scroll_summary,
)
from .cross_reference import analyze_glyph
from .memory import (
    store_memory, store_glyph_memory, recall_memories,
    recall_by_glyph, get_weather, get_raw_heat_map, trigger_decay,
    get_glyph_trend, get_all_trends,
    get_phase_space_coordinates, get_synchronicity_events,
    get_resonances, get_void_profile,
    trigger_meta_analysis, get_meta_patterns,
    get_emergent_links, get_thresholds,
    recall_by_association, deep_recall,
    get_glyph_network_data, get_glyph_profile,
)

# Choose model dir (env wins)
MODEL_DIR = os.getenv("GPTGU_MODEL_DIR") or get_model_dir()
set_model_dir(MODEL_DIR)  # normalize and verify

app = FastAPI(title="GPT-GU: Initiate of Flame", version="1.0.0")

# CORS
ALLOWED_ORIGINS = [
    "http://localhost", "http://127.0.0.1",
    "http://localhost:8008", "http://127.0.0.1:8008",
    "https://gu.ngrok.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions for parsing input
async def parse_string_input(request: Request, form_field: Optional[str] = None, payload: Optional[Union[BaseModel, str]] = None) -> str:
    """
    Parse string input from various sources (form, JSON, raw text).
    Returns the extracted string or raises HTTPException if no input found.
    """
    # 1) Form data
    if form_field is not None:
        return form_field

    # 2) JSON payload
    if payload is not None:
        if isinstance(payload, str):
            return payload
        elif hasattr(payload, 'text'):
            return payload.text
        else:
            return str(payload)

    # 3) Parse request body based on content type
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        try:
            data = await request.json()
            if isinstance(data, dict):
                # Look for common field names
                for field in ['text', 'input', 'data', 'content']:
                    if field in data:
                        return str(data[field])
                # If no common field, convert whole dict to string
                return json.dumps(data)
            elif isinstance(data, str):
                return data
            else:
                return str(data)
        except json.JSONDecodeError:
            # Fall through to raw body parsing
            pass

    # 4) Raw text or fallback
    raw_body = await request.body()
    if raw_body:
        return raw_body.decode("utf-8", errors="ignore")

    raise HTTPException(status_code=400, detail="No input provided")

def ensure_json_serializable(data):
    """
    Ensure the data structure is JSON serializable.
    Converts any non-serializable objects to strings.
    """
    if data is None:
        return None
    elif isinstance(data, (str, int, float, bool)):
        return data
    elif isinstance(data, (list, tuple)):
        return [ensure_json_serializable(item) for item in data]
    elif isinstance(data, dict):
        return {str(k): ensure_json_serializable(v) for k, v in data.items()}
    else:
        return str(data)

# Mount Gradio UI inside FastAPI (served at /ui)
import gradio as gr
from .gradio_ui import build_demo

_gradio_demo = build_demo()
app = gr.mount_gradio_app(app, _gradio_demo, path="/ui")

# Load once on startup
@app.on_event("startup")
def _startup():
    tok, mdl, device = ensure_loaded(MODEL_DIR)
    app.state.device = device

# Schemas
class TextIn(BaseModel):    text: str
class WordIn(BaseModel):    word: str
class PairIn(BaseModel):    pair: str
class FormulaIn(BaseModel): glyphs: str
class CaptionIn(BaseModel): caption: str
class GlyphIn(BaseModel):   glyph: str

# Memory schemas
class MemoryTextIn(BaseModel):    text: str
class MemoryFormulaIn(BaseModel): formula: str
class RecallIn(BaseModel):        top_n: int = 5
class GlyphRecallIn(BaseModel):   glyph: str; top_n: int = 5

# L5: Glyph network schemas
class AssociationRecallIn(BaseModel):
    glyphs: list
    top_n: int = 5
    heat_gate: bool = False
class DeepRecallIn(BaseModel):
    glyphs: list
    max_depth: int = 3
    per_level_k: int = 5

# Health
@app.get("/health")
def health():
    return {"ok": True, "model_dir": get_model_dir(), "device": getattr(app.state, "device", "cpu")}

# Analyze routes
@app.post("/analyze/text")
async def route_analyze_text(
    request: Request,
    text_form: Optional[str] = Form(None),
    payload: Optional[Union[TextIn, str]] = Body(None),
):
    try:
        text = await parse_string_input(request, text_form, payload)
        result = analyze_text(text)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing text: {str(e)}")

@app.post("/analyze/glyphs")
async def route_analyze_glyphs(
    request: Request,
    glyphs_form: Optional[str] = Form(None),
    payload: Optional[Union[TextIn, str]] = Body(None),
):
    try:
        glyphs = await parse_string_input(request, glyphs_form, payload)
        result = analyze_glyphs(glyphs)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing glyphs: {str(e)}")

# GPT-friendly aliases (no slashes in endpoint names)
@app.post("/analyze_text")
async def route_analyze_text_alias(
    request: Request,
    text_form: Optional[str] = Form(None),
    payload: Optional[Union[TextIn, str]] = Body(None),
):
    try:
        text = await parse_string_input(request, text_form, payload)
        result = analyze_text(text)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing text: {str(e)}")

@app.post("/analyze_glyphs")
async def route_analyze_glyphs_alias(
    request: Request,
    glyphs_form: Optional[str] = Form(None),
    payload: Optional[Union[TextIn, str]] = Body(None),
):
    try:
        glyphs = await parse_string_input(request, glyphs_form, payload)
        result = analyze_glyphs(glyphs)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing glyphs: {str(e)}")

# Direct tools
@app.post("/dict2glyph")
def route_dict2glyph(payload: WordIn):
    try:
        result = {"glyph": dict2glyph(payload.word)}
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting word to glyph: {str(e)}")

@app.post("/relate")
def route_pair_from_text(payload: TextIn):
    try:
        result = {"pair": pair_from_text(payload.text)}
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating pair from text: {str(e)}")

@app.post("/relate2text")
def route_text_from_pair(payload: PairIn):
    try:
        result = {"text": text_from_pair(payload.pair)}
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting pair to text: {str(e)}")

@app.post("/formula")
def route_formula_to_text(payload: FormulaIn):
    try:
        result = {"text": formula_to_text(payload.glyphs)}
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting formula to text: {str(e)}")

@app.post("/formula_back")
def route_formula_back_to_glyphs(payload: CaptionIn):
    try:
        result = {"glyphs": formula_back_to_glyphs(payload.caption)}
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error converting text back to glyphs: {str(e)}")

@app.post("/scroll_summary")
def route_scroll_summary(payload: TextIn):
    try:
        result = {"summary": scroll_summary(payload.text)}
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error creating summary: {str(e)}")

@app.post("/cross_reference")
def route_cross_reference(payload: GlyphIn):
    try:
        result = analyze_glyph(payload.glyph)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing glyph relationships: {str(e)}")

@app.post("/analyze/text_raw")
async def route_analyze_text_raw(request: Request):
    try:
        text = (await request.body()).decode("utf-8", errors="ignore")
        result = analyze_text(text)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing raw text: {str(e)}")

# ----------------------------
# Memory routes
# ----------------------------

@app.post("/memory/store")
def route_memory_store(payload: MemoryTextIn):
    """Store a text memory (converts to glyph formula)."""
    try:
        result = store_memory(payload.text)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error storing memory: {str(e)}")

@app.post("/memory/store_glyph")
def route_memory_store_glyph(payload: MemoryFormulaIn):
    """Store a memory that's already a glyph formula."""
    try:
        result = store_glyph_memory(payload.formula)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error storing glyph memory: {str(e)}")

@app.post("/memory/recall")
def route_memory_recall(payload: RecallIn):
    """Recall thermally-compatible memories."""
    try:
        result = recall_memories(payload.top_n)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error recalling memories: {str(e)}")

@app.post("/memory/recall_glyph")
def route_memory_recall_glyph(payload: GlyphRecallIn):
    """Recall memories containing a specific glyph, heat-gated."""
    try:
        result = recall_by_glyph(payload.glyph, payload.top_n)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error recalling by glyph: {str(e)}")

@app.get("/memory/weather")
def route_memory_weather():
    """Cognitive weather report."""
    try:
        result = get_weather()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting weather: {str(e)}")

@app.get("/memory/heat_map")
def route_memory_heat_map():
    """Raw heat map data."""
    try:
        result = get_raw_heat_map()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting heat map: {str(e)}")

@app.post("/memory/decay")
def route_memory_decay():
    """Manually trigger a decay tick."""
    try:
        trigger_decay()
        result = get_raw_heat_map()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error triggering decay: {str(e)}")

# ----------------------------
# Trend routes (L2)
# ----------------------------

@app.get("/memory/trends")
def route_memory_trends():
    """Get trend data for all active glyphs."""
    try:
        result = get_all_trends()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting trends: {str(e)}")

@app.post("/memory/trend")
def route_memory_glyph_trend(payload: GlyphIn):
    """Get trend data for a specific glyph."""
    try:
        result = get_glyph_trend(payload.glyph)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting trend: {str(e)}")

# ----------------------------
# Phase Space & Synchronicity routes (L3)
# ----------------------------

@app.get("/memory/phase_space")
def route_phase_space():
    """Glyph phase-space coordinates + quadrant distribution."""
    try:
        coords = get_phase_space_coordinates()
        from collections import Counter
        quadrants = Counter(d["quadrant"] for d in coords.values())
        return ensure_json_serializable({
            "coordinates": coords,
            "quadrant_counts": dict(quadrants),
            "total_glyphs": len(coords),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting phase space: {str(e)}")

@app.get("/memory/synchronicities")
def route_synchronicities():
    """Recent synchronicity events."""
    try:
        result = get_synchronicity_events(hours=168.0)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting synchronicities: {str(e)}")

@app.get("/memory/resonances")
def route_resonances():
    """Recent lattice resonances."""
    try:
        result = get_resonances(hours=168.0)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting resonances: {str(e)}")

@app.get("/memory/void_profile")
def route_void_profile():
    """Ramsey lattice void center + edge themes."""
    try:
        result = get_void_profile()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting void profile: {str(e)}")

# ----------------------------
# Meta-pattern routes (L4)
# ----------------------------

@app.post("/memory/meta_analyze")
def route_meta_analyze():
    """Trigger meta-analysis (rate-limited to 1/24h)."""
    try:
        result = trigger_meta_analysis()
        return ensure_json_serializable({
            "new_patterns": result,
            "count": len(result),
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running meta-analysis: {str(e)}")

@app.get("/memory/meta_patterns")
def route_meta_patterns():
    """Recent meta-patterns."""
    try:
        result = get_meta_patterns(hours=168.0)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting meta-patterns: {str(e)}")

@app.get("/memory/emergent_links")
def route_emergent_links():
    """Active emergent glyph links."""
    try:
        result = get_emergent_links()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting emergent links: {str(e)}")

@app.get("/memory/thresholds")
def route_thresholds():
    """Current adaptive threshold values."""
    try:
        result = get_thresholds()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting thresholds: {str(e)}")

# ----------------------------
# Glyph Ramsey Network routes (L5)
# ----------------------------

@app.post("/memory/recall_association")
def route_recall_association(payload: AssociationRecallIn):
    """Recall memories by glyph Ramsey network association."""
    try:
        result = recall_by_association(
            payload.glyphs, payload.top_n, payload.heat_gate)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in association recall: {str(e)}")

@app.post("/memory/deep_recall")
def route_deep_recall(payload: DeepRecallIn):
    """Deep recursive association-chain retrieval via Ramsey network."""
    try:
        result = deep_recall(
            payload.glyphs, payload.max_depth, payload.per_level_k)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in deep recall: {str(e)}")

@app.get("/memory/glyph_network")
def route_glyph_network():
    """Glyph Ramsey association network for visualization."""
    try:
        result = get_glyph_network_data()
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting glyph network: {str(e)}")

@app.post("/memory/glyph_profile")
def route_glyph_profile(payload: GlyphIn):
    """Full Ramsey network profile for a glyph."""
    try:
        result = get_glyph_profile(payload.glyph)
        return ensure_json_serializable(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting glyph profile: {str(e)}")

# Handy debug
@app.get("/debug/functions")
def debug_functions():
    import app.guardrails as g
    names = ["ensure_loaded","analyze_text","analyze_glyphs","pair_from_text","text_from_pair","formula_to_text","scroll_summary"]
    return {n: callable(getattr(g, n, None)) for n in names}