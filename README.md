# GPT-GU

**A glyph-based symbolic language for hybrid human-AI reasoning.**

GPT-GU maps 243 concepts to Unicode glyphs and uses a fine-tuned ByT5 model to translate between natural language and glyph formulas. A heat-gated temporal memory system lets the system think in glyphs across conversations, discovering unexpected connections through synchronicity detection, trend analysis, and associative network traversal.

```
you> What are the opposing forces in transformation?
gu>  ⫚ (Transformation) / ∞ (Continuity) — change requires something stable to change against.
     Stored: ⫚₪∞⧉ (salience: 0.74)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  gu.py — Claude CLI (hybrid reasoning REPL)             │
│  app/server.py — FastAPI (30+ endpoints, port 8000)     │
│  app/gradio_ui.py — Web UI (mounted at /ui)             │
├─────────────────────────────────────────────────────────┤
│  L0  app/guardrails.py     Lexicon + ByT5 translation   │
│      app/cross_reference.py  Glyph relationships        │
│  L1  app/salience.py       Dual-track salience scoring   │
│  L2  app/trends.py         Trend engine (SQLite)         │
│  L3  app/phase_space.py    Phase-space quadrants         │
│      app/lattice.py        Prime Ramsey lattice          │
│  L4  app/meta.py           Meta-pattern detection        │
│  L5  app/glyph_network.py  Associative Ramsey network    │
├─────────────────────────────────────────────────────────┤
│  app/memory.py             Heat-gated temporal memory    │
│  app/claude_tools.py       10 Claude tool definitions    │
└─────────────────────────────────────────────────────────┘
```

### The Glyph Lexicon

243 concept-to-glyph mappings plus 180+ pair relationships:

| Concept | Glyph | | Concept | Glyph |
|---|---|---|---|---|
| Psyche | ψ | | Memory | ☍ |
| Recursion | 🜏 | | Joy | ⧫ |
| Transformation | ⫚ | | Hope | ⎃ |
| Continuity | ∞ | | Dream | 🜓 |
| Life | ⚘ | | Power | ⌘ |

### The Model

- **Architecture**: ByT5 (byte-level T5) — 384 vocab, 12 encoder / 4 decoder layers
- **Size**: ~1.2 GB (`model.safetensors`)
- **Tasks**: `DICT2GLYPH`, `TEXT2GLYPH`, `FORMULA`, `SCROLL_SUMMARY`, `PAIR_TEXT2GLYPH`
- **Inference**: CPU by default on Windows; set `USE_CUDA=1` for GPU

---

## Memory System

Memories are stored as **glyph formulas** — text is a lossy translation. Retrieval is gated by cosine similarity between the stored heat snapshot and current system temperature (threshold 0.6).

| Layer | Module | What it does |
|---|---|---|
| L0 | `memory.py` | Heat-gated store/recall, synchronicity detection |
| L1 | `salience.py` | Dual-track salience: model judgment + usage patterns |
| L2 | `trends.py` | Per-glyph level/velocity/jerk/temperature over 168h |
| L3 | `phase_space.py` | 4-quadrant mapping (active/novel/dormant/inactive) |
| L3 | `lattice.py` | Prime Ramsey lattice — structural resonance at 30 prime scales |
| L4 | `meta.py` | Recurring cluster detection, emergent link discovery |
| L5 | `glyph_network.py` | Associative network with deep recursive retrieval |

**Design philosophy**: No identity scaffolding — let emergence happen.

---

## Setup

**Requirements**: Python 3.10+

```bash
git clone https://github.com/DMNK154/Gulamoff.git
cd gpt-gu
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

The ByT5 model weights go in `models/guardrailed_clone_v08/`. Set a custom path with:

```bash
export GPTGU_MODEL_DIR=/path/to/model
```

For the Claude CLI, set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Claude CLI (hybrid reasoning)

```bash
python gu.py
```

Interactive REPL where Claude has access to 10 glyph tools. Slash commands:

| Command | Action |
|---|---|
| `/weather` | Cognitive weather report (hot glyphs, trends, sync events) |
| `/heat` | Raw heat map (top 30 glyphs) |
| `/lexicon` | Print all 243 glyphs |
| `/save PATH` | Save conversation to JSON |
| `/load PATH` | Load conversation from JSON |
| `/model NAME` | Switch Claude model |
| `/clear` | Reset conversation |

### FastAPI Server

```bash
uvicorn app.server:app --reload --port 8000
```

Gradio UI available at `http://localhost:8000/ui`.

**Core endpoints**:

```
POST /analyze/text          Analyze text for glyph concepts
POST /analyze/glyphs        Interpret a glyph formula
POST /dict2glyph            Word → glyph
POST /formula               Glyph formula → prose
POST /formula_back           Text → glyph formula
POST /relate                Text → contrastive glyph pair
POST /cross_reference       Glyph relationship profile
POST /scroll_summary        AI-powered text summarization
```

**Memory endpoints**:

```
POST /memory/store           Store text as glyph memory
POST /memory/recall          Heat-gated retrieval
GET  /memory/weather         Cognitive weather
GET  /memory/heat_map        Raw temperatures
GET  /memory/trends          All glyph trend vectors
GET  /memory/phase_space     Phase-space coordinates
GET  /memory/synchronicities Sync events
GET  /memory/resonances      Lattice resonances
POST /memory/deep_recall     Recursive network retrieval
GET  /memory/glyph_network   Network visualization data
```

### Python API

```python
from app.guardrails import ensure_loaded, dict2glyph, formula_to_text, analyze_text
from app.memory import store_memory, recall_memories, get_weather

ensure_loaded()

dict2glyph("Hope")                    # "⎃"
formula_to_text("⎃⧫")                # "hope meeting joy..."
analyze_text("Recursion breeds growth") # {words: [...], pairs: [...], summary: ...}

store_memory("Through pain I found clarity.")
recall_memories(top_n=5)               # heat-gated results
get_weather()                          # {hot_glyphs, trends, synchronicities, ...}
```

---

## Data Files

| File | Purpose |
|---|---|
| `glyph_dataset_jsonl` | ~1000 training examples (JSONL) |
| `memory_store.jsonl` | Persisted memories (generated at runtime) |
| `heat_map.json` | Current glyph temperatures (generated) |
| `trend_data.db` | SQLite — trends, events, patterns, network (generated) |
| `gpt-schema.json` | OpenAPI 3.1 schema for custom GPT actions |

---

## How It Works

**Translation**: The ByT5 model handles bidirectional glyph-text conversion via task-conditional prompts (`TASK=X\nFORMAT=Y\nPROMPT: Z\nANSWER:`). A `VocabSizeFilter` constrains generation to the 384-token vocabulary.

**Heat map**: Every glyph usage bumps its temperature. Temperatures decay exponentially (5% per tick). When you recall memories, the system compares your current heat landscape to each memory's stored snapshot — only contextually relevant memories surface.

**Synchronicity**: The system detects three kinds of unexpected co-activation:
- **Dormant reactivation** — cold glyphs suddenly retrieved
- **Cross-domain bridges** — unrelated glyphs firing together
- **Semantic convergence** — heat-similar memories sharing no glyphs

**Ramsey lattice**: Glyphs are projected onto a circle at 30 prime scales (2..113). Glyphs sharing slot positions across multiple primes are structurally "resonant" — the network uses this for deep associative retrieval.

---

## Configuration

Key constants you can tune:

```python
# memory.py
DECAY_RATE = 0.05            # heat decay per tick
SIMILARITY_THRESHOLD = 0.6   # cosine sim gate for recall
BUMP_AMOUNT = 1.0            # temperature bump per usage

# trends.py
TREND_WINDOW_HOURS = 168.0   # 1-week sliding window
PRUNE_AGE_HOURS = 720.0      # 30-day event retention

# phase_space.py
QUADRANT_SPLIT = 0.5         # high/low axis threshold

# glyph_network.py
DEFAULT_MIN_SHARED_PRIMES = 5  # edge detection threshold
```
