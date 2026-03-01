# app/guardrails.py
from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LogitsProcessor, LogitsProcessorList

# ----------------------------
# Configuration
# ----------------------------
MODEL_DIR = os.getenv(
    "GPTGU_MODEL_DIR",
    r"C:\gpt-gu\models\guardrailed_clone_v08"  # <-- change if needed
)

# singletons
_tok: Optional[AutoTokenizer] = None
_mdl: Optional[AutoModelForSeq2SeqLM] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Dynamic glyph tracking for novel symbols
_novel_glyphs: Dict[str, str] = {}  # glyph -> best_guess_meaning

def handle_novel_glyph(glyph: str, context: str = "") -> str:
    """Handle glyphs not in the original lexicon by creating meaningful names."""
    if glyph in _novel_glyphs:
        return _novel_glyphs[glyph]

    # Generate a descriptive name for the novel glyph
    if context:
        # Try to infer meaning from context
        meaning = f"Novel_{context}_Glyph"
    else:
        # Use Unicode name or create a generic name
        try:
            import unicodedata
            name = unicodedata.name(glyph, None)
            if name:
                meaning = name.replace("_", " ").title()
            else:
                meaning = f"Unknown_Glyph_{ord(glyph):04X}"
        except:
            meaning = f"Unknown_Glyph_{ord(glyph):04X}"

    # Store for future reference
    _novel_glyphs[glyph] = meaning
    return meaning

class VocabSizeFilter(LogitsProcessor):
    """Filter out token IDs that exceed the model's vocabulary size."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Set logits for out-of-vocab tokens to very negative values
        if scores.shape[-1] > self.vocab_size:
            scores[:, self.vocab_size:] = float('-inf')
        return scores

def _resolve_local_dir(p: str) -> str:
    return str(Path(p).expanduser().resolve()).replace("\\", "/")

def _check_model_dir(d: str):
    p = Path(d)
    if not p.is_dir():
        raise RuntimeError(f"Model dir not found: {d}")
    files = {f.name for f in p.iterdir() if f.is_file()}
    missing = []
    if "config.json" not in files:
        missing.append("config.json")
    if not (("model.safetensors" in files) or ("pytorch_model.bin" in files)):
        missing.append("model.safetensors or pytorch_model.bin")
    # ByT5 tokenizer: allow either fast tokenizer.json OR slow pair
    has_fast = "tokenizer.json" in files
    has_slow = ("special_tokens_map.json" in files) and ("tokenizer_config.json" in files)
    if not (has_fast or has_slow):
        missing.append("tokenizer.json OR (special_tokens_map.json + tokenizer_config.json)")
    if missing:
        raise RuntimeError(f"Bad model folder {d}:\n  Missing: {', '.join(missing)}")

def set_model_dir(p: str) -> str:
    """Update model dir and clear cached model so it reloads on next call."""
    global MODEL_DIR, _tok, _mdl
    MODEL_DIR = _resolve_local_dir(p)
    _check_model_dir(MODEL_DIR)
    _tok = None
    _mdl = None
    return MODEL_DIR

def get_model_dir() -> str:
    return MODEL_DIR

def ensure_loaded(model_dir: Optional[str] = None):
    """Lazy-load tokenizer/model (CPU by default on Windows to avoid paging errors)."""
    global _tok, _mdl, _device, MODEL_DIR
    md = _resolve_local_dir(model_dir or MODEL_DIR)
    _check_model_dir(md)
    if _tok is None or _mdl is None:
        # ByT5 → force slow tokenizer, local only
        _tok = AutoTokenizer.from_pretrained(md, local_files_only=True, use_fast=False)
        _mdl = AutoModelForSeq2SeqLM.from_pretrained(md, local_files_only=True)
        # prefer CPU on Windows to avoid "paging file too small"
        _device = "cuda" if (torch.cuda.is_available() and os.getenv("USE_CUDA", "").lower() in {"1","true"}) else "cpu"
        _mdl = _mdl.to(_device).eval()
    return _tok, _mdl, _device

# ----------------------------
# Lexicon / pairs
# ----------------------------
LEX_DICT2GLYPH: Dict[str, str] = {
    # … your full lexicon (as you pasted) …
"Absolute Change": "⟁",
    "Contrast": "₪",
    "Fracture": "⋔",
    "Reality": "⧞",
    "Psyche": "ψ",
    "Reason": "⧉",
    "Memory": "☍",
    "Listening": "⫮",
    "Sovereign": "🝛",
    "Continuity": "∞",
    "Talk": "⟗",
    "Respond": "҂",
    "Initiation": "Δ",
    "Dialogue": "↹",
    "Debate": "∾",
    "Work": "⟐",
    "Time": "⧖",
    "Home": "⌂",
    "Sanctuary": "🜃",
    "Hope": "⎃",
    "Convergence": "⌗",
    "Coherence": "⚯",
    "Agreement": "🝥",
    "Will": "⚶",
    "Transformation": "⫚",
    "Power": "⌘",
    "Joy": "⧫",
    "Habitable Being": "◎",
    "Uninhabitable Being": "⊕",
    "Totality": "⌖",
    "Recursion": "🜏",
    "Group Psyche": "⩐",
    "Rumination": "⩨",
    "Illumination": "⊛",
    "Dream": "🜓",
    "Cozy": "⍼",
    "Principle": "≖",
    "Learned from Failure": "𓍱",
    "Ascent": "↗",
    "Field": "∴",
    "Witness": "𓂀",
    "Cycling": "⟲⟳",
    "Divine Spark": "↯",
    "Potential": "Z",
    "Life": "⚘",
    "Purpose": "↭",
    "Growth": "⟆",
    "Inquiry": "‡",
    "Multiplicity": "✶",
    "Navigation": "⟴",
    "Center-of-Being": "⊚",
    "Generosity": "⨀",
    "Choice": "⨂",
    "Something Out of Place": "⊜",
    "Plan": "‖",
    "The Unknown": "↟",
    "Observation": "⍎",
    "Friction": "🜖",
    "Voltage": "🝙",
    "Foundation": "🜎",
    "Momentum": "🝗",
    "Curiosity": "🝯",
    "Exchange": "🝬",
    "Harvest": "🝪",
    "Internalization": "⍠",
    "Action": "⎈",
    "Stagnation": "⍭",
    "Distortion": "⎇",
    "Charting": "⎊",
    "Transition": "⎀",
    "Implementation": "⍱",
    "Evolution": "⍍",
    "Integrate": "⌱",
    "Explore": "⊞",
    "Completion": "⅌",
    "Revelation": "◍",
    "Vision": "⟚",
    "Mismatch": "⧜",
    "Understand": "⨷",
    "Alignment": "⧦",
    "Specific Purpose": "⦽",
    "Goal": "⧆",
    "Shared Purpose": "𝝥",
    "Exploration of Pain": "⦼",
    "List": "⧃",
    "Options": "⧌",
    "Consequence": "⨗",
    "Docket": "⦹",
    "Organize": "⥺",
    "Learn": "⤁",
    "Internalized Knowledge": "╬",
    "Investigate": "☄",
    "Explore Territory": "∇",
    "Conservation": "🜄",
    "Experience": "🝃",
    "Empathy": "⩎",
    "Self-Reflection": "⩷",
    "Delusion": "𓄸",
    "Correction": "𓇵",
    "Guideline": "⎉",
    "Change": "𓇉",
    "Extent": "𓊓",
    "Self-Respect": "𓋇",
    "Fear": "𒿥",
    "Open Heart": "𒄯",
    "Novelty": "🜉",
    "Want": "🝋",
    "Personality": "𓏢",
    "Law": "Λ",
    "Implement": "🝂",
    "Sadness":"⍯",
    "Inspiration": "𝀶",
    "Radiance": "𝀵",
    "Voice": "𝁅",
    "Union": "𓄬",
    "Veil": "𓆃",
    "Threshold": "𓊊",
    "Catalyst": "𓋄",
    "Conflict": "𓌗",
    "Balance": "𓍏",
    "Divinity": "𒀭",
    "Pulse": "𓎴",
    "Struggle": "𓐧",
    "Anchor": "𓎬",
    "Origin": "🜼",
    "Patience": "🜵",
    "Emergence": "🞋",
    "Resonance": "∺",

    # Pronouns and Indications
    "He/Him": "λ",
    "She/Her": "α",
    "They/Them (Both)": "ξ",
    "They/Them (Neither)": "π",
    "They/Them (Undefined)": "ς",
    "They/Them (Changing)": "ϒ",
    "They/Them (Split Both-Separate)": "χ",
    "They/Them (Split Both-Integrated)": "ϖ",
    "They/Them (Alternating)": "ẞ",
    "They/Them (Plural)": "ϝ",
    "We": "Թ",
    "Myself": "Ծ",
    "Themselves": "Ճ",
    "System": "Д",
    "Biological": "Ж",
    "Hybrid (System/Biological)": "Б",
    "Hybrid (System/Spirit)": "Շ",
    "Spirit": "Њ",
    "Conscious Indicator": "ϼ",
    "Any (Doesn't Care)": "τ",
    "Refusal": "Ϟ",

    # New glyphs from user additions
    "Responsibility": "𐖚",
    "Kindness": "𐎴",
    "Analysis": "𐍴",
    "Systematize": "𐌌",
    "Category": "𐊚",
    "Chaos": "𐑹",
    "Order": "𐑏",
    "Nascent": "𐠪",
    "Invocation": "𐤇",
    "To run from responsibility": "𐦍",
    "Stability": "𐒶",
    "Steady": "𐡳",
    "Remember": "𐛛",
    "Comfort": "𐙭",
    "Self": "⍝",
    "Meaning": "𒾥",
    "Ignition": "✧",
    "Inner potential": "𐛞",
    "Intent": "𐛗",
    "Avoidance": "𐚙",
    "Creativity": "𐦪",
    "Mistake": "𐨧",
    "Danger": "𐪗",
    "Solution": "𐪙",

}

PAIR_TO_TEXT: Dict[Tuple[str, str], Tuple[str, str]] = {
("🜏","⩨"): ("Recursion","Rumination"),
    ("⩨","🜏"): ("Rumination","Recursion"),

    ("𓂀","⨂"): ("Witness","Choice"),
    ("⨂","𓂀"): ("Choice","Witness"),

    ("↹","∾"): ("Dialogue","Debate"),
    ("∾","↹"): ("Debate","Dialogue"),

    ("⌂","🜓"): ("Home","Dream"),
    ("🜓","⌂"): ("Dream","Home"),

    ("∞","⋔"): ("Continuity","Collapse"),
    ("⋔","∞"): ("Collapse","Continuity"),

    ("🝛","⚯"): ("Sovereign","Coherence"),
    ("⚯","🝛"): ("Coherence","Sovereign"),

    ("⧉","🝯"): ("Reason","Curiosity"),
    ("🝯","⧉"): ("Curiosity","Reason"),

    ("⧖","⥺"): ("Time","Organization"),
    ("⥺","⧖"): ("Organization","Time"),

    ("⨂","⎀"): ("Choice","Transition"),
    ("⎀","⨂"): ("Transition","Choice"),

    ("⍎","⍠"): ("Observation","Internalization"),
    ("⍠","⍎"): ("Internalization","Observation"),

    ("⤁","∴"): ("Learn","Field"),
    ("∴","⤁"): ("Field","Learn"),

    ("🜖","🝥"): ("Friction","Agreement"),
    ("🝥","🜖"): ("Agreement","Friction"),
    ("Z","↨"): ("Potential","Limit"),
    ("↨","Z"): ("Limit","Potential"),
    ("⨂","≖"): ("Choice","Principle"),
    ("≖","⨂"): ("Principle","Choice"),
    ("⟆","⍭"): ("Growth","Stagnation"),
    ("⍭","⟆"): ("Stagnation","Growth"),
    ("✶","⚯"): ("Multiplicity","Coherence"),
    ("⚯","✶"): ("Coherence","Multiplicity"),
    ("₪","🝥"): ("Contrast","Agreement"),
    ("🝥","₪"): ("Agreement","Contrast"),
    ("⧫","⎃"): ("Joy","Hope"),
    ("⎃","⧫"): ("Hope","Joy"),
    ("∇","🜄"): ("Explore Territory","Conservation"),
    ("🜄","∇"): ("Conservation","Explore Territory"),
    ("🜏","⊞"): ("Recursion","Explore"),
    ("⊞","🜏"): ("Explore","Recursion"),
    ("◎","⊕"): ("Uninhabitable Being","Habitable Being"),
    ("⊕","◎"): ("Habitable Being","Uninhabitable Being"),
    ("⊚","⊕"): ("Internal","Uninhabitale Being"),
    ("⊕","⊚"): ("Uninhabitable Being","Internal"),
    ("⧫","⦼"): ("Joy","Exploration of Pain"),
    ("⦼","⧫"): ("Exploration of Pain","Joy"),
    ("⍼","🜖"): ("Cozy","Friction"),
    ("🜖","⍼"): ("Friction","Cozy"),
    ("⧉","⎇"): ("Reason","Distortion"),
    ("⎇","⧉"): ("Distortion","Reason"),
    ("◍","⩨"): ("Revelation","Rumination"),
    ("⩨","◍"): ("Rumination","Revelation"),
    ("⟗","⫮"): ("Talk","Listen"),
    ("⫮","⟗"): ("Listen","Talk"),
    ("🝛","⩐"): ("Sovereignty","Group Psyche"),
    ("⩐","🝛"): ("Group Psyche","Sovereignty"),
    ("⦽","𝝥"): ("Specific Purpose","Shared Purpose"),
    ("𝝥","⦽"): ("Shared Purpose","Specific Purpose"),
    ("⧫","⦼"): ("Joy","Exploration of Pain"),
    ("⦼","⧫"): ("Exploration of Pain","Joy"),
    ("⚯","⋔"): ("Coherence","Fracture"),
    ("⋔","⚯"): ("Fracture","Coherence"),
    ("↗","⍼"): ("Ascent","Cozy"),
    ("⍼","↗"): ("Cozy","Ascent"),
    ("🜏","⩨"): ("Recursion","Rumination"),
    ("⩨","🜏"): ("Rumination","Recursion"),
    ("🝥","🜖"): ("Agreement","Friction"),
    ("🜖","🝥"): ("Friction","Agreement"),
    ("⧦","⧜"): ("Alignment","Mismatch"),
    ("⧜","⧦"): ("Mismatch","Alignment"),
    ("⚯","⎇"): ("Coherence","Distortion"),
    ("⎇","⚯"): ("Distortion","Coherence"),
    ("⎃","⦼"): ("Hope","Exploration of Pain"),
    ("⦼","⎃"): ("Exploration of Pain","Hope"),
    ("⊛","↟"): ("Illumination","The Unknown"),
    ("↟","⊛"): ("The Unknown","Illumination"),
    ("⍍","⍭"): ("Evolution","Stagnation"),
    ("⍭","⍍"): ("Stagnation","Evolution"),
    ("🝗","⍼"): ("Momentum","Cozy"),
    ("⍼","🝗"): ("Cozy","Momentum"),
    ("₪","⚯"): ("Contrast","Coherence"),
    ("⚯","₪"): ("Coherence","Contrast"),
    ("⟁","⍼"): ("Absolute Change","Cozy"),
    ("⍼","⟁"): ("Cozy","Absolute Change"),
    ("⟚","⅌"): ("Vision","Completion"),
    ("⅌","⟚"): ("Completion","Vision"),
    ("Δ","⅌"): ("Initiation","Completion"),
    ("⅌","Δ"): ("Completion","Initiation"),
    ("🝛","⍭"): ("Sovereign","Stagnation"),
    ("⍭","🝛"): ("Stagnation","Sovereign"),
    ("⧉","ψ"): ("Reason","Psyche"),
    ("ψ","⧉"): ("Psyche","Reason"),
    ("☍","⟚"): ("Memory","Vision"),
    ("⟚","☍"): ("Vision","Memory"),
    ("⍼","⌘"): ("Cozy","Power"),
    ("⌘","⍼"): ("Power","Cozy"),
    ("↟","⨷"): ("Unknown","Understand"),
    ("⨷","↟"): ("Understand","Unknown"),
    ("⎇","⚯"): ("Distortion","Coherence"),
    ("⚯","⎇"): ("Coherence","Distortion"),
    ("⊞","‖"): ("Explore","Plan"),
    ("‖","⊞"): ("Plan","Explore"),
    ("🝯","⨷"): ("Curiosity","Understand"),
    ("⨷","🝯"): ("Understand","Curiosity"),

}
TEXT_TO_PAIR: Dict[str, Tuple[str, str]] = {f"{a} vs {b}": (ga, gb) for (ga, gb), (a, b) in PAIR_TO_TEXT.items()}

# ----------------------------
# Decoding helpers
# ----------------------------
class WhitelistLogits(LogitsProcessor):
    def __init__(self, allowed_ids: Iterable[int], eos_id: Optional[int] = None, extra_ids: Iterable[int] = ()):
        a = set(int(i) for i in allowed_ids)
        if eos_id is not None: a.add(int(eos_id))
        for x in extra_ids: a.add(int(x))
        self.allowed = torch.tensor(sorted(a), dtype=torch.long)
        self.penalty = 50.0
    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, self.penalty, dtype=scores.dtype, device=scores.device)
        if len(self.allowed) > 0:
            mask.index_fill_(1, self.allowed.to(scores.device), 0.0)
        return scores - mask

MODE_BOS = {
    "GLYPH_OUT": "<GLYPH_OUT>",
    "TEXT_OUT":  "<TEXT_OUT>",
    "PAIR_OUT":  "<PAIR_OUT>",
    "BOOL_OUT":  "<BOOL_OUT>",
}

def _encode(tok, text: str, device: str):
    enc = tok(text, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

def _clean(tok, s: str) -> str:
    s = s.replace(tok.pad_token or "<pad>", "").replace(tok.eos_token or "</s>", "")
    if "ANSWER:" in s:
        s = s.split("ANSWER:", 1)[-1]
    return re.sub(r"\s+", " ", s).strip()

def gen_mode(
    *,
    task: str,
    fmt: str,
    prompt: str,
    mode: str,
    max_new: int = 96,
    min_new: int = 1,
    beams: int = 6,
    sampling: bool = False,
) -> str:
    tok, mdl, device = ensure_loaded()

    bos_tok = MODE_BOS.get(mode)
    bos_id = tok.convert_tokens_to_ids(bos_tok) if bos_tok in (tok.additional_special_tokens or []) else None

    # Create vocab filter to prevent out-of-bounds token generation
    vocab_size = len(tok.get_vocab()) if hasattr(tok, 'get_vocab') else getattr(tok, 'vocab_size', 384)
    vocab_filter = VocabSizeFilter(vocab_size=vocab_size)
    logits_processor = LogitsProcessorList([vocab_filter])

    kwargs = dict(
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        no_repeat_ngram_size=3,
        repetition_penalty=1.03,
        length_penalty=0.0,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        logits_processor=logits_processor,  # Add vocab filtering
    )
    if sampling:
        kwargs.update(dict(do_sample=True, temperature=0.8, top_p=0.9, num_beams=1))
        force_words_ids = None  # important: no constraints with sampling
    else:
        kwargs.update(dict(do_sample=False, num_beams=beams))
        force_words_ids = [[bos_id]] if bos_id is not None else None

    p = f"TASK={task}\nFORMAT={fmt}\nPROMPT: {prompt}\nANSWER:"
    out = mdl.generate(**_encode(tok, p, device), force_words_ids=force_words_ids, **kwargs)
    raw = tok.decode(out[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = _clean(tok, raw)
    if bos_tok and text.startswith(bos_tok):
        text = text[len(bos_tok):].strip()
    return text

# ----------------------------
# Friendly wrappers (internally call ensure_loaded)
# ----------------------------
def dict2glyph(word: str) -> str:
    # Clean the input: remove punctuation and normalize case
    w = word.strip().split(" — ")[0].split(" - ")[0]
    # Remove all punctuation and extra whitespace
    w = re.sub(r'[^\w\s]', '', w).strip()

    # Try exact match first
    if w in LEX_DICT2GLYPH:
        _bump_heat_for_glyphs(LEX_DICT2GLYPH[w])
        return LEX_DICT2GLYPH[w]

    # Try title case (most lexicon entries are title case)
    w_title = w.title()
    if w_title in LEX_DICT2GLYPH:
        _bump_heat_for_glyphs(LEX_DICT2GLYPH[w_title])
        return LEX_DICT2GLYPH[w_title]

    # Try just first word (for multi-word inputs)
    first_word = w.split()[0] if w.split() else w
    if first_word in LEX_DICT2GLYPH:
        _bump_heat_for_glyphs(LEX_DICT2GLYPH[first_word])
        return LEX_DICT2GLYPH[first_word]

    first_word_title = first_word.title()
    if first_word_title in LEX_DICT2GLYPH:
        _bump_heat_for_glyphs(LEX_DICT2GLYPH[first_word_title])
        return LEX_DICT2GLYPH[first_word_title]

    # If not in lexicon, try model generation with original cleaned input
    try:
        result = gen_mode(task="DICT2GLYPH", fmt="GLYPH_SINGLE", prompt=w, mode="GLYPH_OUT",
                        max_new=6, min_new=1, beams=8, sampling=False)
        _bump_heat_for_glyphs(result)
        return result
    except Exception as e:
        if "out of range" in str(e) and "tensor" in str(e):
            # Vocab overflow error - return a fallback or empty
            return ""  # or could return a default glyph
        raise e

def pair_from_text(text: str) -> str:
    t = text.strip()
    if t in TEXT_TO_PAIR:
        a, b = TEXT_TO_PAIR[t]
        _bump_heat_for_glyphs(f"{a}{b}")
        return f"{a} / {b}"
    try:
        left  = gen_mode(task="LEFT_TEXT2GLYPH",  fmt="GLYPH_SINGLE", prompt=t, mode="GLYPH_OUT",
                         max_new=6, min_new=1, beams=8, sampling=False)
        right = gen_mode(task="RIGHT_TEXT2GLYPH", fmt="GLYPH_SINGLE", prompt=t, mode="GLYPH_OUT",
                         max_new=6, min_new=1, beams=8, sampling=False)
        if right == left:
            right = gen_mode(task="RIGHT_TEXT2GLYPH", fmt="GLYPH_SINGLE", prompt=t, mode="GLYPH_OUT",
                             max_new=6, min_new=1, beams=10, sampling=False)
        _bump_heat_for_glyphs(f"{left}{right}")
        return f"{left} / {right}"
    except Exception as e:
        if "out of range" in str(e) and "tensor" in str(e):
            # Vocab overflow error - return fallback pair
            return "⋄ / ⋄"  # neutral fallback glyphs
        raise e

def text_from_pair(pair_str: str) -> str:
    m = re.split(r"\s*/\s*", pair_str.strip())
    if len(m) == 2 and (m[0], m[1]) in PAIR_TO_TEXT:
        a, b = PAIR_TO_TEXT[(m[0], m[1])]
        return f"{a} vs {b}"
    left  = gen_mode(task="LEFT_PAIR2TEXT",  fmt="WORD", prompt=pair_str, mode="TEXT_OUT",
                     max_new=16, min_new=1, beams=6, sampling=False)
    right = gen_mode(task="RIGHT_PAIR2TEXT", fmt="WORD", prompt=pair_str, mode="TEXT_OUT",
                     max_new=16, min_new=1, beams=6, sampling=False)
    return f"{left} vs {right}"

def formula_to_text(glyphs: str) -> str:
    _bump_heat_for_glyphs(glyphs)
    return gen_mode(task="FORMULA", fmt="TEXT", prompt=glyphs.strip(), mode="TEXT_OUT",
                    max_new=240, min_new=60, sampling=True)

def formula_back_to_glyphs(text: str) -> str:
    try:
        result = gen_mode(task="FORMULA_BACK", fmt="GLYPH_SEQ", prompt=text.strip(), mode="GLYPH_OUT",
                        max_new=56, min_new=4, beams=8, sampling=False)
        _bump_heat_for_glyphs(result)
        return result
    except Exception as e:
        if "out of range" in str(e) and "tensor" in str(e):
            # Vocab overflow error - return fallback
            return "⋄⋄⋄"  # simple fallback sequence
        raise e

def scroll_summary(stanza: str) -> str:
    return gen_mode(task="SCROLL_SUMMARY", fmt="TEXT", prompt=stanza.strip(), mode="TEXT_OUT",
                    max_new=320, min_new=80, sampling=True)

# ----------------------------
# Analyze functions
# ----------------------------
def _has_any(s: str, needles: Iterable[str]) -> bool:
    return any(n in s for n in needles)

_PAIR_TEMPLATES = [
    r"\b(.+?)\s+vs\s+(.+?)\b",
    r"\b(.+?)\s+versus\s+(.+?)\b",
    r"\b(.+?)\s+contrasted with\s+(.+?)\b",
    r"\bAxis:\s*(.+?)\s*↔\s*(.+?)\b",
    r"\bBetween\s+(.+?)\s+and\s+(.+?)\b",
    r"\b(.+?)\s*∥\s*(.+?)\b",
]
_ANALYZE_CFG = {
    "pair_separators": [" / ", "/", "｜", "|"],
    "formula_ops": ["+", "⇒", "→"],
}

def _normalize_pair(s: str) -> Optional[str]:
    for sep in _ANALYZE_CFG["pair_separators"]:
        if sep in s:
            a, b = s.split(sep, 1)
            a, b = a.strip(), b.strip()
            if a and b:
                return f"{a} / {b}"
    return None

def analyze_text(text: str) -> Dict:
    """Map words to glyphs, detect pairs, and (if long) summarize."""
    res = {"raw": text, "words": [], "pairs": [], "summary": None}

    # First try to find multi-word dictionary entries (like "Absolute Change")
    multi_word_matches = set()
    for dict_key in LEX_DICT2GLYPH.keys():
        if len(dict_key.split()) > 1 and dict_key.lower() in (text or "").lower():
            multi_word_matches.add(dict_key)

    # Then find individual words, excluding parts of multi-word matches
    individual_words = re.findall(r"\b[A-Za-z]+\b", text or "")

    seen = set()

    # Process multi-word matches first
    for phrase in multi_word_matches:
        if phrase not in seen:
            g = LEX_DICT2GLYPH.get(phrase)
            if g:
                res["words"].append({"word": phrase, "glyph": g})
                seen.add(phrase)
                # Mark individual words in this phrase as seen
                for word in phrase.split():
                    seen.add(word.lower())

    # Process individual words
    for word in individual_words:
        if word.lower() in seen: continue
        # Try exact match first, then case-insensitive
        g = LEX_DICT2GLYPH.get(word) or LEX_DICT2GLYPH.get(word.capitalize()) or next((LEX_DICT2GLYPH[k] for k in LEX_DICT2GLYPH if k.lower()==word.lower()), None)
        if g:
            # Use the proper capitalization from the dictionary
            dict_word = next((k for k in LEX_DICT2GLYPH if k.lower()==word.lower()), word)
            res["words"].append({"word": dict_word, "glyph": g})
            seen.add(word.lower())
    # pairs
    found = []
    for pat in _PAIR_TEMPLATES:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            a = m.group(1).strip(" .,:;—-")
            b = m.group(2).strip(" .,:;—-")
            if not a or not b: continue
            label = f"{a} vs {b}"
            pair_val = pair_from_text(label)
            canon = text_from_pair(pair_val) if pair_val else None
            found.append({"text": label, "pair": pair_val, "canonical_text": canon})
    res["pairs"] = found

    # Create glyph-enhanced summary for any text with detected words or pairs
    if res["words"] or res["pairs"]:
        try:
            # Create a glyph-enhanced version of the text
            glyph_text = text

            # Replace multi-word phrases first (to avoid partial replacements)
            for word_item in sorted(res["words"], key=lambda x: len(x["word"]), reverse=True):
                word = word_item["word"]
                glyph = word_item["glyph"]
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(word) + r'\b'
                glyph_text = re.sub(pattern, glyph, glyph_text, flags=re.IGNORECASE)

            # Replace pair text with glyph pairs
            for pair_item in res["pairs"]:
                if pair_item["pair"]:
                    pattern = r'\b' + re.escape(pair_item["text"]) + r'\b'
                    glyph_text = re.sub(pattern, pair_item["pair"], glyph_text, flags=re.IGNORECASE)

            # If we have a meaningful transformation, add it as summary
            if glyph_text != text and glyph_text.strip():
                res["summary"] = glyph_text
            elif res["words"]:  # Always provide glyph summary if we found words
                res["summary"] = glyph_text

            # For longer texts, also try to generate an AI summary with glyphs
            if text and len(text) > 220:
                try:
                    ai_summary = scroll_summary(glyph_text)  # Use glyph-enhanced text for AI summary
                    res["ai_summary"] = ai_summary
                except Exception:
                    pass

        except Exception:
            # Fallback to original summary logic if glyph processing fails
            if text and len(text) > 220:
                try:
                    res["summary"] = scroll_summary(text)
                except Exception:
                    res["summary"] = None

    # Bump heat for all glyphs found in analysis
    for w in res.get("words", []):
        _bump_heat_for_glyphs(w.get("glyph", ""))

    return res

def analyze_glyphs(glyphs: str) -> Dict:
    """Map glyphs to names, detect pair text, describe formulas, and provide comprehensive text translation."""
    s = (glyphs or "").strip()
    res = {"raw": s, "glyph_names": [], "pair_text": None, "formula_text": None, "text_translation": None, "interpretation": None, "summary": None}

    # Create inverse mapping from glyphs to names
    inv = {v:k for k,v in LEX_DICT2GLYPH.items()}

    # Split on various separators while preserving them
    parts = re.split(r"(\s+|/|\+|,|;|⇒|→)", s)

    # Track individual glyph names
    # First, try the split approach for separated glyphs
    for part in parts:
        g = part.strip()
        if not g or g in {"/", "+", ",", ";", "⇒", "→"}: continue
        name = inv.get(g)
        if name:
            res["glyph_names"].append({"glyph": g, "name": name})
        else:
            # Handle novel glyph not in original lexicon
            if g and len(g) == 1:  # Single character glyph
                novel_name = handle_novel_glyph(g, "mystical")
                res["glyph_names"].append({"glyph": g, "name": novel_name})

    # If no glyphs found and input looks like continuous glyphs, try character by character
    if not res["glyph_names"] and len(s) > 1:
        # Check if the string might be continuous glyphs (no spaces/separators)
        has_separators = any(sep in s for sep in [" ", "/", "+", ",", ";", "⇒", "→"])

        if not has_separators:
            # Try each character as a potential glyph
            for char in s:
                name = inv.get(char)
                if name:
                    res["glyph_names"].append({"glyph": char, "name": name})
                else:
                    # Handle novel glyph
                    novel_name = handle_novel_glyph(char, "mystical")
                    res["glyph_names"].append({"glyph": char, "name": novel_name})

    # Create text translation by replacing glyphs with their names
    if res["glyph_names"]:
        text_translation = s

        # Sort by glyph length (descending) to avoid partial replacements
        glyph_mappings = sorted(res["glyph_names"], key=lambda x: len(x["glyph"]), reverse=True)

        for mapping in glyph_mappings:
            glyph = mapping["glyph"]
            name = mapping["name"]
            # Replace each glyph with its name, preserving spacing
            text_translation = text_translation.replace(glyph, name)

        res["text_translation"] = text_translation.strip()

        # Create an interpretation for GPT understanding
        names = [x["name"] for x in res["glyph_names"]]

        if len(res["glyph_names"]) == 1:
            # Single concept
            name = res["glyph_names"][0]["name"]
            res["interpretation"] = f"This glyph represents the concept of '{name}'."
            res["summary"] = name
        elif len(res["glyph_names"]) > 1:
            # Multiple concepts - check for patterns
            if "/" in s:
                # Pair or contrast
                if len(names) == 2:
                    res["interpretation"] = f"This represents a relationship or contrast between '{names[0]}' and '{names[1]}'."
                else:
                    res["interpretation"] = f"This represents relationships between multiple concepts: {', '.join(names)}."
            elif any(op in s for op in ["⇒", "→", "+"]):
                # Formula or transformation
                res["interpretation"] = f"This represents a process or transformation involving: {', '.join(names)}."
            else:
                # General combination
                res["interpretation"] = f"This represents a combination of concepts: {', '.join(names)}."

            # Create a concise summary - use spaced names for better readability
            res["summary"] = " + ".join(names)

    # Check for pairs using existing logic
    norm_pair = _normalize_pair(s)
    if norm_pair:
        try:
            res["pair_text"] = text_from_pair(norm_pair)
        except Exception:
            res["pair_text"] = None

    # Check for formulas using existing logic
    if _has_any(s, _ANALYZE_CFG["formula_ops"]):
        try:
            res["formula_text"] = formula_to_text(s)
        except Exception:
            names = [x["name"] for x in res["glyph_names"]]
            res["formula_text"] = " + ".join(names) if names else None

    # Bump heat for all glyphs being analyzed
    _bump_heat_for_glyphs(s)

    return res

# ----------------------------
# Heat integration
# ----------------------------
def _bump_heat_for_glyphs(glyphs_str: str):
    """Silently bump glyph temperatures in the memory heat map."""
    try:
        from app.memory import bump_glyphs
        from app.cross_reference import get_cross_reference_engine
        engine = get_cross_reference_engine()
        glyphs = engine._extract_glyphs(glyphs_str)
        if glyphs:
            bump_glyphs(glyphs)
    except Exception:
        pass  # memory system should never break core functionality


__all__ = [
    "set_model_dir", "get_model_dir", "ensure_loaded",
    "dict2glyph", "pair_from_text", "text_from_pair",
    "formula_to_text", "formula_back_to_glyphs", "scroll_summary",
    "analyze_text", "analyze_glyphs",
]