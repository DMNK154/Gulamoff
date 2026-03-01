import os, sys, json
from app.guardrails import load_model, build_runtime

MODEL_DIR = os.environ.get("GU_MODEL_DIR", os.path.join(os.path.dirname(__file__), "model"))

tok, mdl, _ = load_model(MODEL_DIR)
rt = build_runtime(tok, mdl)

tasks = {
  "dict2glyph": rt["dict2glyph"],
  "pair_from_text": rt["pair_from_text"],
  "text_from_pair": rt["text_from_pair"],
  "formula_to_text": rt["formula_to_text"],
  "formula_back_to_glyphs": rt["formula_back_to_glyphs"],
  "scroll_summary": rt["scroll_summary"],
}

if len(sys.argv) < 3 or sys.argv[1] not in tasks:
    print("Usage: python cli.py <task> <prompt>")
    print("Tasks:", ", ".join(tasks.keys()))
    sys.exit(1)

task = sys.argv[1]
prompt = " ".join(sys.argv[2:])
print(tasks[task](prompt))