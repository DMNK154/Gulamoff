# ui.py
import gradio as gr
import os
import json
from app.guardrails import (
    dict2glyph, pair_from_text, text_from_pair,
    formula_to_text, formula_back_to_glyphs, scroll_summary
)
from app.cross_reference import analyze_glyph
from app.memory import (
    store_memory, store_glyph_memory, recall_memories,
    get_weather, get_raw_heat_map, trigger_decay,
)

def do_dict2glyph(x): return dict2glyph(x)
def do_pair_from_text(x): return pair_from_text(x)
def do_text_from_pair(x): return text_from_pair(x)
def do_formula_to_text(x): return formula_to_text(x)
def do_formula_back(x): return formula_back_to_glyphs(x)
def do_scroll_summary(x): return scroll_summary(x)

def do_store_memory(text):
    """Store a text memory and return formatted result."""
    if not text.strip():
        return "", ""
    try:
        record = store_memory(text)
        formula_display = f"**Formula:** `{record['formula']}`\n\n"
        formula_display += f"**Glyphs:** {' '.join(record['glyphs'])}\n\n"
        formula_display += f"**Store Salience:** {record['store_salience']}\n\n"
        formula_display += f"**ID:** {record['id']}"

        heat_display = "**Heat at storage:**\n"
        if record["heat"]:
            for g, t in sorted(record["heat"].items(), key=lambda x: -x[1])[:10]:
                heat_display += f"- {g} = {t:.2f}\n"
        else:
            heat_display += "*No active heat*\n"

        return formula_display, heat_display
    except Exception as e:
        return f"Error: {str(e)}", ""

def do_store_glyph_memory(formula):
    """Store a glyph formula memory and return formatted result."""
    if not formula.strip():
        return "", ""
    try:
        record = store_glyph_memory(formula)
        formula_display = f"**Formula:** `{record['formula']}`\n\n"
        formula_display += f"**Glyphs:** {' '.join(record['glyphs'])}\n\n"
        formula_display += f"**Store Salience:** {record['store_salience']}\n\n"
        formula_display += f"**ID:** {record['id']}"

        heat_display = "**Heat at storage:**\n"
        if record["heat"]:
            for g, t in sorted(record["heat"].items(), key=lambda x: -x[1])[:10]:
                heat_display += f"- {g} = {t:.2f}\n"
        else:
            heat_display += "*No active heat*\n"

        return formula_display, heat_display
    except Exception as e:
        return f"Error: {str(e)}", ""

def do_recall():
    """Recall thermally-compatible memories."""
    try:
        results = recall_memories(top_n=10)
        if not results:
            return "*No thermally-compatible memories found.*\n\n*The current heat state doesn't match any stored memory signatures.*"

        output = f"## {len(results)} Memories Recalled\n\n"
        for r in results:
            output += f"### {r['id']} — `{r['formula']}`\n"
            output += f"- **Similarity:** {r.get('_similarity', 0):.2f} | "
            output += f"**Score:** {r.get('_recall_score', 0):.2f} | "
            output += f"**Activations:** {r['activation_count']}\n"
            output += f"- **Glyphs:** {' '.join(r['glyphs'])}\n"
            output += f"- **Stored:** {r['timestamp']}\n\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

def do_weather():
    """Get cognitive weather report."""
    try:
        w = get_weather()

        output = "# Cognitive Weather\n\n"
        output += f"**Total Heat:** {w['total_heat']} | "
        output += f"**Memories:** {w['total_memories']} | "
        output += f"**Thermally Active:** {w['active_memories']} | "
        output += f"**Decay Ticks:** {w['tick_count']}\n\n"

        output += "## Hot Glyphs\n"
        if w["hot_glyphs"]:
            for g in w["hot_glyphs"]:
                bar = "█" * int(g["temp"])
                output += f"- **{g['glyph']}** {g['name']} — {g['temp']} {bar}\n"
        else:
            output += "*All glyphs are cold*\n"

        output += "\n## Synchronicities\n"
        if w["synchronicities"]:
            for s in w["synchronicities"]:
                output += f"- **{s['glyph_a']}** ({s['name_a']}) + **{s['glyph_b']}** ({s['name_b']}) "
                output += f"— rarity: {s['rarity_score']} (co-occurrence: {s['co_occurrence']})\n"
        else:
            output += "*No synchronicity events detected*\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"

def do_decay():
    """Trigger decay and return updated heat map."""
    try:
        trigger_decay()
        hm = get_raw_heat_map()
        output = f"**Decay tick applied.** Total heat: {hm['total_heat']} | Tick #{hm['tick_count']}\n\n"
        temps = hm.get("temperatures", {})
        if temps:
            for g, t in sorted(temps.items(), key=lambda x: -x[1])[:15]:
                bar = "█" * int(t)
                output += f"- {g} = {t:.2f} {bar}\n"
        else:
            output += "*All glyphs are cold*\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"

def do_cross_reference(glyph_or_name):
    """Analyze glyph relationships and return formatted results."""
    if not glyph_or_name.strip():
        return "Please enter a glyph or concept name", "", "", "", ""

    try:
        result = analyze_glyph(glyph_or_name.strip())

        # Format complementary glyphs
        complementary_text = "## Complementary Glyphs\n"
        complementary_text += "*Glyphs that frequently appear together in formulas*\n\n"
        if result["complementary"]:
            for item in result["complementary"]:
                complementary_text += f"- **{item['glyph']}** {item['name']} (appears together {item['frequency']}x)\n"
        else:
            complementary_text += "*No complementary glyphs found*\n"

        # Format oppositions
        oppositions_text = "## Oppositional Glyphs\n"
        oppositions_text += "*Glyphs representing contrasting concepts*\n\n"
        if result["oppositions"]:
            for item in result["oppositions"]:
                oppositions_text += f"- **{item['glyph']}** {item['name']}\n"
        else:
            oppositions_text += "*No oppositions found*\n"

        # Format transformations
        transformations_text = "## Transformations\n"
        transformations_text += "*What this glyph transforms into (A → B patterns)*\n\n"
        if result["transformations"]:
            for item in result["transformations"]:
                transformations_text += f"- **{result['glyph']}** → **{item['glyph']}** ({result['name']} → {item['name']})\n"
        else:
            transformations_text += "*No transformation patterns found*\n"

        # Format formula examples
        formulas_text = "## Formula Examples\n"
        formulas_text += "*Scroll formulas containing this glyph*\n\n"
        if result["formula_examples"]:
            for i, formula in enumerate(result["formula_examples"], 1):
                formulas_text += f"{i}. `{formula}`\n"
        else:
            formulas_text += "*No formula examples found*\n"

        # Format contexts
        contexts_text = "## Scroll Contexts\n"
        contexts_text += "*Wisdom from the scrolls*\n\n"
        if result["contexts"]:
            for i, context in enumerate(result["contexts"], 1):
                contexts_text += f"{i}. {context}\n\n"
        else:
            contexts_text += "*No scroll contexts found*\n"

        # Summary at top
        summary = f"# {result['glyph']} — {result['name']}\n\n"
        summary += f"**Complementary glyphs:** {len(result['complementary'])} | "
        summary += f"**Oppositions:** {len(result['oppositions'])} | "
        summary += f"**Transformations:** {len(result['transformations'])}\n"

        return summary, complementary_text, oppositions_text, transformations_text, formulas_text + "\n" + contexts_text

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""

with gr.Blocks() as demo:
    gr.Markdown("# GPT GU: Initiate of Flame")
    with gr.Tab("DICT2GLYPH"):
        inp = gr.Textbox(label="Word")
        out = gr.Textbox(label="Glyph")
        btn = gr.Button("Run")
        btn.click(do_dict2glyph, inp, out)

    with gr.Tab("PAIR"):
        tinp = gr.Textbox(label="Text (e.g., Potential vs Limit)")
        pout = gr.Textbox(label="Glyph Pair (G / H)")
        btn2 = gr.Button("Run")
        btn2.click(do_pair_from_text, tinp, pout)

    with gr.Tab("PAIR→TEXT"):
        pin = gr.Textbox(label="Glyph Pair (e.g., Z / ↨)")
        tout = gr.Textbox(label="Text (A vs B)")
        btn3 = gr.Button("Run")
        btn3.click(do_text_from_pair, pin, tout)

    with gr.Tab("FORMULA"):
        fin = gr.Textbox(label="Glyph Formula")
        fout = gr.Textbox(label="Text")
        gr.Button("Run").click(do_formula_to_text, fin, fout)

    with gr.Tab("FORMULA_BACK"):
        b_in = gr.Textbox(label="Caption/Text")
        b_out = gr.Textbox(label="Glyph Sequence")
        gr.Button("Run").click(do_formula_back, b_in, b_out)

    with gr.Tab("SCROLL SUMMARY"):
        sin = gr.Textbox(label="Stanza", lines=8)
        sout = gr.Textbox(label="Summary", lines=8)
        gr.Button("Summarize").click(do_scroll_summary, sin, sout)

    with gr.Tab("CROSS REFERENCE"):
        gr.Markdown("### Explore Glyph Relationships")
        gr.Markdown("Discover complementary glyphs, oppositions, transformations, and scroll wisdom")

        with gr.Row():
            xref_input = gr.Textbox(
                label="Enter Glyph or Concept Name",
                placeholder="e.g., Joy, ⧫, Potential, Z",
                scale=3
            )
            xref_btn = gr.Button("Analyze", scale=1, variant="primary")

        xref_summary = gr.Markdown(label="Summary")

        with gr.Row():
            with gr.Column():
                xref_complementary = gr.Markdown(label="Complementary")
            with gr.Column():
                xref_oppositions = gr.Markdown(label="Oppositions")

        with gr.Row():
            with gr.Column():
                xref_transformations = gr.Markdown(label="Transformations")
            with gr.Column():
                xref_examples = gr.Markdown(label="Examples & Context")

        xref_btn.click(
            do_cross_reference,
            inputs=xref_input,
            outputs=[xref_summary, xref_complementary, xref_oppositions, xref_transformations, xref_examples]
        )

        # Add some example buttons
        gr.Markdown("### Quick Examples")
        with gr.Row():
            ex1 = gr.Button("Joy", size="sm")
            ex2 = gr.Button("Potential", size="sm")
            ex3 = gr.Button("Recursion", size="sm")
            ex4 = gr.Button("Power", size="sm")

        ex1.click(lambda: "Joy", outputs=xref_input)
        ex2.click(lambda: "Potential", outputs=xref_input)
        ex3.click(lambda: "Recursion", outputs=xref_input)
        ex4.click(lambda: "Power", outputs=xref_input)

    with gr.Tab("MEMORY"):
        gr.Markdown("### Glyph Memory System")
        gr.Markdown("Heat-gated temporal memory. The system thinks in glyphs.")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### Store Memory")
                mem_text_in = gr.Textbox(
                    label="Text (will be converted to glyph formula)",
                    placeholder="e.g., compression and recursion in memory systems",
                    lines=2
                )
                mem_store_btn = gr.Button("Store from Text", variant="primary")
                gr.Markdown("---")
                mem_formula_in = gr.Textbox(
                    label="Glyph Formula (store directly)",
                    placeholder="e.g., ⧞◬⟲⧉",
                )
                mem_glyph_store_btn = gr.Button("Store Formula")

            with gr.Column(scale=2):
                mem_record = gr.Markdown(label="Stored Record")
                mem_heat_snap = gr.Markdown(label="Heat Snapshot")

        mem_store_btn.click(do_store_memory, inputs=mem_text_in, outputs=[mem_record, mem_heat_snap])
        mem_glyph_store_btn.click(do_store_glyph_memory, inputs=mem_formula_in, outputs=[mem_record, mem_heat_snap])

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Recall")
                gr.Markdown("*Surfaces memories whose stored heat matches the current cognitive state*")
                mem_recall_btn = gr.Button("Recall", variant="primary")
                mem_recall_out = gr.Markdown(label="Recalled Memories")
                mem_recall_btn.click(fn=do_recall, inputs=None, outputs=mem_recall_out)

            with gr.Column():
                gr.Markdown("#### Cognitive Weather")
                mem_weather_btn = gr.Button("Weather Report", variant="secondary")
                mem_weather_out = gr.Markdown(label="Weather")
                mem_weather_btn.click(fn=do_weather, inputs=None, outputs=mem_weather_out)

                gr.Markdown("---")
                mem_decay_btn = gr.Button("Trigger Decay", size="sm")
                mem_decay_out = gr.Markdown(label="Heat Map")
                mem_decay_btn.click(fn=do_decay, inputs=None, outputs=mem_decay_out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=3001)