# app/gradio_ui.py
"""
Gradio UI builder for GPT-GU.
Can be mounted inside FastAPI (server.py) or run standalone (ui.py).
"""
import gradio as gr

from app.guardrails import (
    dict2glyph, pair_from_text, text_from_pair,
    formula_to_text, formula_back_to_glyphs, scroll_summary
)
from app.cross_reference import analyze_glyph
from app.memory import (
    store_memory, store_glyph_memory, recall_memories,
    get_weather, get_raw_heat_map, trigger_decay,
    get_phase_space_coordinates, get_synchronicity_events,
    get_resonances, get_void_profile,
    trigger_meta_analysis, get_meta_patterns,
    get_emergent_links, get_thresholds,
)


def do_dict2glyph(x): return dict2glyph(x)
def do_pair_from_text(x): return pair_from_text(x)
def do_text_from_pair(x): return text_from_pair(x)
def do_formula_to_text(x): return formula_to_text(x)
def do_formula_back(x): return formula_back_to_glyphs(x)
def do_scroll_summary(x): return scroll_summary(x)


def do_store_memory(text):
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
            output += f"- **Salience:** model={r.get('salience_model', '?'):.2f} "
            output += f"usage={r.get('salience_usage', '?'):.2f} "
            output += f"composite={r.get('salience_composite', '?'):.2f}\n"
            output += f"- **Glyphs:** {' '.join(r['glyphs'])}\n"
            output += f"- **Stored:** {r['timestamp']}\n\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def do_weather():
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
                bar = "\u2588" * int(g["temp"])
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

        # L2: Trending glyphs
        output += "\n## Trending Glyphs\n"
        trending = w.get("trending_glyphs", [])
        if trending:
            for g in trending:
                vel = g.get("velocity", 0)
                arrow = "\u2191" if vel > 0.1 else "\u2193" if vel < -0.1 else "\u2194"
                output += (f"- **{g['glyph']}** "
                           f"temp: {g.get('temperature', 0):.2f} "
                           f"level: {g.get('level', 0):.2f} "
                           f"vel: {arrow}{abs(vel):.2f} "
                           f"({g.get('event_count', 0)} events)\n")
        else:
            output += "*No trend data yet*\n"

        # L2: Retrieval stats
        stats = w.get("retrieval_stats", {})
        if stats and stats.get("total_retrievals", 0) > 0:
            output += "\n## Retrieval Stats\n"
            output += f"- **Total retrievals:** {stats['total_retrievals']}\n"
            output += f"- **Avg results:** {stats.get('avg_results', 0):.1f}\n"
            output += f"- **Avg score:** {stats.get('avg_score', 0):.4f}\n"
            most = stats.get("most_retrieved", [])
            if most:
                output += "- **Most retrieved:** "
                output += ", ".join(f"{m['glyph']}({m['count']})" for m in most)
                output += "\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


def do_decay():
    try:
        trigger_decay()
        hm = get_raw_heat_map()
        output = f"**Decay tick applied.** Total heat: {hm['total_heat']} | Tick #{hm['tick_count']}\n\n"
        temps = hm.get("temperatures", {})
        if temps:
            for g, t in sorted(temps.items(), key=lambda x: -x[1])[:15]:
                bar = "\u2588" * int(t)
                output += f"- {g} = {t:.2f} {bar}\n"
        else:
            output += "*All glyphs are cold*\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def do_cross_reference(glyph_or_name):
    if not glyph_or_name.strip():
        return "Please enter a glyph or concept name", "", "", "", ""

    try:
        result = analyze_glyph(glyph_or_name.strip())

        complementary_text = "## Complementary Glyphs\n"
        complementary_text += "*Glyphs that frequently appear together in formulas*\n\n"
        if result["complementary"]:
            for item in result["complementary"]:
                complementary_text += f"- **{item['glyph']}** {item['name']} (appears together {item['frequency']}x)\n"
        else:
            complementary_text += "*No complementary glyphs found*\n"

        oppositions_text = "## Oppositional Glyphs\n"
        oppositions_text += "*Glyphs representing contrasting concepts*\n\n"
        if result["oppositions"]:
            for item in result["oppositions"]:
                oppositions_text += f"- **{item['glyph']}** {item['name']}\n"
        else:
            oppositions_text += "*No oppositions found*\n"

        transformations_text = "## Transformations\n"
        transformations_text += "*What this glyph transforms into (A \u2192 B patterns)*\n\n"
        if result["transformations"]:
            for item in result["transformations"]:
                transformations_text += f"- **{result['glyph']}** \u2192 **{item['glyph']}** ({result['name']} \u2192 {item['name']})\n"
        else:
            transformations_text += "*No transformation patterns found*\n"

        formulas_text = "## Formula Examples\n"
        formulas_text += "*Scroll formulas containing this glyph*\n\n"
        if result["formula_examples"]:
            for i, formula in enumerate(result["formula_examples"], 1):
                formulas_text += f"{i}. `{formula}`\n"
        else:
            formulas_text += "*No formula examples found*\n"

        contexts_text = "## Scroll Contexts\n"
        contexts_text += "*Wisdom from the scrolls*\n\n"
        if result["contexts"]:
            for i, context in enumerate(result["contexts"], 1):
                contexts_text += f"{i}. {context}\n\n"
        else:
            contexts_text += "*No scroll contexts found*\n"

        summary = f"# {result['glyph']} \u2014 {result['name']}\n\n"
        summary += f"**Complementary glyphs:** {len(result['complementary'])} | "
        summary += f"**Oppositions:** {len(result['oppositions'])} | "
        summary += f"**Transformations:** {len(result['transformations'])}\n"

        return summary, complementary_text, oppositions_text, transformations_text, formulas_text + "\n" + contexts_text

    except Exception as e:
        return f"Error: {str(e)}", "", "", "", ""


def do_phase_space():
    try:
        coords = get_phase_space_coordinates()
        if not coords:
            return "*No phase-space data yet. Store memories and trigger recalls first.*"

        # Group by quadrant
        quadrants = {}
        for glyph, data in coords.items():
            q = data["quadrant"]
            if q not in quadrants:
                quadrants[q] = []
            quadrants[q].append((glyph, data))

        output = "# Phase Space\n\n"
        quadrant_labels = {
            "ACTIVE_DEEP_WORK": "Active Deep Work (high temp, high retrieval)",
            "NOVEL_EXPLORATION": "Novel Exploration (high temp, low retrieval)",
            "DORMANT_REACTIVATION": "Dormant Reactivation (low temp, high retrieval)",
            "INACTIVE": "Inactive (low temp, low retrieval)",
        }
        for q_name, q_label in quadrant_labels.items():
            glyphs_in_q = quadrants.get(q_name, [])
            output += f"## {q_label}\n"
            if glyphs_in_q:
                glyphs_in_q.sort(key=lambda x: -x[1].get("x", 0))
                for g, d in glyphs_in_q[:8]:
                    output += (f"- **{g}** temp={d['x']:.2f} "
                               f"ret={d['y']:.2f} off_diag={d['off_diagonal']:.2f}\n")
            else:
                output += "*empty*\n"
            output += "\n"

        # Synchronicity events
        events = get_synchronicity_events(hours=24)
        output += f"## Recent Synchronicity Events ({len(events)})\n"
        if events:
            for evt in events[:10]:
                glyphs_str = " ".join(evt.get("glyphs", []))
                output += (f"- **{evt['event_type']}** [{glyphs_str}] "
                           f"strength={evt.get('strength', 0):.2f}\n")
        else:
            output += "*No synchronicity events in the last 24h*\n"

        # Resonances
        resonances = get_resonances(hours=168)
        output += f"\n## Lattice Resonances ({len(resonances)})\n"
        if resonances:
            for res in resonances[:10]:
                primes_str = ", ".join(str(p) for p in res.get("shared_primes", []))
                output += (f"- Events {res['event_a_id']}\u2194{res['event_b_id']} "
                           f"| primes: [{primes_str}] "
                           f"| strength={res.get('resonance_strength', 0):.2f} "
                           f"| chance={res.get('chance', 0):.2e}\n")
        else:
            output += "*No resonances detected yet*\n"

        # Void profile
        void = get_void_profile()
        if void and void.get("void_center") is not None:
            output += f"\n## Void Profile\n"
            output += f"- **Center:** {void['void_center']:.4f} rad\n"
            output += f"- **Void events:** {void.get('void_event_count', 0)}\n"
            output += f"- **Edge events:** {void.get('edge_event_count', 0)}\n"
            edge = void.get("edge_themes", [])
            if edge:
                output += f"- **Edge themes:** {' '.join(edge[:10])}\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


def do_meta():
    try:
        output = "# Meta-Patterns\n\n"

        # Patterns
        patterns = get_meta_patterns(hours=168)
        output += f"## Discovered Patterns ({len(patterns)})\n"
        if patterns:
            for p in patterns[:10]:
                glyphs_str = " ".join(p.get("glyph_cluster", []))
                output += (f"- **{p['pattern_type']}** [{glyphs_str}]\n"
                           f"  confidence={p.get('confidence', 0):.2f} "
                           f"ratio={p.get('significance_ratio', 0):.1f} "
                           f"size={p.get('cluster_size', 0)} "
                           f"status={p.get('status', '?')}\n")
        else:
            output += "*No meta-patterns discovered yet*\n"

        # Emergent links
        links = get_emergent_links()
        output += f"\n## Emergent Links ({len(links)})\n"
        if links:
            for link in links[:10]:
                glyphs_str = " ".join(link.get("glyph_group", []))
                output += (f"- [{glyphs_str}] "
                           f"type={link.get('pattern_type', '?')} "
                           f"confidence={link.get('confidence', 0):.2f}\n")
        else:
            output += "*No emergent links yet*\n"

        # Thresholds
        thresholds = get_thresholds()
        output += "\n## Adaptive Thresholds\n"
        if thresholds:
            for name, data in thresholds.items():
                burn = " (BURN-IN)" if data.get("in_burn_in") else ""
                output += (f"- **{name}**: {data.get('effective_value', '?')}"
                           f" (default={data.get('default_value', '?')}){burn}\n")
        else:
            output += "*Thresholds not initialized*\n"

        return output
    except Exception as e:
        return f"Error: {str(e)}"


def do_trigger_meta():
    try:
        patterns = trigger_meta_analysis()
        if patterns:
            output = f"## {len(patterns)} New Pattern(s) Found\n\n"
            for p in patterns:
                glyphs_str = " ".join(p.get("glyph_cluster", []))
                output += (f"- **{p['pattern_type']}** [{glyphs_str}] "
                           f"confidence={p.get('confidence', 0):.2f}\n")
            return output
        else:
            return "*No new patterns found (may be rate-limited to 1/24h, or insufficient synchronicity events).*"
    except Exception as e:
        return f"Error: {str(e)}"


def build_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks demo."""
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

        with gr.Tab("PAIR\u2192TEXT"):
            pin = gr.Textbox(label="Glyph Pair (e.g., Z / \u21a8)")
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
                    placeholder="e.g., Joy, \u29eb, Potential, Z",
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
                        placeholder="e.g., \u29de\u25ec\u27f2\u29c9",
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

        with gr.Tab("PHASE SPACE"):
            gr.Markdown("### Phase Space & Synchronicity Detection")
            gr.Markdown("Glyph coordinates in (temperature x retrieval frequency) space, "
                        "synchronicity events, lattice resonances, and void profile.")
            ps_btn = gr.Button("Compute Phase Space", variant="primary")
            ps_out = gr.Markdown(label="Phase Space")
            ps_btn.click(fn=do_phase_space, inputs=None, outputs=ps_out)

        with gr.Tab("META"):
            gr.Markdown("### Meta-Pattern Analysis & Feedback")
            gr.Markdown("Discovered patterns from clustering synchronicity events, "
                        "emergent glyph links, and adaptive thresholds.")

            with gr.Row():
                meta_view_btn = gr.Button("View Patterns", variant="secondary")
                meta_run_btn = gr.Button("Run Meta-Analysis", variant="primary")

            meta_out = gr.Markdown(label="Meta Patterns")
            meta_view_btn.click(fn=do_meta, inputs=None, outputs=meta_out)
            meta_run_btn.click(fn=do_trigger_meta, inputs=None, outputs=meta_out)

    return demo
