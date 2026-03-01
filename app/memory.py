# app/memory.py
"""
Glyph Memory Engine for GPT-GU

Temporal, heat-gated memory system where the system thinks in glyphs.
Memories are stored as glyph formulas (primary representation).
Text is a lossy translation layer.

Each memory carries a heat signature — the glyph temperature landscape
at the time of storage — which acts as the retrieval key. Only memories
whose stored heat matches the current heat map are surfaced.
"""
from __future__ import annotations
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.guardrails import LEX_DICT2GLYPH, formula_back_to_glyphs
from app.salience import (
    compute_store_salience, compute_usage_salience,
    compute_composite, upgrade_record, SUCCESS_WINDOW_SECONDS,
)

# ----------------------------
# Constants
# ----------------------------
DECAY_RATE = 0.05           # 5% exponential decay per tick
SIMILARITY_THRESHOLD = 0.6  # minimum cosine similarity for recall gate
COLD_THRESHOLD = 0.1        # below this, glyph temp is pruned from heat map
BUMP_AMOUNT = 1.0           # default temperature bump per usage

_DATA_DIR = Path(__file__).parent.parent
_DEFAULT_STORE = _DATA_DIR / "memory_store.jsonl"
_DEFAULT_HEAT = _DATA_DIR / "heat_map.json"


class GlyphMemory:
    """
    Heat-gated glyph memory system.

    The heat map tracks activation intensity per glyph across all
    interactions. Memories are stored with a snapshot of the heat map
    at creation time. Retrieval only surfaces memories whose stored
    heat signature is similar to the current heat state.
    """

    def __init__(self, store_path: str = None, heat_path: str = None):
        self.store_path = Path(store_path) if store_path else _DEFAULT_STORE
        self.heat_path = Path(heat_path) if heat_path else _DEFAULT_HEAT

        # Inverse lexicon for glyph name lookups
        self.glyph_to_name: Dict[str, str] = {v: k for k, v in LEX_DICT2GLYPH.items()}

        # In-memory state
        self.memories: List[dict] = []
        self.temperatures: Dict[str, float] = {}
        self.last_decay: str = datetime.now(timezone.utc).isoformat()
        self.tick_count: int = 0
        self._next_id: int = 1

        # L1: Track recent recalls for success heuristic
        # {memory_id: timestamp_iso} — if re-recalled within window, mark previous as success
        self._recent_recalls: Dict[str, str] = {}

        # Load persisted state
        self._load_heat()
        self._load_memories()

    # ----------------------------------------------------------------
    # Heat Map
    # ----------------------------------------------------------------

    def bump(self, glyphs: List[str], amount: float = BUMP_AMOUNT):
        """Increase temperature for given glyphs."""
        for g in glyphs:
            self.temperatures[g] = self.temperatures.get(g, 0.0) + amount
        self._save_heat()

        # L2: Log tag events for trend engine
        try:
            from app.trends import get_trend_engine
            get_trend_engine().log_tag_events_batch(glyphs, "bump", amount)
        except Exception:
            pass  # trend tracking never breaks core

    def decay(self):
        """Apply exponential decay to all temperatures and prune cold glyphs."""
        pruned = {}
        for g, temp in self.temperatures.items():
            new_temp = temp * (1.0 - DECAY_RATE)
            if new_temp >= COLD_THRESHOLD:
                pruned[g] = new_temp
        self.temperatures = pruned
        self.last_decay = datetime.now(timezone.utc).isoformat()
        self.tick_count += 1
        self._save_heat()

    def get_heat_map(self) -> Dict[str, float]:
        """Return current temperatures (only above cold threshold)."""
        return {g: t for g, t in self.temperatures.items() if t >= COLD_THRESHOLD}

    def heat_relevance_score(self) -> float:
        """Overall heat intensity — sum of all temperatures."""
        return sum(self.temperatures.values())

    # ----------------------------------------------------------------
    # Memory Storage
    # ----------------------------------------------------------------

    def store(self, text: str) -> dict:
        """Store a new memory from text.

        Converts text to a glyph formula (the system thinks in glyphs),
        extracts the glyph list, snapshots the current heat map,
        and persists the record.
        """
        formula = formula_back_to_glyphs(text)
        return self._store_record(formula)

    def store_formula(self, formula: str) -> dict:
        """Store a memory that's already a glyph formula."""
        return self._store_record(formula)

    def _store_record(self, formula: str) -> dict:
        """Internal: create and persist a memory record."""
        glyphs = self._extract_glyphs(formula)

        # Snapshot current heat map before bumping
        heat_snapshot = self.get_heat_map()

        # Compute store salience from current cognitive intensity
        total_heat = self.heat_relevance_score()
        store_sal = compute_store_salience(total_heat)

        now_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "id": f"m_{self._next_id:04d}",
            "formula": formula,
            "glyphs": glyphs,
            "timestamp": now_iso,
            "activation_count": 0,
            "store_salience": round(store_sal, 4),
            # L1: Dual-track salience
            "salience_model": round(store_sal, 4),
            "salience_usage": 0.5,
            "salience_composite": round(compute_composite(store_sal, 0.5), 4),
            "last_accessed": now_iso,
            "retrieval_success_count": 0,
            "retrieval_total_count": 0,
            "is_archived": False,
            "heat": heat_snapshot,
            "source_scroll": None,
        }

        self._next_id += 1
        self.memories.append(record)

        # Bump heat for the memory's own glyphs
        self.bump(glyphs)

        # L2: Log store event (higher weight than bump)
        try:
            from app.trends import get_trend_engine
            get_trend_engine().log_tag_events_batch(glyphs, "store", 1.5)
        except Exception:
            pass

        # L5: Update glyph network index and tensor
        try:
            from app.glyph_network import get_glyph_network
            net = get_glyph_network()
            net.index_memory(record["id"], glyphs)
            net.update_cooccurrence(glyphs)
            net.update_semiotic(glyphs)
        except Exception:
            pass  # glyph network never breaks core

        # Persist
        self._append_memory(record)
        return record

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    def recall(self, top_n: int = 5) -> List[dict]:
        """Retrieve memories matching current heat state.

        Only memories whose stored heat signature has cosine similarity
        above SIMILARITY_THRESHOLD with the current heat map are
        candidates. Scored by similarity * composite salience.
        """
        current_heat = self.get_heat_map()
        if not current_heat:
            return []

        scored = []
        for mem in self.memories:
            if mem.get("is_archived", False):
                continue
            sim = self._cosine_similarity(mem["heat"], current_heat)
            if sim >= SIMILARITY_THRESHOLD:
                composite = mem.get("salience_composite",
                                    mem.get("store_salience", 0.5))
                score = sim * (composite + 0.1)
                scored.append((score, sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        now_iso = datetime.now(timezone.utc).isoformat()
        results = []
        for score, sim, mem in scored[:top_n]:
            # L1: Check retrieval success heuristic
            self._check_retrieval_success(mem["id"], now_iso)

            mem["activation_count"] += 1
            mem["retrieval_total_count"] = mem.get("retrieval_total_count", 0) + 1
            mem["last_accessed"] = now_iso

            # Recompute usage salience
            usage = compute_usage_salience(
                mem["activation_count"],
                mem["last_accessed"],
                mem.get("retrieval_success_count", 0),
                mem.get("retrieval_total_count", 1),
            )
            mem["salience_usage"] = round(usage, 4)
            mem["salience_composite"] = round(
                compute_composite(
                    mem.get("salience_model", mem.get("store_salience", 0.5)),
                    usage,
                ), 4
            )

            # Track for success heuristic
            self._recent_recalls[mem["id"]] = now_iso

            results.append({
                **mem,
                "_recall_score": round(score, 4),
                "_similarity": round(sim, 4),
            })

        # Persist updated salience and activation counts
        if results:
            self._save_memories()

            # L2: Log retrieval event and glyph recall events
            try:
                from app.trends import get_trend_engine, get_retrieval_tracker
                get_retrieval_tracker().log_retrieval("heat_match", results)
                all_glyphs = set()
                for r in results:
                    all_glyphs.update(r.get("glyphs", []))
                if all_glyphs:
                    get_trend_engine().log_tag_events_batch(
                        list(all_glyphs), "recall", 0.5)
            except Exception:
                pass

            # L3: Trigger synchronicity detection and lattice resonance
            try:
                from app.phase_space import get_phase_space
                from app.lattice import get_lattice
                new_events = get_phase_space().run_all_detectors()
                if new_events:
                    lattice = get_lattice()
                    for evt in new_events:
                        if evt.get("id"):
                            lattice.detect_resonances(evt["id"])
            except Exception:
                pass

            # L5: Update glyph network from retrieval
            try:
                from app.glyph_network import get_glyph_network
                net = get_glyph_network()
                all_glyphs_list = list(all_glyphs)
                if len(all_glyphs_list) >= 2:
                    net.update_cooccurrence(all_glyphs_list)
                # Feed synchronicity events into tensor
                if new_events:
                    for evt in new_events:
                        evt_glyphs = evt.get("glyphs", [])
                        if isinstance(evt_glyphs, str):
                            import json as _json
                            try:
                                evt_glyphs = _json.loads(evt_glyphs)
                            except Exception:
                                evt_glyphs = []
                        strength = evt.get("strength", 0.0)
                        for i, ga in enumerate(evt_glyphs):
                            for gb in evt_glyphs[i + 1:]:
                                net.update_synchronicity(ga, gb, strength)
            except Exception:
                pass

        return results

    def recall_by_glyph(self, glyph: str, top_n: int = 5) -> List[dict]:
        """Retrieve memories containing a specific glyph, still heat-gated."""
        current_heat = self.get_heat_map()
        if not current_heat:
            return []

        scored = []
        for mem in self.memories:
            if mem.get("is_archived", False):
                continue
            if glyph not in mem["glyphs"]:
                continue
            sim = self._cosine_similarity(mem["heat"], current_heat)
            if sim >= SIMILARITY_THRESHOLD:
                composite = mem.get("salience_composite",
                                    mem.get("store_salience", 0.5))
                score = sim * (composite + 0.1)
                scored.append((score, sim, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        now_iso = datetime.now(timezone.utc).isoformat()
        results = []
        for score, sim, mem in scored[:top_n]:
            self._check_retrieval_success(mem["id"], now_iso)

            mem["activation_count"] += 1
            mem["retrieval_total_count"] = mem.get("retrieval_total_count", 0) + 1
            mem["last_accessed"] = now_iso

            usage = compute_usage_salience(
                mem["activation_count"],
                mem["last_accessed"],
                mem.get("retrieval_success_count", 0),
                mem.get("retrieval_total_count", 1),
            )
            mem["salience_usage"] = round(usage, 4)
            mem["salience_composite"] = round(
                compute_composite(
                    mem.get("salience_model", mem.get("store_salience", 0.5)),
                    usage,
                ), 4
            )

            self._recent_recalls[mem["id"]] = now_iso

            results.append({
                **mem,
                "_recall_score": round(score, 4),
                "_similarity": round(sim, 4),
            })

        if results:
            self._save_memories()

            # L2: Log retrieval event and glyph recall events
            try:
                from app.trends import get_trend_engine, get_retrieval_tracker
                get_retrieval_tracker().log_retrieval("glyph_filter", results)
                all_glyphs = set()
                for r in results:
                    all_glyphs.update(r.get("glyphs", []))
                if all_glyphs:
                    get_trend_engine().log_tag_events_batch(
                        list(all_glyphs), "recall", 0.5)
            except Exception:
                pass

            # L3: Trigger synchronicity detection and lattice resonance
            try:
                from app.phase_space import get_phase_space
                from app.lattice import get_lattice
                new_events = get_phase_space().run_all_detectors()
                if new_events:
                    lattice = get_lattice()
                    for evt in new_events:
                        if evt.get("id"):
                            lattice.detect_resonances(evt["id"])
            except Exception:
                pass

        return results

    def _check_retrieval_success(self, memory_id: str, now_iso: str):
        """If this memory was recalled recently, count that as a success."""
        prev_ts = self._recent_recalls.get(memory_id)
        if prev_ts is None:
            return
        try:
            prev_dt = datetime.fromisoformat(prev_ts)
            now_dt = datetime.fromisoformat(now_iso)
            if prev_dt.tzinfo is None:
                prev_dt = prev_dt.replace(tzinfo=timezone.utc)
            if now_dt.tzinfo is None:
                now_dt = now_dt.replace(tzinfo=timezone.utc)
            elapsed = (now_dt - prev_dt).total_seconds()
            if 0 < elapsed <= SUCCESS_WINDOW_SECONDS:
                # Find the memory and bump its success count
                for mem in self.memories:
                    if mem["id"] == memory_id:
                        mem["retrieval_success_count"] = (
                            mem.get("retrieval_success_count", 0) + 1
                        )
                        break
        except (ValueError, TypeError):
            pass

    def _cosine_similarity(self, heat_a: dict, heat_b: dict) -> float:
        """Cosine similarity between two heat maps (sparse vectors)."""
        if not heat_a or not heat_b:
            return 0.0

        # All glyphs present in either map
        all_glyphs = set(heat_a.keys()) | set(heat_b.keys())

        dot = 0.0
        mag_a = 0.0
        mag_b = 0.0
        for g in all_glyphs:
            a = heat_a.get(g, 0.0)
            b = heat_b.get(g, 0.0)
            dot += a * b
            mag_a += a * a
            mag_b += b * b

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (math.sqrt(mag_a) * math.sqrt(mag_b))

    # ----------------------------------------------------------------
    # Synchronicity Detection
    # ----------------------------------------------------------------

    def detect_synchronicity(self) -> List[dict]:
        """Detect co-activation of glyphs that rarely co-occur.

        Uses training data co-occurrences as baseline expectation.
        When two hot glyphs have low or zero co-occurrence in the
        training corpus, that's a synchronicity event.
        """
        try:
            from app.cross_reference import get_cross_reference_engine
            xref = get_cross_reference_engine()
        except Exception:
            return []

        hot = self.get_heat_map()
        hot_glyphs = [g for g, t in sorted(hot.items(), key=lambda x: -x[1])
                      if t >= 1.0]  # only meaningfully hot glyphs

        if len(hot_glyphs) < 2:
            return []

        events = []
        seen = set()
        for i, ga in enumerate(hot_glyphs):
            for gb in hot_glyphs[i + 1:]:
                pair = tuple(sorted([ga, gb]))
                if pair in seen:
                    continue
                seen.add(pair)

                # Check training corpus co-occurrence
                co_count = xref.co_occurrences.get(ga, {}).get(gb, 0)

                # Low co-occurrence + both hot = synchronicity
                if co_count <= 1:
                    name_a = self.glyph_to_name.get(ga, ga)
                    name_b = self.glyph_to_name.get(gb, gb)
                    temp_a = hot[ga]
                    temp_b = hot[gb]
                    # Rarity score: higher when both are hotter and co-occurrence is lower
                    rarity = (temp_a + temp_b) / (1.0 + co_count)
                    events.append({
                        "glyph_a": ga,
                        "glyph_b": gb,
                        "name_a": name_a,
                        "name_b": name_b,
                        "temp_a": round(temp_a, 2),
                        "temp_b": round(temp_b, 2),
                        "co_occurrence": co_count,
                        "rarity_score": round(rarity, 2),
                    })

        events.sort(key=lambda x: x["rarity_score"], reverse=True)
        return events[:10]  # top 10 synchronicity events

    # ----------------------------------------------------------------
    # Cognitive Weather
    # ----------------------------------------------------------------

    def weather_report(self) -> dict:
        """The system's current cognitive state in its own language."""
        hot = self.get_heat_map()

        # Top glyphs by temperature
        sorted_glyphs = sorted(hot.items(), key=lambda x: -x[1])[:10]
        hot_glyphs = [
            {"glyph": g, "name": self.glyph_to_name.get(g, g), "temp": round(t, 2)}
            for g, t in sorted_glyphs
        ]

        # Count thermally reachable memories
        current_heat = self.get_heat_map()
        active_count = 0
        for mem in self.memories:
            sim = self._cosine_similarity(mem["heat"], current_heat)
            if sim >= SIMILARITY_THRESHOLD:
                active_count += 1

        # L2: Trend data
        trending_glyphs = []
        retrieval_stats = {}
        try:
            from app.trends import get_trend_engine, get_retrieval_tracker
            trending_glyphs = get_trend_engine().get_trending_glyphs(top_n=10)
            retrieval_stats = get_retrieval_tracker().get_retrieval_stats()
        except Exception:
            pass

        # L3: Phase space + synchronicity events
        phase_data = {}
        lattice_data = {}
        try:
            from app.phase_space import get_phase_space
            ps = get_phase_space()
            phase_data = {
                "quadrant_counts": ps.get_quadrant_distribution(),
                "recent_sync_events": ps.get_recent_events(hours=24),
            }
        except Exception:
            pass

        try:
            from app.lattice import get_lattice
            lattice = get_lattice()
            lattice_data = {
                "recent_resonances": lattice.get_recent_resonances(hours=24),
                "void_profile": lattice.compute_void_profile(),
            }
        except Exception:
            pass

        # L4: Meta patterns (passively trigger analysis if due)
        meta_data = {}
        try:
            from app.meta import (
                get_meta_analyzer, get_feedback_engine,
                get_adaptive_thresholds,
            )
            analyzer = get_meta_analyzer()
            new_patterns = analyzer.run_analysis()
            if new_patterns:
                fb = get_feedback_engine()
                for pattern in new_patterns:
                    fb.process_pattern(pattern)

            meta_data = {
                "recent_patterns": analyzer.get_recent_patterns(hours=168),
                "thresholds": get_adaptive_thresholds().get_all(),
                "emergent_links": get_feedback_engine().get_emergent_links(),
            }
        except Exception:
            pass

        # L5: Glyph Ramsey network stats
        network_data = {}
        try:
            from app.glyph_network import get_glyph_network
            net = get_glyph_network()
            export = net.export_network(max_nodes=0)
            network_data = export.get("stats", {})
        except Exception:
            pass

        return {
            "hot_glyphs": hot_glyphs,
            "total_heat": round(self.heat_relevance_score(), 2),
            "total_memories": len(self.memories),
            "active_memories": active_count,
            "tick_count": self.tick_count,
            "synchronicities": self.detect_synchronicity(),
            "trending_glyphs": trending_glyphs,
            "retrieval_stats": retrieval_stats,
            "phase_space": phase_data,
            "lattice": lattice_data,
            "meta_patterns": meta_data,
            "glyph_network": network_data,
        }

    # ----------------------------------------------------------------
    # Glyph Extraction
    # ----------------------------------------------------------------

    def _extract_glyphs(self, text: str) -> List[str]:
        """Extract individual glyphs from a formula string."""
        import re
        cleaned = re.sub(r'[→⇒+:\-\s/]', ' ', text)

        glyphs = []
        for char in cleaned:
            if char.strip() and char in self.glyph_to_name:
                glyphs.append(char)

        # Fallback: character by character on original
        if not glyphs:
            for char in text:
                if char in self.glyph_to_name:
                    glyphs.append(char)

        return glyphs

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def _load_heat(self):
        """Load heat map from disk."""
        if self.heat_path.exists():
            try:
                with open(self.heat_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.temperatures = data.get("temperatures", {})
                self.last_decay = data.get("last_decay", self.last_decay)
                self.tick_count = data.get("tick_count", 0)
            except (json.JSONDecodeError, KeyError):
                pass  # start fresh

    def _save_heat(self):
        """Persist heat map to disk."""
        data = {
            "temperatures": self.temperatures,
            "last_decay": self.last_decay,
            "tick_count": self.tick_count,
        }
        with open(self.heat_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_memories(self):
        """Load all memories from JSONL file."""
        self.memories = []
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            upgrade_record(record)
                            self.memories.append(record)
                        except json.JSONDecodeError:
                            continue
            except FileNotFoundError:
                pass

        # Set next ID based on existing memories
        if self.memories:
            max_id = 0
            for m in self.memories:
                try:
                    num = int(m["id"].split("_")[1])
                    if num > max_id:
                        max_id = num
                except (ValueError, IndexError, KeyError):
                    pass
            self._next_id = max_id + 1

    def _append_memory(self, record: dict):
        """Append a single memory to JSONL file."""
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_memories(self):
        """Rewrite entire memory store (for activation_count updates)."""
        with open(self.store_path, "w", encoding="utf-8") as f:
            for mem in self.memories:
                # Strip transient recall metadata before saving
                clean = {k: v for k, v in mem.items()
                         if not k.startswith("_")}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ----------------------------
# Singleton
# ----------------------------
_memory_engine: Optional[GlyphMemory] = None


def get_memory_engine() -> GlyphMemory:
    """Get or create the singleton memory engine."""
    global _memory_engine
    if _memory_engine is None:
        _memory_engine = GlyphMemory()
    return _memory_engine


# ----------------------------
# Convenience functions
# ----------------------------

def store_memory(text: str) -> dict:
    """Store a text memory (converts to glyph formula)."""
    return get_memory_engine().store(text)


def store_glyph_memory(formula: str) -> dict:
    """Store a memory that's already a glyph formula."""
    return get_memory_engine().store_formula(formula)


def recall_memories(top_n: int = 5) -> list:
    """Recall thermally-compatible memories."""
    return get_memory_engine().recall(top_n)


def recall_by_glyph(glyph: str, top_n: int = 5) -> list:
    """Recall memories containing a specific glyph, heat-gated."""
    return get_memory_engine().recall_by_glyph(glyph, top_n)


def get_weather() -> dict:
    """Get cognitive weather report."""
    return get_memory_engine().weather_report()


def bump_glyphs(glyphs: list):
    """Bump temperature for given glyphs."""
    get_memory_engine().bump(glyphs)


def trigger_decay():
    """Manually trigger a decay tick."""
    get_memory_engine().decay()


def get_raw_heat_map() -> dict:
    """Get the raw heat map data."""
    engine = get_memory_engine()
    return {
        "temperatures": engine.get_heat_map(),
        "total_heat": round(engine.heat_relevance_score(), 2),
        "last_decay": engine.last_decay,
        "tick_count": engine.tick_count,
    }


def get_glyph_trend(glyph: str) -> dict:
    """Get trend data for a specific glyph."""
    try:
        from app.trends import get_trend_engine
        return get_trend_engine().compute_glyph_trend(glyph)
    except Exception:
        return {}


def get_all_trends() -> dict:
    """Get trend data for all active glyphs."""
    try:
        from app.trends import get_trend_engine
        return get_trend_engine().compute_all_trends()
    except Exception:
        return {}


def mark_retrieval_success(memory_id: str):
    """Explicitly mark a memory retrieval as successful.

    For future use when an LLM feedback loop is added.
    """
    engine = get_memory_engine()
    for mem in engine.memories:
        if mem["id"] == memory_id:
            mem["retrieval_success_count"] = (
                mem.get("retrieval_success_count", 0) + 1
            )
            engine._save_memories()
            break


# ----------------------------
# L3 convenience functions
# ----------------------------

def get_phase_space_coordinates() -> dict:
    """Get phase-space coordinates for all active glyphs."""
    try:
        from app.phase_space import get_phase_space
        return get_phase_space().compute_glyph_coordinates()
    except Exception:
        return {}


def get_synchronicity_events(hours: float = 168.0) -> list:
    """Get recent synchronicity events."""
    try:
        from app.phase_space import get_phase_space
        return get_phase_space().get_recent_events(hours)
    except Exception:
        return []


def get_resonances(hours: float = 168.0) -> list:
    """Get recent lattice resonances."""
    try:
        from app.lattice import get_lattice
        return get_lattice().get_recent_resonances(hours)
    except Exception:
        return []


def get_void_profile() -> dict:
    """Get Ramsey lattice void profile."""
    try:
        from app.lattice import get_lattice
        return get_lattice().compute_void_profile()
    except Exception:
        return {}


# ----------------------------
# L4 convenience functions
# ----------------------------

def trigger_meta_analysis() -> list:
    """Manually trigger meta-analysis (rate-limited to 1/24h)."""
    try:
        from app.meta import get_meta_analyzer, get_feedback_engine
        analyzer = get_meta_analyzer()
        patterns = analyzer.run_analysis()
        if patterns:
            fb = get_feedback_engine()
            for p in patterns:
                fb.process_pattern(p)
        return patterns
    except Exception:
        return []


def get_meta_patterns(hours: float = 168.0) -> list:
    """Get recent meta-patterns."""
    try:
        from app.meta import get_meta_analyzer
        return get_meta_analyzer().get_recent_patterns(hours)
    except Exception:
        return []


def get_emergent_links() -> list:
    """Get active emergent links."""
    try:
        from app.meta import get_feedback_engine
        return get_feedback_engine().get_emergent_links()
    except Exception:
        return []


def get_thresholds() -> dict:
    """Get current adaptive threshold values."""
    try:
        from app.meta import get_adaptive_thresholds
        return get_adaptive_thresholds().get_all()
    except Exception:
        return {}


# ----------------------------
# L5: Glyph Ramsey Network convenience functions
# ----------------------------

def recall_by_association(query_glyphs: list, top_n: int = 5,
                          heat_gate: bool = False) -> list:
    """Recall memories by glyph Ramsey network association."""
    try:
        from app.glyph_network import get_glyph_network
        engine = get_memory_engine()
        net = get_glyph_network()
        current_heat = engine.get_heat_map() if heat_gate else None
        return net.recall_by_association(
            query_glyphs, engine.memories, top_n,
            heat_gate=heat_gate, current_heat=current_heat,
        )
    except Exception:
        return []


def deep_recall(seed_glyphs: list, max_depth: int = 3,
                per_level_k: int = 5) -> list:
    """Deep recursive association-chain retrieval via Ramsey network."""
    try:
        from app.glyph_network import get_glyph_network
        engine = get_memory_engine()
        net = get_glyph_network()
        return net.deep_recall(
            seed_glyphs, engine.memories,
            max_depth=max_depth, per_level_k=per_level_k,
            current_heat=engine.get_heat_map(),
        )
    except Exception:
        return []


def get_glyph_network_data(min_composite: float = 0.05) -> dict:
    """Get glyph Ramsey network for visualization."""
    try:
        from app.glyph_network import get_glyph_network
        return get_glyph_network().export_network(min_composite=min_composite)
    except Exception:
        return {"nodes": [], "edges": [], "stats": {}}


def get_glyph_profile(glyph: str) -> dict:
    """Get full Ramsey network profile for a glyph."""
    try:
        from app.glyph_network import get_glyph_network
        return get_glyph_network().get_glyph_profile(glyph)
    except Exception:
        return {}
