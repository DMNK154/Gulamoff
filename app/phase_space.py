# app/phase_space.py
"""
Layer 3: Phase Space & Synchronicity Detection for GPT-GU

Maps each glyph to a 2D coordinate (trend temperature x retrieval frequency)
and detects three types of synchronicity events:
  1. Dormant Reactivation: cold glyphs with anomalous retrieval rates
  2. Cross-Domain Bridges: novel co-occurrence of unrelated glyphs
  3. Semantic Convergence: heat-similar memories with no shared glyphs
"""
from __future__ import annotations
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# ----------------------------
# Constants
# ----------------------------
QUADRANT_SPLIT = 0.5
SQRT2 = math.sqrt(2)
SYNC_MIN_INTERVAL_SECONDS = 60      # rate limit: max 1 sync run per 60s
MAX_RECENT_RETRIEVALS = 20          # how many retrieval events to analyze
DORMANT_TEMP_CEILING = 0.3          # temperature must be below this for dormant
INACTIVE_JUMP_BONUS = 1.5           # strength multiplier for INACTIVE→DORMANT jumps


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_quadrant(temperature: float, retrieval_freq: float) -> str:
    """Assign a glyph to one of four phase-space quadrants."""
    high_temp = temperature >= QUADRANT_SPLIT
    high_ret = retrieval_freq >= QUADRANT_SPLIT
    if high_temp and high_ret:
        return "ACTIVE_DEEP_WORK"
    elif high_temp and not high_ret:
        return "NOVEL_EXPLORATION"
    elif not high_temp and high_ret:
        return "DORMANT_REACTIVATION"
    else:
        return "INACTIVE"


class PhaseSpace:
    """Phase-space analysis and synchronicity detection."""

    def __init__(self, conn):
        self._conn = conn
        self._last_sync_run: Optional[str] = None
        self._prev_quadrants: Dict[str, str] = {}

    # ----------------------------------------------------------------
    # Phase Space Coordinates
    # ----------------------------------------------------------------

    def compute_glyph_coordinates(self) -> Dict[str, dict]:
        """Map each active glyph to phase-space coordinates.

        Returns {glyph: {x, y, quadrant, off_diagonal}}.
        """
        from app.trends import get_trend_engine, get_retrieval_tracker

        trends = get_trend_engine().compute_all_trends()
        tracker = get_retrieval_tracker()

        coords = {}
        for glyph, trend in trends.items():
            temp = trend.get("temperature", 0.0)
            ret_freq = tracker.get_glyph_retrieval_frequency(glyph)
            quadrant = _classify_quadrant(temp, ret_freq)
            off_diag = (ret_freq - temp) / SQRT2

            coords[glyph] = {
                "x": round(temp, 4),
                "y": round(ret_freq, 4),
                "quadrant": quadrant,
                "off_diagonal": round(off_diag, 4),
                "level": trend.get("level", 0.0),
                "velocity": trend.get("velocity", 0.0),
            }

        return coords

    def get_quadrant_distribution(self) -> Dict[str, int]:
        """Count glyphs in each quadrant."""
        coords = self.compute_glyph_coordinates()
        dist = defaultdict(int)
        for data in coords.values():
            dist[data["quadrant"]] += 1
        return dict(dist)

    # ----------------------------------------------------------------
    # Synchronicity Detector 1: Dormant Reactivation
    # ----------------------------------------------------------------

    def detect_dormant_reactivation(self, coords: Dict[str, dict],
                                     thresholds: Dict[str, float]
                                     ) -> List[dict]:
        """Detect glyphs in DORMANT_REACTIVATION with anomalous retrieval rates.

        Triggered when:
          - z_score > thresholds[z_score_threshold]
          - temperature < DORMANT_TEMP_CEILING
          - Glyph is in DORMANT_REACTIVATION quadrant
        """
        z_threshold = thresholds.get("z_score_threshold", 2.0)

        # Collect all retrieval frequencies
        all_freqs = [c["y"] for c in coords.values() if c["y"] > 0]
        if len(all_freqs) < 2:
            return []

        mean_freq = statistics.mean(all_freqs)
        stdev_freq = statistics.stdev(all_freqs)
        if stdev_freq == 0:
            return []

        events = []
        for glyph, data in coords.items():
            if data["quadrant"] != "DORMANT_REACTIVATION":
                continue
            if data["x"] >= DORMANT_TEMP_CEILING:
                continue

            z_score = (data["y"] - mean_freq) / stdev_freq
            if z_score <= z_threshold:
                continue

            # Check if just jumped from INACTIVE
            prev = self._prev_quadrants.get(glyph, "INACTIVE")
            bonus = INACTIVE_JUMP_BONUS if prev == "INACTIVE" else 1.0
            strength = z_score * bonus

            events.append({
                "event_type": "dormant_reactivation",
                "glyphs": [glyph],
                "strength": round(strength, 4),
                "details": {
                    "z_score": round(z_score, 4),
                    "retrieval_freq": data["y"],
                    "temperature": data["x"],
                    "jumped_from_inactive": prev == "INACTIVE",
                },
            })

        return events

    # ----------------------------------------------------------------
    # Synchronicity Detector 2: Cross-Domain Bridges
    # ----------------------------------------------------------------

    def detect_cross_domain_bridges(self, thresholds: Dict[str, float]
                                     ) -> List[dict]:
        """Detect novel co-occurrence of unrelated glyphs in retrieval results.

        For glyph pairs with zero training co-occurrence appearing together
        in a retrieval, compute information-theoretic surprise.
        Triggered when surprise > thresholds[surprise_threshold].
        """
        surprise_threshold = thresholds.get("surprise_threshold", 3.0)

        try:
            from app.cross_reference import get_cross_reference_engine
            xref = get_cross_reference_engine()
        except Exception:
            return []

        try:
            from app.memory import get_memory_engine
            engine = get_memory_engine()
            total_memories = max(len(engine.memories), 1)
        except Exception:
            return []

        # Get recent retrieval glyph sets
        rows = self._conn.execute(
            "SELECT glyphs_in_results FROM retrieval_events "
            "ORDER BY timestamp DESC LIMIT ?",
            (MAX_RECENT_RETRIEVALS,),
        ).fetchall()

        if not rows:
            return []

        events = []
        seen_pairs = set()

        for (glyphs_json,) in rows:
            if not glyphs_json:
                continue
            try:
                glyphs = json.loads(glyphs_json)
            except (json.JSONDecodeError, TypeError):
                continue

            for i, ga in enumerate(glyphs):
                for gb in glyphs[i + 1:]:
                    pair = tuple(sorted([ga, gb]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    # Check training co-occurrence
                    co_count = xref.co_occurrences.get(ga, {}).get(gb, 0)
                    if co_count > 0:
                        continue  # not novel

                    # Compute surprise
                    # freq = fraction of memories containing this glyph
                    freq_a = sum(1 for m in engine.memories
                                 if ga in m.get("glyphs", [])) / total_memories
                    freq_b = sum(1 for m in engine.memories
                                 if gb in m.get("glyphs", [])) / total_memories

                    if freq_a == 0 or freq_b == 0:
                        # If glyph never appears in memories, can't compute surprise
                        # Use a high default surprise
                        surprise = 5.0
                    else:
                        expected = freq_a * freq_b * total_memories
                        surprise = -math.log(max(expected / total_memories, 1e-10))

                    if surprise <= surprise_threshold:
                        continue

                    events.append({
                        "event_type": "cross_domain_bridge",
                        "glyphs": list(pair),
                        "strength": round(surprise, 4),
                        "details": {
                            "surprise_nats": round(surprise, 4),
                            "freq_a": round(freq_a, 4),
                            "freq_b": round(freq_b, 4),
                            "co_occurrence": co_count,
                        },
                    })

        return events

    # ----------------------------------------------------------------
    # Synchronicity Detector 3: Semantic Convergence
    # ----------------------------------------------------------------

    def detect_semantic_convergence(self, thresholds: Dict[str, float]
                                    ) -> List[dict]:
        """Detect memories from different retrievals with high heat similarity
        but no shared glyphs.

        Uses stored heat-map snapshots as the embedding space.
        Triggered when cosine_similarity > thresholds[convergence_threshold].
        """
        conv_threshold = thresholds.get("convergence_threshold", 0.7)

        try:
            from app.memory import get_memory_engine
            engine = get_memory_engine()
        except Exception:
            return []

        # Get recent retrieval events with their memory hits
        ret_rows = self._conn.execute(
            "SELECT re.id, rmh.memory_id FROM retrieval_events re "
            "JOIN retrieval_memory_hits rmh ON re.id = rmh.retrieval_event_id "
            "ORDER BY re.timestamp DESC LIMIT ?",
            (MAX_RECENT_RETRIEVALS * 5,),
        ).fetchall()

        if not ret_rows:
            return []

        # Group memory IDs by retrieval event
        retrieval_memories = defaultdict(list)
        for ret_id, mem_id in ret_rows:
            retrieval_memories[ret_id].append(mem_id)

        # Build memory lookup
        mem_lookup = {m["id"]: m for m in engine.memories}

        # Compare memories from different retrieval events
        events = []
        seen_pairs = set()
        ret_ids = list(retrieval_memories.keys())

        for i, ret_a in enumerate(ret_ids):
            for ret_b in ret_ids[i + 1:]:
                for mem_id_a in retrieval_memories[ret_a]:
                    for mem_id_b in retrieval_memories[ret_b]:
                        if mem_id_a == mem_id_b:
                            continue
                        pair = tuple(sorted([mem_id_a, mem_id_b]))
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)

                        mem_a = mem_lookup.get(mem_id_a)
                        mem_b = mem_lookup.get(mem_id_b)
                        if not mem_a or not mem_b:
                            continue

                        # Check no shared glyphs
                        glyphs_a = set(mem_a.get("glyphs", []))
                        glyphs_b = set(mem_b.get("glyphs", []))
                        if glyphs_a & glyphs_b:
                            continue  # shared glyphs = not convergence

                        # Compute heat-map similarity
                        sim = engine._cosine_similarity(
                            mem_a.get("heat", {}),
                            mem_b.get("heat", {}),
                        )
                        if sim <= conv_threshold:
                            continue

                        # Convergence detected
                        combined_glyphs = list(glyphs_a | glyphs_b)
                        events.append({
                            "event_type": "semantic_convergence",
                            "glyphs": combined_glyphs,
                            "strength": round(sim, 4),
                            "details": {
                                "similarity": round(sim, 4),
                                "memory_a": mem_id_a,
                                "memory_b": mem_id_b,
                                "glyphs_a": list(glyphs_a),
                                "glyphs_b": list(glyphs_b),
                            },
                        })

        return events

    # ----------------------------------------------------------------
    # Orchestrator
    # ----------------------------------------------------------------

    def run_all_detectors(self, thresholds: Optional[Dict[str, float]] = None
                          ) -> List[dict]:
        """Run all three synchronicity detectors.

        Rate-limited to once per SYNC_MIN_INTERVAL_SECONDS.
        Returns list of new synchronicity events written to the database.
        """
        now = datetime.now(timezone.utc)

        # Rate limiting
        if self._last_sync_run:
            try:
                last = datetime.fromisoformat(self._last_sync_run)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < SYNC_MIN_INTERVAL_SECONDS:
                    return []
            except (ValueError, TypeError):
                pass

        self._last_sync_run = now.isoformat()

        if thresholds is None:
            try:
                from app.meta import get_adaptive_thresholds
                thresholds = get_adaptive_thresholds().get_all_values()
            except Exception:
                thresholds = {}

        # Compute coordinates
        coords = self.compute_glyph_coordinates()
        if not coords:
            return []

        # Run all three detectors
        all_events = []
        all_events.extend(self.detect_dormant_reactivation(coords, thresholds))
        all_events.extend(self.detect_cross_domain_bridges(thresholds))
        all_events.extend(self.detect_semantic_convergence(thresholds))

        # Update previous quadrants for next run
        self._prev_quadrants = {g: d["quadrant"] for g, d in coords.items()}

        # Write events to database
        written = []
        now_iso = now.isoformat()
        for evt in all_events:
            quadrant_map = {}
            for g in evt["glyphs"]:
                if g in coords:
                    quadrant_map[g] = coords[g]["quadrant"]

            cur = self._conn.execute(
                "INSERT INTO synchronicity_events "
                "(timestamp, event_type, glyphs, strength, details, quadrants) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now_iso, evt["event_type"],
                 json.dumps(evt["glyphs"], ensure_ascii=False),
                 evt["strength"],
                 json.dumps(evt.get("details", {}), ensure_ascii=False),
                 json.dumps(quadrant_map, ensure_ascii=False)),
            )
            evt["id"] = cur.lastrowid
            written.append(evt)

        if written:
            self._conn.commit()

        return written

    # ----------------------------------------------------------------
    # Query Helpers
    # ----------------------------------------------------------------

    def get_recent_events(self, hours: float = 168.0) -> List[dict]:
        """Get synchronicity events from the past N hours."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            "SELECT id, timestamp, event_type, glyphs, strength, details, "
            "quadrants, lattice_angle, lattice_slots "
            "FROM synchronicity_events WHERE timestamp >= ? "
            "ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()

        return [
            {
                "id": r[0], "timestamp": r[1], "event_type": r[2],
                "glyphs": json.loads(r[3]) if r[3] else [],
                "strength": r[4],
                "details": json.loads(r[5]) if r[5] else {},
                "quadrants": json.loads(r[6]) if r[6] else {},
                "lattice_angle": r[7],
                "lattice_slots": json.loads(r[8]) if r[8] else {},
            }
            for r in rows
        ]


# ----------------------------
# Singleton
# ----------------------------
_phase_space: Optional[PhaseSpace] = None


def get_phase_space() -> PhaseSpace:
    """Get or create the singleton phase space engine."""
    global _phase_space
    if _phase_space is None:
        from app.trends import get_trend_engine
        _phase_space = PhaseSpace(get_trend_engine()._conn)
    return _phase_space
