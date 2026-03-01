# app/lattice.py
"""
Layer 3: Prime Ramsey Lattice for GPT-GU

Each synchronicity event is projected onto a circular lattice via PCA of
glyph co-activation patterns. At 30 prime scales (2..113), events are assigned
slots. Events sharing slots at multiple incommensurate primes are "resonant" —
structurally aligned in embedding space in ways invisible to tag-based clustering.
"""
from __future__ import annotations
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------
# Constants
# ----------------------------
PRIMES_30 = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
]
TWO_PI = 2 * math.pi
PCA_REFRESH_HOURS = 24.0            # recompute PCA every 24 hours
VOID_INNER_FRACTION = 0.30          # inner 30% of angle distribution = void
RESONANCE_LOOKBACK_HOURS = 168.0    # 1 week

from app.trends import TREND_WINDOW_HOURS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PrimeRamseyLattice:
    """Projects synchronicity events onto a prime-indexed circular lattice."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._pca_components: Optional[np.ndarray] = None  # (n_glyphs, 2)
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_glyph_index: Optional[Dict[str, int]] = None
        self._pca_last_computed: Optional[datetime] = None

    # ----------------------------------------------------------------
    # Co-Activation Matrix & PCA
    # ----------------------------------------------------------------

    def _build_coactivation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Build co-activation matrix from tag_events.

        Returns (glyph_list, matrix) where matrix is symmetric.
        """
        from app.trends import get_trend_engine
        pairs = get_trend_engine().get_coactivation_data()

        if not pairs:
            return [], np.array([])

        # Collect all glyphs
        all_glyphs = set()
        for ga, gb, _ in pairs:
            all_glyphs.add(ga)
            all_glyphs.add(gb)

        glyph_list = sorted(all_glyphs)
        n = len(glyph_list)
        if n < 2:
            return glyph_list, np.zeros((n, n))

        glyph_idx = {g: i for i, g in enumerate(glyph_list)}
        matrix = np.zeros((n, n), dtype=np.float64)

        for ga, gb, weight in pairs:
            i, j = glyph_idx[ga], glyph_idx[gb]
            matrix[i, j] += weight
            matrix[j, i] += weight

        return glyph_list, matrix

    def _compute_pca(self):
        """Compute first 2 principal components via SVD.

        Cached for PCA_REFRESH_HOURS.
        """
        now = datetime.now(timezone.utc)
        if (self._pca_last_computed is not None
                and (now - self._pca_last_computed).total_seconds()
                < PCA_REFRESH_HOURS * 3600
                and self._pca_components is not None):
            return

        glyph_list, matrix = self._build_coactivation_matrix()

        if len(glyph_list) < 3 or matrix.size == 0:
            # Degenerate case: not enough data for PCA
            self._pca_glyph_index = {g: i for i, g in enumerate(glyph_list)}
            self._pca_mean = np.zeros(len(glyph_list))
            self._pca_components = np.zeros((len(glyph_list), 2))
            self._pca_last_computed = now
            return

        # Center the matrix
        self._pca_mean = matrix.mean(axis=0)
        centered = matrix - self._pca_mean

        # SVD: keep first 2 components
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Projection matrix: first 2 right-singular vectors
            self._pca_components = Vt[:2].T  # shape: (n_glyphs, 2)
        except np.linalg.LinAlgError:
            self._pca_components = np.zeros((len(glyph_list), 2))

        self._pca_glyph_index = {g: i for i, g in enumerate(glyph_list)}
        self._pca_last_computed = now

    # ----------------------------------------------------------------
    # Event Projection
    # ----------------------------------------------------------------

    def compute_event_angle(self, glyphs: List[str]) -> float:
        """Project a set of glyphs to an angle in [0, 2pi).

        1. Build centroid vector from the glyphs' co-activation profiles.
        2. Project to 2D via PCA.
        3. Convert to angle via atan2.
        """
        self._compute_pca()

        if (self._pca_components is None
                or self._pca_glyph_index is None
                or len(self._pca_glyph_index) == 0):
            return 0.0

        n = len(self._pca_glyph_index)
        centroid = np.zeros(n, dtype=np.float64)
        count = 0

        for g in glyphs:
            idx = self._pca_glyph_index.get(g)
            if idx is not None:
                centroid[idx] = 1.0
                count += 1

        if count == 0:
            return 0.0

        # Center
        centered = centroid - self._pca_mean
        # Project to 2D
        point_2d = centered @ self._pca_components  # shape: (2,)

        x, y = float(point_2d[0]), float(point_2d[1])
        theta = math.atan2(y, x) % TWO_PI
        return theta

    # ----------------------------------------------------------------
    # Slot Assignment
    # ----------------------------------------------------------------

    @staticmethod
    def assign_slots(theta: float) -> Dict[int, int]:
        """For each of 30 primes, compute slot: floor(theta * p / 2pi) mod p."""
        slots = {}
        for p in PRIMES_30:
            slot = int(math.floor(theta * p / TWO_PI)) % p
            slots[p] = slot
        return slots

    # ----------------------------------------------------------------
    # Resonance Detection
    # ----------------------------------------------------------------

    def detect_resonances(self, new_event_id: int,
                          min_shared: Optional[int] = None) -> List[dict]:
        """Compare a new synchronicity event against recent events on the lattice.

        Events sharing slots at >= min_shared primes are "resonant."
        """
        if min_shared is None:
            try:
                from app.meta import get_adaptive_thresholds
                min_shared = int(get_adaptive_thresholds().get("min_resonance_primes"))
            except Exception:
                min_shared = 5

        # Get the new event
        row = self._conn.execute(
            "SELECT glyphs, lattice_angle, lattice_slots FROM synchronicity_events "
            "WHERE id = ?",
            (new_event_id,),
        ).fetchone()

        if not row:
            return []

        glyphs_json, angle, slots_json = row

        # Compute angle and slots if not already set
        if angle is None or slots_json is None:
            glyphs = json.loads(glyphs_json) if glyphs_json else []
            angle = self.compute_event_angle(glyphs)
            slots = self.assign_slots(angle)
            self._conn.execute(
                "UPDATE synchronicity_events "
                "SET lattice_angle = ?, lattice_slots = ? WHERE id = ?",
                (angle, json.dumps(slots), new_event_id),
            )
            self._conn.commit()
        else:
            slots = json.loads(slots_json) if slots_json else {}

        if not slots:
            return []

        # Get recent events with lattice data
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=RESONANCE_LOOKBACK_HOURS)).isoformat()
        recent_rows = self._conn.execute(
            "SELECT id, lattice_slots FROM synchronicity_events "
            "WHERE id != ? AND timestamp >= ? AND lattice_slots IS NOT NULL",
            (new_event_id, cutoff),
        ).fetchall()

        resonances = []
        for other_id, other_slots_json in recent_rows:
            try:
                other_slots = json.loads(other_slots_json) if other_slots_json else {}
            except (json.JSONDecodeError, TypeError):
                continue

            # Find shared primes (same slot at same prime)
            shared = []
            for p in PRIMES_30:
                sp = str(p)
                slot_new = slots.get(sp, slots.get(p))
                slot_other = other_slots.get(sp, other_slots.get(p))
                if slot_new is not None and slot_other is not None:
                    if int(slot_new) == int(slot_other):
                        shared.append(p)

            if len(shared) < min_shared:
                continue

            # Compute resonance strength
            strength = sum(math.log(p) for p in shared)
            chance = math.exp(-strength)

            now = _now_iso()
            cur = self._conn.execute(
                "INSERT INTO resonances "
                "(timestamp, event_a_id, event_b_id, shared_primes, "
                " resonance_strength, chance) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (now, new_event_id, other_id,
                 json.dumps(shared), round(strength, 4), chance),
            )
            resonances.append({
                "id": cur.lastrowid,
                "event_a_id": new_event_id,
                "event_b_id": other_id,
                "shared_primes": shared,
                "resonance_strength": round(strength, 4),
                "chance": chance,
            })

        if resonances:
            self._conn.commit()

        return resonances

    # ----------------------------------------------------------------
    # Void Profiling
    # ----------------------------------------------------------------

    def compute_void_profile(self) -> dict:
        """Compute the void center and edge themes.

        1. Circular mean of all recent event angles.
        2. Inner 30% by angular distance = void region.
        3. Edge themes at the boundary.
        """
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=RESONANCE_LOOKBACK_HOURS)).isoformat()

        rows = self._conn.execute(
            "SELECT id, glyphs, lattice_angle FROM synchronicity_events "
            "WHERE timestamp >= ? AND lattice_angle IS NOT NULL",
            (cutoff,),
        ).fetchall()

        if not rows:
            return {"void_center": None, "edge_themes": [], "event_count": 0}

        angles = []
        events = []
        for eid, glyphs_json, angle in rows:
            angles.append(angle)
            events.append({
                "id": eid,
                "glyphs": json.loads(glyphs_json) if glyphs_json else [],
                "angle": angle,
            })

        # Circular mean
        sin_sum = sum(math.sin(a) for a in angles)
        cos_sum = sum(math.cos(a) for a in angles)
        circ_mean = math.atan2(sin_sum / len(angles),
                               cos_sum / len(angles)) % TWO_PI

        # Angular distances from mean
        for evt in events:
            diff = abs(evt["angle"] - circ_mean)
            evt["angular_distance"] = min(diff, TWO_PI - diff)

        events.sort(key=lambda e: e["angular_distance"])

        # Inner 30% = void region
        void_boundary = int(len(events) * VOID_INNER_FRACTION)
        void_events = events[:void_boundary] if void_boundary > 0 else []

        # Edge themes: events at the 25th-35th percentile of distance
        edge_lo = int(len(events) * 0.25)
        edge_hi = int(len(events) * 0.35)
        edge_events = events[edge_lo:edge_hi]

        # Collect edge glyphs
        edge_glyphs = set()
        for evt in edge_events:
            edge_glyphs.update(evt["glyphs"])

        return {
            "void_center": round(circ_mean, 4),
            "void_event_count": len(void_events),
            "edge_themes": list(edge_glyphs),
            "edge_event_count": len(edge_events),
            "total_events": len(events),
        }

    # ----------------------------------------------------------------
    # Query Helpers
    # ----------------------------------------------------------------

    def get_recent_resonances(self, hours: float = 168.0) -> List[dict]:
        """Get recent resonances."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            "SELECT id, timestamp, event_a_id, event_b_id, shared_primes, "
            "resonance_strength, chance "
            "FROM resonances WHERE timestamp >= ? ORDER BY resonance_strength DESC",
            (cutoff,),
        ).fetchall()

        return [
            {
                "id": r[0], "timestamp": r[1],
                "event_a_id": r[2], "event_b_id": r[3],
                "shared_primes": json.loads(r[4]) if r[4] else [],
                "resonance_strength": r[5], "chance": r[6],
            }
            for r in rows
        ]


# ----------------------------
# Singleton
# ----------------------------
_lattice: Optional[PrimeRamseyLattice] = None


def get_lattice() -> PrimeRamseyLattice:
    """Get or create the singleton lattice."""
    global _lattice
    if _lattice is None:
        from app.trends import get_trend_engine
        _lattice = PrimeRamseyLattice(get_trend_engine()._conn)
    return _lattice
