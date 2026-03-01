# app/glyph_network.py
"""
Layer 5: Glyph Ramsey Network for GPT-GU

A Chicory-inspired associative memory network where glyphs form the atomic
units of association. Unlike Chicory's letter network where letters are
meaningless and gain meaning through co-occurrence, glyphs are inherently
meaningful — each carries a named concept and pair relationships.

The Prime Ramsey Lattice is the primary topology:
  1. Each glyph is projected onto a circle via PCA of co-activation data.
  2. At 30 prime scales (2..113), each glyph gets slot assignments.
  3. Glyph pairs sharing slots at >= N primes are structurally connected.
  4. A relational tensor enriches each edge with co-occurrence, semantic,
     semiotic, and synchronicity signals.
  5. Deep retrieval traverses the Ramsey network to discover memories
     through chains of structural resonance.
"""
from __future__ import annotations
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
PCA_REFRESH_HOURS = 24.0

# Ramsey edge detection
DEFAULT_MIN_SHARED_PRIMES = 5

# Tensor composite weights (Ramsey-dominant)
W_RAMSEY = 0.30
W_COOCCURRENCE = 0.20
W_SEMANTIC = 0.20
W_SEMIOTIC = 0.15           # split: 0.075 forward + 0.075 reverse
W_SYNCHRONICITY = 0.15

# Deep retrieval
DEEP_RETRIEVAL_DECAY = 0.7
EXPLORATION_BIAS_RATE = 0.15
DEFAULT_DEEP_MAX_DEPTH = 3
DEFAULT_DEEP_PER_LEVEL_K = 5

# Association retrieval
DEFAULT_ASSOCIATION_THRESHOLD = 0.05

# ----------------------------
# Schema (extends trend_data.db)
# ----------------------------
_GLYPH_NETWORK_SCHEMA = """
-- L5: Glyph lattice positions
CREATE TABLE IF NOT EXISTS glyph_positions (
    glyph TEXT PRIMARY KEY,
    angle REAL NOT NULL,
    prime_slots TEXT NOT NULL,
    last_computed TEXT NOT NULL
);

-- L5: Glyph-to-Memory inverted index
CREATE TABLE IF NOT EXISTS glyph_memory_index (
    glyph TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    UNIQUE(glyph, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_gmi_glyph ON glyph_memory_index(glyph);
CREATE INDEX IF NOT EXISTS idx_gmi_memory ON glyph_memory_index(memory_id);

-- L5: Glyph edges (Ramsey resonance + tensor signals)
CREATE TABLE IF NOT EXISTS glyph_edges (
    glyph_a TEXT NOT NULL,
    glyph_b TEXT NOT NULL,
    ramsey_strength REAL NOT NULL DEFAULT 0.0,
    shared_primes TEXT,
    cooccurrence_strength REAL NOT NULL DEFAULT 0.0,
    synchronicity_strength REAL NOT NULL DEFAULT 0.0,
    semantic_strength REAL NOT NULL DEFAULT 0.0,
    semiotic_forward REAL NOT NULL DEFAULT 0.0,
    semiotic_reverse REAL NOT NULL DEFAULT 0.0,
    composite REAL NOT NULL DEFAULT 0.0,
    last_updated TEXT NOT NULL,
    UNIQUE(glyph_a, glyph_b)
);
CREATE INDEX IF NOT EXISTS idx_ge_a ON glyph_edges(glyph_a);
CREATE INDEX IF NOT EXISTS idx_ge_b ON glyph_edges(glyph_b);
CREATE INDEX IF NOT EXISTS idx_ge_composite ON glyph_edges(composite DESC);

-- L5: Deep retrieval event log
CREATE TABLE IF NOT EXISTS deep_retrieval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    seed_glyphs TEXT NOT NULL,
    max_depth INTEGER NOT NULL,
    total_results INTEGER NOT NULL,
    levels_explored INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dre_ts ON deep_retrieval_events(timestamp);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _compute_composite(ramsey: float, cooc: float, semantic: float,
                       semiotic_fwd: float, semiotic_rev: float,
                       sync: float) -> float:
    semiotic_avg = (semiotic_fwd + semiotic_rev) / 2.0
    raw = (W_RAMSEY * ramsey
           + W_COOCCURRENCE * cooc
           + W_SEMANTIC * semantic
           + W_SEMIOTIC * semiotic_avg
           + W_SYNCHRONICITY * sync)
    return min(1.0, max(0.0, raw))


class GlyphRamseyNetwork:
    """Glyph-native association network with Ramsey lattice as primary topology."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._ensure_schema()

        # PCA cache
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._pca_glyph_list: Optional[List[str]] = None
        self._pca_glyph_index: Optional[Dict[str, int]] = None
        self._pca_last_computed: Optional[datetime] = None

        # Edge cache for fast traversal
        self._edge_cache: Dict[str, Dict[str, float]] = {}
        self._cache_valid = False

    def _ensure_schema(self):
        self._conn.executescript(_GLYPH_NETWORK_SCHEMA)
        self._conn.commit()

    # ================================================================
    # 1. Ramsey Lattice for Individual Glyphs
    # ================================================================

    def _build_coactivation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Build co-activation matrix from tag_events.

        Each glyph's row is its co-activation feature vector.
        Falls back to training data co-occurrences when live
        co-activation data is too sparse for meaningful PCA.
        """
        from app.trends import get_trend_engine, TREND_WINDOW_HOURS
        pairs = get_trend_engine().get_coactivation_data(TREND_WINDOW_HOURS)

        # Fallback: use training co-occurrences if live data is sparse
        if len(pairs) < 10:
            try:
                from app.cross_reference import get_cross_reference_engine
                xref = get_cross_reference_engine()
                seen = set()
                fallback_pairs = []
                for ga, counter in xref.co_occurrences.items():
                    for gb, count in counter.items():
                        pair = tuple(sorted([ga, gb]))
                        if pair not in seen:
                            seen.add(pair)
                            fallback_pairs.append(
                                (pair[0], pair[1], float(count)))
                if fallback_pairs:
                    pairs = fallback_pairs
            except Exception:
                pass

        if not pairs:
            return [], np.array([])

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
        """Compute PCA projection for all glyphs. Cached for 24h."""
        now = datetime.now(timezone.utc)
        if (self._pca_last_computed is not None
                and (now - self._pca_last_computed).total_seconds()
                < PCA_REFRESH_HOURS * 3600
                and self._pca_components is not None):
            return

        glyph_list, matrix = self._build_coactivation_matrix()

        if len(glyph_list) < 3 or matrix.size == 0:
            self._pca_glyph_list = glyph_list
            self._pca_glyph_index = {g: i for i, g in enumerate(glyph_list)}
            self._pca_mean = np.zeros(len(glyph_list))
            self._pca_components = np.zeros((len(glyph_list), 2))
            self._pca_last_computed = now
            return

        self._pca_mean = matrix.mean(axis=0)
        centered = matrix - self._pca_mean

        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            self._pca_components = Vt[:2].T  # (n_glyphs, 2)
        except np.linalg.LinAlgError:
            self._pca_components = np.zeros((len(glyph_list), 2))

        self._pca_glyph_list = glyph_list
        self._pca_glyph_index = {g: i for i, g in enumerate(glyph_list)}
        self._pca_last_computed = now

    def compute_glyph_angle(self, glyph: str) -> float:
        """Project a single glyph to an angle in [0, 2pi).

        Uses the glyph's row in the co-activation matrix projected via PCA.
        """
        self._compute_pca()

        if (self._pca_components is None
                or self._pca_glyph_index is None
                or len(self._pca_glyph_index) == 0):
            return 0.0

        idx = self._pca_glyph_index.get(glyph)
        if idx is None:
            return 0.0

        n = len(self._pca_glyph_index)
        feature = np.zeros(n, dtype=np.float64)
        feature[idx] = 1.0
        centered = feature - self._pca_mean
        point_2d = centered @ self._pca_components

        x, y = float(point_2d[0]), float(point_2d[1])
        return math.atan2(y, x) % TWO_PI

    @staticmethod
    def assign_slots(theta: float) -> Dict[int, int]:
        """For each of 30 primes, compute slot: floor(theta * p / 2pi) mod p."""
        slots = {}
        for p in PRIMES_30:
            slot = int(math.floor(theta * p / TWO_PI)) % p
            slots[p] = slot
        return slots

    def compute_all_glyph_positions(self):
        """Compute lattice positions for ALL glyphs with co-activation data.

        Stores results in glyph_positions table. This is the core Ramsey
        lattice computation — each glyph gets an angle and prime slots.
        """
        self._compute_pca()

        if not self._pca_glyph_list:
            return

        now = _now_iso()
        rows = []
        for glyph in self._pca_glyph_list:
            angle = self.compute_glyph_angle(glyph)
            slots = self.assign_slots(angle)
            rows.append((glyph, angle, json.dumps(slots), now))

        self._conn.executemany(
            "INSERT OR REPLACE INTO glyph_positions "
            "(glyph, angle, prime_slots, last_computed) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def detect_ramsey_edges(self, min_shared: int = DEFAULT_MIN_SHARED_PRIMES):
        """Detect Ramsey edges: glyph pairs sharing slots at >= min_shared primes.

        Compares all glyph position pairs and creates edges where structural
        resonance is detected. Edge strength = sum(log(p)) for shared primes.
        """
        rows = self._conn.execute(
            "SELECT glyph, prime_slots FROM glyph_positions"
        ).fetchall()

        if len(rows) < 2:
            return

        # Parse positions
        positions = []
        for glyph, slots_json in rows:
            slots = json.loads(slots_json) if slots_json else {}
            positions.append((glyph, slots))

        now = _now_iso()
        edges_to_upsert = []

        for i, (ga, slots_a) in enumerate(positions):
            for gb, slots_b in positions[i + 1:]:
                shared = []
                for p in PRIMES_30:
                    sp = str(p)
                    slot_a = slots_a.get(sp, slots_a.get(p))
                    slot_b = slots_b.get(sp, slots_b.get(p))
                    if slot_a is not None and slot_b is not None:
                        if int(slot_a) == int(slot_b):
                            shared.append(p)

                if len(shared) >= min_shared:
                    strength = sum(math.log(p) for p in shared)
                    # Normalize to [0, 1] via sigmoid
                    norm_strength = _sigmoid(strength - 10.0)  # center around ~10
                    pair = tuple(sorted([ga, gb]))
                    edges_to_upsert.append((
                        pair[0], pair[1], norm_strength,
                        json.dumps(shared), now,
                    ))

        if edges_to_upsert:
            for ga, gb, strength, shared_json, ts in edges_to_upsert:
                self._conn.execute(
                    "INSERT INTO glyph_edges "
                    "(glyph_a, glyph_b, ramsey_strength, shared_primes, "
                    " last_updated) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(glyph_a, glyph_b) DO UPDATE SET "
                    "ramsey_strength = excluded.ramsey_strength, "
                    "shared_primes = excluded.shared_primes, "
                    "last_updated = excluded.last_updated",
                    (ga, gb, strength, shared_json, ts),
                )
            self._conn.commit()
            self._invalidate_cache()

    # ================================================================
    # 2. Relational Tensor
    # ================================================================

    def _upsert_edge(self, glyph_a: str, glyph_b: str, now: str, **signals):
        """Insert or update specific signal columns for a glyph edge.

        Automatically recomputes composite after update.
        """
        pair = tuple(sorted([glyph_a, glyph_b]))

        # Check if edge exists
        row = self._conn.execute(
            "SELECT ramsey_strength, cooccurrence_strength, "
            "synchronicity_strength, semantic_strength, "
            "semiotic_forward, semiotic_reverse "
            "FROM glyph_edges WHERE glyph_a = ? AND glyph_b = ?",
            pair,
        ).fetchone()

        if row:
            current = {
                "ramsey_strength": row[0],
                "cooccurrence_strength": row[1],
                "synchronicity_strength": row[2],
                "semantic_strength": row[3],
                "semiotic_forward": row[4],
                "semiotic_reverse": row[5],
            }
            for k, v in signals.items():
                if k in current:
                    current[k] = v

            composite = _compute_composite(
                current["ramsey_strength"],
                current["cooccurrence_strength"],
                current["semantic_strength"],
                current["semiotic_forward"],
                current["semiotic_reverse"],
                current["synchronicity_strength"],
            )

            sets = ", ".join(f"{k} = ?" for k in signals if k in current)
            vals = [signals[k] for k in signals if k in current]
            sets += ", composite = ?, last_updated = ?"
            vals.extend([composite, now])

            self._conn.execute(
                f"UPDATE glyph_edges SET {sets} "
                "WHERE glyph_a = ? AND glyph_b = ?",
                vals + [pair[0], pair[1]],
            )
        else:
            defaults = {
                "ramsey_strength": 0.0,
                "cooccurrence_strength": 0.0,
                "synchronicity_strength": 0.0,
                "semantic_strength": 0.0,
                "semiotic_forward": 0.0,
                "semiotic_reverse": 0.0,
            }
            defaults.update({k: v for k, v in signals.items() if k in defaults})

            composite = _compute_composite(
                defaults["ramsey_strength"],
                defaults["cooccurrence_strength"],
                defaults["semantic_strength"],
                defaults["semiotic_forward"],
                defaults["semiotic_reverse"],
                defaults["synchronicity_strength"],
            )

            self._conn.execute(
                "INSERT INTO glyph_edges "
                "(glyph_a, glyph_b, ramsey_strength, shared_primes, "
                " cooccurrence_strength, synchronicity_strength, "
                " semantic_strength, semiotic_forward, semiotic_reverse, "
                " composite, last_updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pair[0], pair[1],
                 defaults["ramsey_strength"], None,
                 defaults["cooccurrence_strength"],
                 defaults["synchronicity_strength"],
                 defaults["semantic_strength"],
                 defaults["semiotic_forward"],
                 defaults["semiotic_reverse"],
                 composite, now),
            )

    def seed_tensor(self):
        """One-time seeding from existing data sources.

        1. Compute glyph positions and Ramsey edges
        2. Enrich edges from cross_reference and PAIR_TO_TEXT
        3. Process existing memories for cooccurrence + semiotic
        """
        now = _now_iso()

        # Step 1: Ramsey lattice
        self.compute_all_glyph_positions()
        self.detect_ramsey_edges()

        # Step 2: Cross-reference co-occurrences
        try:
            from app.cross_reference import get_cross_reference_engine
            xref = get_cross_reference_engine()

            # Co-occurrences -> cooccurrence_strength
            seen = set()
            for glyph_a, counter in xref.co_occurrences.items():
                for glyph_b, count in counter.items():
                    pair = tuple(sorted([glyph_a, glyph_b]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    cooc = _sigmoid(math.log(1 + count) - 2.0)
                    self._upsert_edge(pair[0], pair[1], now,
                                      cooccurrence_strength=cooc)

            # Oppositions -> semantic_strength
            for glyph_a, opps in xref.oppositions.items():
                for glyph_b in opps:
                    self._upsert_edge(glyph_a, glyph_b, now,
                                      semantic_strength=0.6)

            # Transformations -> semiotic
            for glyph_a, targets in xref.transformations.items():
                for glyph_b in targets:
                    pair = tuple(sorted([glyph_a, glyph_b]))
                    if pair[0] == glyph_a:
                        self._upsert_edge(pair[0], pair[1], now,
                                          semiotic_forward=0.7)
                    else:
                        self._upsert_edge(pair[0], pair[1], now,
                                          semiotic_reverse=0.7)
        except Exception:
            pass

        # Step 3: PAIR_TO_TEXT -> semantic_strength
        try:
            from app.guardrails import PAIR_TO_TEXT
            for (ga, gb) in PAIR_TO_TEXT:
                self._upsert_edge(ga, gb, now, semantic_strength=0.8)
        except Exception:
            pass

        # Step 4: Existing memories -> cooccurrence + semiotic
        try:
            from app.memory import get_memory_engine
            engine = get_memory_engine()
            for mem in engine.memories:
                glyphs = mem.get("glyphs", [])
                if len(glyphs) >= 2:
                    self._update_cooccurrence_batch(glyphs, now)
                    self._update_semiotic_batch(glyphs, now)
        except Exception:
            pass

        self._conn.commit()
        self._invalidate_cache()

    def update_cooccurrence(self, glyphs: List[str]):
        """Increment cooccurrence for all pairs. Called on store/recall."""
        if len(glyphs) < 2:
            return
        now = _now_iso()
        self._update_cooccurrence_batch(glyphs, now)
        self._conn.commit()
        self._invalidate_cache()

    def _update_cooccurrence_batch(self, glyphs: List[str], now: str):
        """Internal: update cooccurrence without commit."""
        for i, ga in enumerate(glyphs):
            for gb in glyphs[i + 1:]:
                pair = tuple(sorted([ga, gb]))
                row = self._conn.execute(
                    "SELECT cooccurrence_strength FROM glyph_edges "
                    "WHERE glyph_a = ? AND glyph_b = ?",
                    pair,
                ).fetchone()
                current = row[0] if row else 0.0
                new_val = min(1.0, current + 0.01)
                self._upsert_edge(pair[0], pair[1], now,
                                  cooccurrence_strength=new_val)

    def update_synchronicity(self, glyph_a: str, glyph_b: str,
                             strength: float):
        """Update synchronicity edge from L3 events."""
        now = _now_iso()
        pair = tuple(sorted([glyph_a, glyph_b]))
        row = self._conn.execute(
            "SELECT synchronicity_strength FROM glyph_edges "
            "WHERE glyph_a = ? AND glyph_b = ?",
            pair,
        ).fetchone()
        current = row[0] if row else 0.0
        # Blend: running average weighted toward new signal
        new_val = min(1.0, current * 0.7 + _sigmoid(strength - 2.0) * 0.3)
        self._upsert_edge(pair[0], pair[1], now,
                          synchronicity_strength=new_val)
        self._conn.commit()
        self._invalidate_cache()

    def update_semiotic(self, glyphs: List[str]):
        """Update directional probabilities from formula ordering."""
        if len(glyphs) < 2:
            return
        now = _now_iso()
        self._update_semiotic_batch(glyphs, now)
        self._conn.commit()
        self._invalidate_cache()

    def _update_semiotic_batch(self, glyphs: List[str], now: str):
        """Internal: update semiotic from ordered glyph list."""
        for i in range(len(glyphs) - 1):
            ga, gb = glyphs[i], glyphs[i + 1]
            pair = tuple(sorted([ga, gb]))
            row = self._conn.execute(
                "SELECT semiotic_forward, semiotic_reverse FROM glyph_edges "
                "WHERE glyph_a = ? AND glyph_b = ?",
                pair,
            ).fetchone()

            fwd = row[0] if row else 0.0
            rev = row[1] if row else 0.0

            # If ga < gb (natural order), this is a forward observation
            if ga == pair[0]:
                fwd = min(1.0, fwd + 0.02)
            else:
                rev = min(1.0, rev + 0.02)

            self._upsert_edge(pair[0], pair[1], now,
                              semiotic_forward=fwd, semiotic_reverse=rev)

    def get_edge(self, glyph_a: str, glyph_b: str) -> Optional[dict]:
        """Get full tensor data for a glyph pair."""
        pair = tuple(sorted([glyph_a, glyph_b]))
        row = self._conn.execute(
            "SELECT glyph_a, glyph_b, ramsey_strength, shared_primes, "
            "cooccurrence_strength, synchronicity_strength, "
            "semantic_strength, semiotic_forward, semiotic_reverse, "
            "composite, last_updated "
            "FROM glyph_edges WHERE glyph_a = ? AND glyph_b = ?",
            pair,
        ).fetchone()

        if not row:
            return None

        return {
            "glyph_a": row[0], "glyph_b": row[1],
            "ramsey_strength": row[2],
            "shared_primes": json.loads(row[3]) if row[3] else [],
            "cooccurrence_strength": row[4],
            "synchronicity_strength": row[5],
            "semantic_strength": row[6],
            "semiotic_forward": row[7],
            "semiotic_reverse": row[8],
            "composite": row[9],
            "last_updated": row[10],
        }

    def get_neighbors(self, glyph: str, top_n: int = 10,
                      min_composite: float = DEFAULT_ASSOCIATION_THRESHOLD
                      ) -> List[dict]:
        """Get strongest connected glyphs by composite tensor weight."""
        if self._cache_valid and glyph in self._edge_cache:
            cached = self._edge_cache[glyph]
            items = sorted(cached.items(), key=lambda x: -x[1])
            return [{"glyph": g, "composite": c}
                    for g, c in items[:top_n] if c >= min_composite]

        rows = self._conn.execute(
            "SELECT glyph_b, composite FROM glyph_edges "
            "WHERE glyph_a = ? AND composite >= ? "
            "UNION ALL "
            "SELECT glyph_a, composite FROM glyph_edges "
            "WHERE glyph_b = ? AND composite >= ? "
            "ORDER BY composite DESC LIMIT ?",
            (glyph, min_composite, glyph, min_composite, top_n),
        ).fetchall()

        return [{"glyph": r[0], "composite": r[1]} for r in rows]

    def _invalidate_cache(self):
        self._cache_valid = False
        self._edge_cache.clear()

    def _warm_cache(self):
        """Load all edges into memory for fast traversal."""
        if self._cache_valid:
            return

        self._edge_cache.clear()
        rows = self._conn.execute(
            "SELECT glyph_a, glyph_b, composite FROM glyph_edges "
            "WHERE composite >= ?",
            (DEFAULT_ASSOCIATION_THRESHOLD,),
        ).fetchall()

        for ga, gb, comp in rows:
            self._edge_cache.setdefault(ga, {})[gb] = comp
            self._edge_cache.setdefault(gb, {})[ga] = comp

        self._cache_valid = True

    # ================================================================
    # 3. Inverted Index
    # ================================================================

    def rebuild_index(self, memories: List[dict]):
        """Full rebuild of glyph-memory index from memory list."""
        self._conn.execute("DELETE FROM glyph_memory_index")
        rows = []
        for mem in memories:
            if mem.get("is_archived", False):
                continue
            mem_id = mem["id"]
            for pos, glyph in enumerate(mem.get("glyphs", [])):
                rows.append((glyph, mem_id, pos))

        if rows:
            self._conn.executemany(
                "INSERT OR IGNORE INTO glyph_memory_index "
                "(glyph, memory_id, position) VALUES (?, ?, ?)",
                rows,
            )
        self._conn.commit()

    def index_memory(self, memory_id: str, glyphs: List[str]):
        """Add a single memory to the inverted index."""
        for pos, glyph in enumerate(glyphs):
            self._conn.execute(
                "INSERT OR IGNORE INTO glyph_memory_index "
                "(glyph, memory_id, position) VALUES (?, ?, ?)",
                (glyph, memory_id, pos),
            )
        self._conn.commit()

    def remove_memory(self, memory_id: str):
        """Remove a memory from the index."""
        self._conn.execute(
            "DELETE FROM glyph_memory_index WHERE memory_id = ?",
            (memory_id,),
        )
        self._conn.commit()

    def get_memories_for_glyph(self, glyph: str) -> List[str]:
        """Return all memory_ids containing this glyph."""
        rows = self._conn.execute(
            "SELECT memory_id FROM glyph_memory_index WHERE glyph = ?",
            (glyph,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_memories_for_glyphs(self, glyphs: List[str]) -> Dict[str, List[str]]:
        """Batch lookup: {glyph: [memory_ids]}."""
        result = {}
        for glyph in glyphs:
            result[glyph] = self.get_memories_for_glyph(glyph)
        return result

    def get_shared_memories(self, glyph_a: str, glyph_b: str) -> List[str]:
        """Memory IDs containing both glyphs."""
        rows = self._conn.execute(
            "SELECT a.memory_id FROM glyph_memory_index a "
            "INNER JOIN glyph_memory_index b "
            "ON a.memory_id = b.memory_id "
            "WHERE a.glyph = ? AND b.glyph = ?",
            (glyph_a, glyph_b),
        ).fetchall()
        return [r[0] for r in rows]

    # ================================================================
    # 4. Association Retrieval
    # ================================================================

    def recall_by_association(self, query_glyphs: List[str],
                              memories: List[dict],
                              top_n: int = 5,
                              heat_gate: bool = False,
                              current_heat: Optional[Dict[str, float]] = None,
                              similarity_threshold: float = 0.6,
                              ) -> List[dict]:
        """Retrieve memories scored by Ramsey network association.

        For each candidate memory:
          1. Find glyph overlap with query
          2. Sum tensor composite weights between overlapping glyphs
          3. Add neighbor contributions (glyphs connected via Ramsey edges)
        """
        if not query_glyphs:
            return []

        self._warm_cache()

        # Build lookup: memory_id -> memory
        mem_by_id = {m["id"]: m for m in memories
                     if not m.get("is_archived", False)}

        # Find candidate memory IDs via inverted index
        query_set = set(query_glyphs)
        candidate_ids: Set[str] = set()

        # Direct: memories containing query glyphs
        for g in query_glyphs:
            for mid in self.get_memories_for_glyph(g):
                candidate_ids.add(mid)

        # Neighbors: memories containing Ramsey neighbors of query glyphs
        neighbor_glyphs = set()
        for g in query_glyphs:
            neighbors = self.get_neighbors(g, top_n=5)
            for nb in neighbors:
                ng = nb["glyph"]
                neighbor_glyphs.add(ng)
                for mid in self.get_memories_for_glyph(ng):
                    candidate_ids.add(mid)

        scored = []
        for mid in candidate_ids:
            mem = mem_by_id.get(mid)
            if mem is None:
                continue

            # Optional heat gate
            if heat_gate and current_heat:
                from app.memory import GlyphMemory
                sim = GlyphMemory._cosine_similarity(
                    None, mem.get("heat", {}), current_heat)
                if sim < similarity_threshold:
                    continue

            mem_glyphs = set(mem.get("glyphs", []))
            score = 0.0
            shared = []

            for mg in mem_glyphs:
                if mg in query_set:
                    # Direct overlap: full credit
                    score += 1.0
                    shared.append(mg)
                else:
                    # Indirect: tensor connection to query glyphs
                    for qg in query_glyphs:
                        comp = self._edge_cache.get(mg, {}).get(qg, 0.0)
                        if comp > 0:
                            score += comp * 0.5  # half credit for indirect

            if score <= 0:
                continue

            # Weight by salience
            composite_sal = mem.get("salience_composite",
                                    mem.get("store_salience", 0.5))
            final_score = score * (composite_sal + 0.1)

            scored.append({
                **mem,
                "_association_score": round(score, 4),
                "_final_score": round(final_score, 4),
                "_shared_glyphs": shared,
            })

        scored.sort(key=lambda x: x["_final_score"], reverse=True)
        return scored[:top_n]

    # ================================================================
    # 5. Deep Retrieval (Association Chains)
    # ================================================================

    def deep_recall(self, seed_glyphs: List[str],
                    memories: List[dict],
                    max_depth: int = DEFAULT_DEEP_MAX_DEPTH,
                    per_level_k: int = DEFAULT_DEEP_PER_LEVEL_K,
                    current_heat: Optional[Dict[str, float]] = None,
                    ) -> List[dict]:
        """Recursive association-chain retrieval via Ramsey network.

        Level 0: Direct glyph overlap with seed.
        Level 1+: Follow strongest Ramsey edges to new glyphs, find memories.
        Deeper levels favor older memories (exploration bias).
        """
        if not seed_glyphs:
            return []

        self._warm_cache()

        mem_by_id = {m["id"]: m for m in memories
                     if not m.get("is_archived", False)}

        results = []
        seen_memory_ids: Set[str] = set()
        seen_glyphs: Set[str] = set(seed_glyphs)
        current_glyphs: Set[str] = set(seed_glyphs)

        for depth in range(max_depth + 1):
            # Find candidate memories via inverted index
            candidate_ids: Set[str] = set()
            for g in current_glyphs:
                for mid in self.get_memories_for_glyph(g):
                    if mid not in seen_memory_ids:
                        candidate_ids.add(mid)

            # Score candidates
            level_scored = []
            for mid in candidate_ids:
                mem = mem_by_id.get(mid)
                if mem is None:
                    continue

                mem_glyphs = set(mem.get("glyphs", []))
                score = 0.0
                shared = []

                for mg in mem_glyphs:
                    if mg in current_glyphs:
                        score += 1.0
                        shared.append(mg)
                    else:
                        for cg in current_glyphs:
                            comp = self._edge_cache.get(mg, {}).get(cg, 0.0)
                            if comp > 0:
                                score += comp * 0.5

                if score <= 0:
                    continue

                # Depth decay
                depth_factor = DEEP_RETRIEVAL_DECAY ** depth

                # Exploration bias: deeper = prefer older memories
                try:
                    ts = mem.get("timestamp", "")
                    mem_dt = datetime.fromisoformat(ts)
                    if mem_dt.tzinfo is None:
                        mem_dt = mem_dt.replace(tzinfo=timezone.utc)
                    age_hours = max(0, (datetime.now(timezone.utc) - mem_dt
                                        ).total_seconds() / 3600)
                except (ValueError, TypeError):
                    age_hours = 0.0

                age_bonus = 1.0 + (EXPLORATION_BIAS_RATE * depth
                                   * math.log(1 + age_hours / 168.0))

                final_score = score * depth_factor * age_bonus

                level_scored.append({
                    **mem,
                    "_depth": depth,
                    "_association_score": round(score, 4),
                    "_depth_factor": round(depth_factor, 4),
                    "_age_bonus": round(age_bonus, 4),
                    "_final_score": round(final_score, 4),
                    "_shared_glyphs": shared,
                    "_expansion_front": list(current_glyphs),
                })

            # Take top per_level_k
            level_scored.sort(key=lambda x: x["_final_score"], reverse=True)
            level_results = level_scored[:per_level_k]
            results.extend(level_results)

            for r in level_results:
                seen_memory_ids.add(r["id"])

            # Expand: follow strongest Ramsey edges
            if depth < max_depth:
                new_glyphs = self._expand_glyphs(
                    current_glyphs, seen_glyphs, per_level_k)
                if not new_glyphs:
                    break
                seen_glyphs.update(new_glyphs)
                current_glyphs = set(new_glyphs)

        # Log deep retrieval event
        try:
            self._conn.execute(
                "INSERT INTO deep_retrieval_events "
                "(timestamp, seed_glyphs, max_depth, total_results, "
                " levels_explored) VALUES (?, ?, ?, ?, ?)",
                (_now_iso(), json.dumps(seed_glyphs), max_depth,
                 len(results), min(max_depth + 1, depth + 1)),
            )
            self._conn.commit()
        except Exception:
            pass

        results.sort(key=lambda x: x["_final_score"], reverse=True)
        return results

    def _expand_glyphs(self, current_glyphs: Set[str],
                       seen_glyphs: Set[str],
                       per_level_k: int) -> List[str]:
        """Follow strongest Ramsey edges to discover new glyphs."""
        candidates: Dict[str, float] = {}

        for g in current_glyphs:
            neighbors = self._edge_cache.get(g, {})
            for ng, comp in neighbors.items():
                if ng not in seen_glyphs:
                    if ng not in candidates or comp > candidates[ng]:
                        candidates[ng] = comp

        sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])
        return [g for g, _ in sorted_cands[:per_level_k]]

    # ================================================================
    # 6. Network Visualization
    # ================================================================

    def export_network(self, min_composite: float = 0.05,
                       max_nodes: int = 100) -> dict:
        """Export glyph network as nodes + edges for visualization."""
        try:
            from app.guardrails import LEX_DICT2GLYPH
            glyph_to_name = {v: k for k, v in LEX_DICT2GLYPH.items()}
        except Exception:
            glyph_to_name = {}

        # Get positions
        pos_rows = self._conn.execute(
            "SELECT glyph, angle, prime_slots FROM glyph_positions"
        ).fetchall()

        glyph_set = set()
        nodes = []
        for glyph, angle, slots_json in pos_rows:
            if max_nodes and len(nodes) >= max_nodes:
                break
            glyph_set.add(glyph)
            mem_count = self._conn.execute(
                "SELECT COUNT(*) FROM glyph_memory_index WHERE glyph = ?",
                (glyph,),
            ).fetchone()[0]

            nodes.append({
                "glyph": glyph,
                "name": glyph_to_name.get(glyph, glyph),
                "angle": round(angle, 4),
                "prime_slots": json.loads(slots_json) if slots_json else {},
                "memory_count": mem_count,
            })

        # Get edges
        edge_rows = self._conn.execute(
            "SELECT glyph_a, glyph_b, ramsey_strength, shared_primes, "
            "cooccurrence_strength, synchronicity_strength, "
            "semantic_strength, semiotic_forward, semiotic_reverse, "
            "composite FROM glyph_edges WHERE composite >= ?",
            (min_composite,),
        ).fetchall()

        edges = []
        for r in edge_rows:
            if r[0] in glyph_set and r[1] in glyph_set:
                edges.append({
                    "source": r[0], "target": r[1],
                    "ramsey_strength": r[2],
                    "shared_primes": json.loads(r[3]) if r[3] else [],
                    "cooccurrence_strength": r[4],
                    "synchronicity_strength": r[5],
                    "semantic_strength": r[6],
                    "semiotic_forward": r[7],
                    "semiotic_reverse": r[8],
                    "composite": r[9],
                })

        # Stats
        n_nodes = len(nodes)
        n_edges = len(edges)
        avg_degree = (2 * n_edges / n_nodes) if n_nodes > 0 else 0
        max_possible = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
        density = n_edges / max_possible if max_possible > 0 else 0

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": n_nodes,
                "total_edges": n_edges,
                "avg_degree": round(avg_degree, 2),
                "density": round(density, 4),
            },
        }

    def get_glyph_profile(self, glyph: str) -> dict:
        """Complete profile of a glyph in the network."""
        try:
            from app.guardrails import LEX_DICT2GLYPH
            glyph_to_name = {v: k for k, v in LEX_DICT2GLYPH.items()}
        except Exception:
            glyph_to_name = {}

        # Position
        pos_row = self._conn.execute(
            "SELECT angle, prime_slots FROM glyph_positions WHERE glyph = ?",
            (glyph,),
        ).fetchone()

        # Neighbors
        neighbors = self.get_neighbors(glyph, top_n=20)
        for nb in neighbors:
            nb["name"] = glyph_to_name.get(nb["glyph"], nb["glyph"])

        # Memory count
        mem_ids = self.get_memories_for_glyph(glyph)

        return {
            "glyph": glyph,
            "name": glyph_to_name.get(glyph, glyph),
            "angle": round(pos_row[0], 4) if pos_row else None,
            "prime_slots": json.loads(pos_row[1]) if pos_row else {},
            "degree": len(neighbors),
            "neighbors": neighbors,
            "memory_count": len(mem_ids),
            "memory_ids": mem_ids[:20],
        }


# ----------------------------
# Singleton
# ----------------------------
_glyph_network: Optional[GlyphRamseyNetwork] = None


def get_glyph_network() -> GlyphRamseyNetwork:
    """Get or create the singleton glyph Ramsey network.

    Shares the same SQLite connection as TrendEngine.
    Seeds the tensor on first creation if the glyph_edges table is empty.
    """
    global _glyph_network
    if _glyph_network is None:
        from app.trends import get_trend_engine
        conn = get_trend_engine()._conn
        _glyph_network = GlyphRamseyNetwork(conn)

        # Seed tensor if empty
        row = conn.execute(
            "SELECT COUNT(*) FROM glyph_edges"
        ).fetchone()
        if row[0] == 0:
            _glyph_network.seed_tensor()

        # Rebuild inverted index from existing memories
        try:
            from app.memory import get_memory_engine
            engine = get_memory_engine()
            if engine.memories:
                _glyph_network.rebuild_index(engine.memories)
        except Exception:
            pass

    return _glyph_network
