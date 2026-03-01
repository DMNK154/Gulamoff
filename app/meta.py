# app/meta.py
"""
Layer 4: Meta-Pattern Analysis & Feedback for GPT-GU

AdaptiveThresholds: EMA-smoothed detection thresholds with burn-in support.
MetaAnalyzer: Clusters synchronicity events, tests significance, detects cross-domain themes.
FeedbackEngine: Creates emergent glyph links, boosts memory salience from discovered patterns.
"""
from __future__ import annotations
import json
import math
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Constants
# ----------------------------
EMA_ALPHA = 0.1                     # ~43 observations effective memory
BURN_IN_HOURS = 48.0
BURN_IN_MULTIPLIER = 1.5
ANALYSIS_INTERVAL_HOURS = 24.0
LOOKBACK_PERIODS = 7                # 7 * 24h = 168h lookback
SALIENCE_BOOST = 0.05

THRESHOLD_DEFAULTS = {
    "z_score_threshold": 2.0,
    "surprise_threshold": 3.0,
    "convergence_threshold": 0.7,
    "min_resonance_primes": 5,
    "base_rate_multiplier": 3.0,
    "clustering_jaccard_threshold": 0.7,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ================================================================
# Adaptive Thresholds
# ================================================================

class AdaptiveThresholds:
    """EMA-smoothed detection thresholds with burn-in support."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._seed_defaults()

    def _seed_defaults(self):
        """Insert default thresholds if they don't exist."""
        now = _now_iso()
        for name, default in THRESHOLD_DEFAULTS.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO adaptive_thresholds "
                "(name, current_value, default_value, last_updated) "
                "VALUES (?, ?, ?, ?)",
                (name, default, default, now),
            )
        self._conn.commit()

    def get(self, name: str) -> float:
        """Get threshold value. Returns value * 1.5 during burn-in."""
        row = self._conn.execute(
            "SELECT current_value, burn_in_until FROM adaptive_thresholds "
            "WHERE name = ?",
            (name,),
        ).fetchone()

        if row is None:
            return THRESHOLD_DEFAULTS.get(name, 1.0)

        value, burn_in_until = row
        if burn_in_until:
            try:
                until = datetime.fromisoformat(burn_in_until)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) < until:
                    return value * BURN_IN_MULTIPLIER
            except (ValueError, TypeError):
                pass

        return value

    def get_all(self) -> Dict[str, dict]:
        """Return all thresholds with metadata."""
        rows = self._conn.execute(
            "SELECT name, current_value, default_value, last_updated, burn_in_until "
            "FROM adaptive_thresholds"
        ).fetchall()

        result = {}
        now = datetime.now(timezone.utc)
        for name, current, default, updated, burn_until in rows:
            in_burn_in = False
            if burn_until:
                try:
                    until = datetime.fromisoformat(burn_until)
                    if until.tzinfo is None:
                        until = until.replace(tzinfo=timezone.utc)
                    in_burn_in = now < until
                except (ValueError, TypeError):
                    pass

            result[name] = {
                "current_value": current,
                "effective_value": current * BURN_IN_MULTIPLIER if in_burn_in else current,
                "default_value": default,
                "last_updated": updated,
                "in_burn_in": in_burn_in,
            }
        return result

    def get_all_values(self) -> Dict[str, float]:
        """Return all thresholds as {name: effective_value}."""
        all_t = self.get_all()
        return {name: data["effective_value"] for name, data in all_t.items()}

    def update(self, name: str, observed_value: float):
        """EMA update: new = alpha * observed + (1-alpha) * current."""
        row = self._conn.execute(
            "SELECT current_value FROM adaptive_thresholds WHERE name = ?",
            (name,),
        ).fetchone()

        if row is None:
            return

        current = row[0]
        new_value = EMA_ALPHA * observed_value + (1 - EMA_ALPHA) * current
        self._conn.execute(
            "UPDATE adaptive_thresholds SET current_value = ?, last_updated = ? "
            "WHERE name = ?",
            (round(new_value, 6), _now_iso(), name),
        )
        self._conn.commit()

    def enter_burn_in(self):
        """Set burn-in period for all thresholds."""
        until = (datetime.now(timezone.utc)
                 + timedelta(hours=BURN_IN_HOURS)).isoformat()
        self._conn.execute(
            "UPDATE adaptive_thresholds SET burn_in_until = ?",
            (until,),
        )
        self._conn.commit()


# ================================================================
# Meta Analyzer
# ================================================================

class MetaAnalyzer:
    """Clusters synchronicity events to discover recurring and cross-domain patterns."""

    def __init__(self, conn: sqlite3.Connection,
                 thresholds: AdaptiveThresholds):
        self._conn = conn
        self._thresholds = thresholds
        self._last_analysis: Optional[str] = None

    def run_analysis(self) -> List[dict]:
        """Run meta-analysis on synchronicity events from the past lookback window.

        Rate-limited to once per ANALYSIS_INTERVAL_HOURS.
        Returns list of new meta_patterns discovered.
        """
        now = datetime.now(timezone.utc)

        # Rate limiting
        if self._last_analysis:
            try:
                last = datetime.fromisoformat(self._last_analysis)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < ANALYSIS_INTERVAL_HOURS * 3600:
                    return []
            except (ValueError, TypeError):
                pass

        self._last_analysis = now.isoformat()

        # Fetch synchronicity events from lookback window
        lookback_hours = LOOKBACK_PERIODS * ANALYSIS_INTERVAL_HOURS
        cutoff = (now - timedelta(hours=lookback_hours)).isoformat()

        rows = self._conn.execute(
            "SELECT id, timestamp, event_type, glyphs, strength, details "
            "FROM synchronicity_events WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,),
        ).fetchall()

        if len(rows) < 3:
            return []  # need at least 3 events for meaningful clustering

        events = []
        for row in rows:
            events.append({
                "id": row[0],
                "timestamp": row[1],
                "event_type": row[2],
                "glyphs": json.loads(row[3]) if row[3] else [],
                "strength": row[4],
                "details": json.loads(row[5]) if row[5] else {},
            })

        # Cluster events
        clusters = self._cluster_events(events)

        # Count active glyphs
        glyph_row = self._conn.execute(
            "SELECT COUNT(DISTINCT glyph) FROM tag_events WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()
        total_active_glyphs = max(glyph_row[0] if glyph_row else 1, 1)

        # Evaluate each cluster
        new_patterns = []
        for cluster in clusters:
            pattern = self._test_significance(
                cluster, len(events), total_active_glyphs
            )
            if pattern:
                new_patterns.append(pattern)

        return new_patterns

    def _cluster_events(self, events: List[dict]) -> List[List[dict]]:
        """Hierarchical clustering with Jaccard distance, average linkage."""
        n = len(events)
        if n < 2:
            return [events] if events else []

        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
        except ImportError:
            # Fallback: treat all events as one cluster
            return [events]

        # Build condensed Jaccard distance matrix
        glyph_sets = [set(e["glyphs"]) for e in events]
        dist_matrix = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = glyph_sets[i], glyph_sets[j]
                union = a | b
                if not union:
                    dist_matrix.append(1.0)
                else:
                    dist_matrix.append(1.0 - len(a & b) / len(union))

        if not dist_matrix:
            return [events]

        threshold = self._thresholds.get("clustering_jaccard_threshold")
        Z = linkage(dist_matrix, method='average')
        labels = fcluster(Z, t=threshold, criterion='distance')

        # Group events by cluster label
        clusters_dict = defaultdict(list)
        for i, label in enumerate(labels):
            clusters_dict[label].append(events[i])

        return list(clusters_dict.values())

    def _test_significance(self, cluster: List[dict], total_events: int,
                           total_active_glyphs: int) -> Optional[dict]:
        """Test if a cluster is statistically significant."""
        if len(cluster) < 2:
            return None

        # Unique glyphs in this cluster
        cluster_glyphs = set()
        event_ids = []
        for evt in cluster:
            cluster_glyphs.update(evt["glyphs"])
            event_ids.append(evt["id"])

        if not cluster_glyphs:
            return None

        tag_share = len(cluster_glyphs) / total_active_glyphs
        expected = total_events * tag_share
        ratio = len(cluster) / max(expected, 0.01)

        base_rate = self._thresholds.get("base_rate_multiplier")
        if ratio < base_rate:
            return None

        # Cross-domain validation
        pattern_type, confidence = self._validate_cross_domain(
            cluster, cluster_glyphs, ratio
        )

        now = _now_iso()
        # Write to database
        cur = self._conn.execute(
            "INSERT INTO meta_patterns "
            "(timestamp, pattern_type, glyph_cluster, event_ids, "
            " cluster_size, significance_ratio, confidence, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'active')",
            (now, pattern_type, json.dumps(list(cluster_glyphs)),
             json.dumps(event_ids), len(cluster),
             round(ratio, 4), round(confidence, 4)),
        )
        self._conn.commit()

        return {
            "id": cur.lastrowid,
            "timestamp": now,
            "pattern_type": pattern_type,
            "glyph_cluster": list(cluster_glyphs),
            "event_ids": event_ids,
            "cluster_size": len(cluster),
            "significance_ratio": round(ratio, 4),
            "confidence": round(confidence, 4),
        }

    def _validate_cross_domain(self, cluster: List[dict],
                                cluster_glyphs: set,
                                ratio: float) -> Tuple[str, float]:
        """Check if cluster spans 2+ disconnected glyph groups.

        Builds a glyph graph where edges exist if two glyphs share >2
        memories. Uses BFS for connected component detection.
        """
        # Build adjacency from memory co-occurrence
        try:
            from app.memory import get_memory_engine
            engine = get_memory_engine()
            memories = engine.memories
        except Exception:
            return ("recurring_sync", min(1.0, ratio / 9.0))

        # Count shared memories per glyph pair
        pair_shared = defaultdict(int)
        glyph_list = list(cluster_glyphs)
        for mem in memories:
            mem_glyphs = set(mem.get("glyphs", []))
            cluster_in_mem = mem_glyphs & cluster_glyphs
            gl = list(cluster_in_mem)
            for i, ga in enumerate(gl):
                for gb in gl[i + 1:]:
                    pair_shared[(ga, gb)] += 1
                    pair_shared[(gb, ga)] += 1

        # Build adjacency list (edge if >2 shared memories)
        adj = defaultdict(set)
        for (ga, gb), count in pair_shared.items():
            if count > 2:
                adj[ga].add(gb)
                adj[gb].add(ga)

        # BFS connected components
        visited = set()
        components = []
        for g in glyph_list:
            if g in visited:
                continue
            component = set()
            queue = deque([g])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited and neighbor in cluster_glyphs:
                        queue.append(neighbor)
            if component:
                components.append(component)

        if len(components) >= 2:
            return ("cross_domain_theme", min(1.0, ratio / 6.0))
        else:
            return ("recurring_sync", min(1.0, ratio / 9.0))

    def get_recent_patterns(self, hours: float = 168.0) -> List[dict]:
        """Query recent meta-patterns."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=hours)).isoformat()
        rows = self._conn.execute(
            "SELECT id, timestamp, pattern_type, glyph_cluster, event_ids, "
            "cluster_size, significance_ratio, confidence, status "
            "FROM meta_patterns WHERE timestamp >= ? ORDER BY timestamp DESC",
            (cutoff,),
        ).fetchall()

        return [
            {
                "id": r[0], "timestamp": r[1], "pattern_type": r[2],
                "glyph_cluster": json.loads(r[3]) if r[3] else [],
                "event_ids": json.loads(r[4]) if r[4] else [],
                "cluster_size": r[5], "significance_ratio": r[6],
                "confidence": r[7], "status": r[8],
            }
            for r in rows
        ]


# ================================================================
# Feedback Engine
# ================================================================

class FeedbackEngine:
    """Feeds meta-pattern discoveries back into Layer 1."""

    def __init__(self, conn: sqlite3.Connection,
                 thresholds: AdaptiveThresholds):
        self._conn = conn
        self._thresholds = thresholds

    def process_pattern(self, pattern: dict):
        """Apply feedback actions for a discovered meta-pattern.

        cross_domain_theme: create emergent link + salience boost
        recurring_sync: salience boost only
        """
        if pattern["pattern_type"] == "cross_domain_theme":
            self._create_emergent_link(pattern)

        self._boost_salience(pattern)

    def _create_emergent_link(self, pattern: dict):
        """Create emergent link entry and store as a synthesized memory."""
        glyphs = pattern.get("glyph_cluster", [])
        if not glyphs:
            return

        # Pick representative glyphs (highest-temperature from each
        # connected component would be ideal, but we use all cluster
        # glyphs as the group for simplicity)
        now = _now_iso()
        self._conn.execute(
            "INSERT INTO emergent_links "
            "(glyph_group, pattern_type, confidence, created_at, "
            " source_pattern_id, is_active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (json.dumps(glyphs), pattern["pattern_type"],
             pattern["confidence"], now, pattern.get("id")),
        )
        self._conn.commit()

        # Store as a synthesized memory formula
        try:
            from app.memory import get_memory_engine
            formula = "→".join(glyphs[:6])  # join representative glyphs
            engine = get_memory_engine()
            record = engine.store_formula(formula)
            record["source_scroll"] = f"emergent_pattern_{pattern.get('id', '?')}"
            engine._save_memories()
        except Exception:
            pass  # feedback never breaks core

    def _boost_salience(self, pattern: dict):
        """Boost salience_model for all memories involved in the pattern's events."""
        event_ids = pattern.get("event_ids", [])
        if not event_ids:
            return

        # Get glyphs from the synchronicity events
        placeholders = ",".join("?" * len(event_ids))
        rows = self._conn.execute(
            f"SELECT glyphs FROM synchronicity_events WHERE id IN ({placeholders})",
            event_ids,
        ).fetchall()

        involved_glyphs = set()
        for (glyphs_json,) in rows:
            if glyphs_json:
                try:
                    involved_glyphs.update(json.loads(glyphs_json))
                except (json.JSONDecodeError, TypeError):
                    continue

        if not involved_glyphs:
            return

        # Find and boost memories containing these glyphs
        try:
            from app.memory import get_memory_engine
            from app.salience import compute_usage_salience, compute_composite
            engine = get_memory_engine()

            boosted = 0
            for mem in engine.memories:
                mem_glyphs = set(mem.get("glyphs", []))
                if mem_glyphs & involved_glyphs:
                    old_model = mem.get("salience_model", 0.5)
                    new_model = min(1.0, max(0.0, old_model + SALIENCE_BOOST))
                    mem["salience_model"] = round(new_model, 4)

                    # Recompute composite
                    usage = compute_usage_salience(
                        mem.get("activation_count", 0),
                        mem.get("last_accessed", mem.get("timestamp", "")),
                        mem.get("retrieval_success_count", 0),
                        mem.get("retrieval_total_count", 1),
                    )
                    mem["salience_usage"] = round(usage, 4)
                    mem["salience_composite"] = round(
                        compute_composite(new_model, usage), 4
                    )

                    # Log the boost
                    self._conn.execute(
                        "INSERT INTO salience_boosts "
                        "(timestamp, memory_id, old_salience_model, "
                        " new_salience_model, source_pattern_id) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (_now_iso(), mem["id"], old_model,
                         new_model, pattern.get("id")),
                    )
                    boosted += 1

            if boosted > 0:
                self._conn.commit()
                engine._save_memories()

        except Exception:
            pass  # feedback never breaks core

    def get_emergent_links(self, active_only: bool = True) -> List[dict]:
        """Query emergent links."""
        query = "SELECT id, glyph_group, pattern_type, confidence, created_at, source_pattern_id, is_active FROM emergent_links"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY created_at DESC"

        rows = self._conn.execute(query).fetchall()
        return [
            {
                "id": r[0],
                "glyph_group": json.loads(r[1]) if r[1] else [],
                "pattern_type": r[2], "confidence": r[3],
                "created_at": r[4], "source_pattern_id": r[5],
                "is_active": bool(r[6]),
            }
            for r in rows
        ]


# ----------------------------
# Singletons
# ----------------------------
_thresholds: Optional[AdaptiveThresholds] = None
_meta_analyzer: Optional[MetaAnalyzer] = None
_feedback_engine: Optional[FeedbackEngine] = None


def get_adaptive_thresholds() -> AdaptiveThresholds:
    """Get or create the singleton adaptive thresholds."""
    global _thresholds
    if _thresholds is None:
        from app.trends import get_trend_engine
        _thresholds = AdaptiveThresholds(get_trend_engine()._conn)
    return _thresholds


def get_meta_analyzer() -> MetaAnalyzer:
    """Get or create the singleton meta-analyzer."""
    global _meta_analyzer
    if _meta_analyzer is None:
        conn = get_adaptive_thresholds()._conn
        _meta_analyzer = MetaAnalyzer(conn, get_adaptive_thresholds())
    return _meta_analyzer


def get_feedback_engine() -> FeedbackEngine:
    """Get or create the singleton feedback engine."""
    global _feedback_engine
    if _feedback_engine is None:
        conn = get_adaptive_thresholds()._conn
        _feedback_engine = FeedbackEngine(conn, get_adaptive_thresholds())
    return _feedback_engine
