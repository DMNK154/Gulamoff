# app/trends.py
"""
Layer 2: Trend Engine & Retrieval Tracking for GPT-GU

TrendEngine computes per-glyph trend vectors (level, velocity, jerk, temperature)
over a sliding window from logged tag events.

RetrievalTracker logs every retrieval event and computes normalized retrieval
frequency per glyph.

Both share a single SQLite database in WAL mode.
"""
from __future__ import annotations
import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ----------------------------
# Constants
# ----------------------------
TREND_WINDOW_HOURS = 168.0              # 1 week sliding window
TREND_HALFLIFE = TREND_WINDOW_HOURS / 2  # 84 hours
TREND_LAMBDA = math.log(2) / TREND_HALFLIFE
PRUNE_AGE_HOURS = 720.0                 # 30 days — oldest events to keep
PRUNE_INTERVAL = 100                    # prune every N tag events

# Temperature composite weights
WEIGHT_LEVEL = 0.50
WEIGHT_VELOCITY = 0.35
WEIGHT_JERK = 0.15

_DATA_DIR = Path(__file__).parent.parent
_DEFAULT_DB = _DATA_DIR / "trend_data.db"

# ----------------------------
# Schema
# ----------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tag_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    glyph TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tag_glyph_ts
    ON tag_events(glyph, timestamp);

CREATE TABLE IF NOT EXISTS retrieval_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query_type TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    top_score REAL,
    glyphs_in_results TEXT
);
CREATE INDEX IF NOT EXISTS idx_retr_ts
    ON retrieval_events(timestamp);

CREATE TABLE IF NOT EXISTS retrieval_memory_hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    retrieval_event_id INTEGER NOT NULL,
    memory_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL,
    FOREIGN KEY (retrieval_event_id) REFERENCES retrieval_events(id)
);

-- L3: Phase Space & Synchronicity
CREATE TABLE IF NOT EXISTS synchronicity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    glyphs TEXT NOT NULL,
    strength REAL NOT NULL,
    details TEXT,
    quadrants TEXT,
    lattice_angle REAL,
    lattice_slots TEXT
);
CREATE INDEX IF NOT EXISTS idx_sync_ts ON synchronicity_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_sync_type ON synchronicity_events(event_type);

-- L3: Prime Ramsey Lattice
CREATE TABLE IF NOT EXISTS resonances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_a_id INTEGER NOT NULL,
    event_b_id INTEGER NOT NULL,
    shared_primes TEXT NOT NULL,
    resonance_strength REAL NOT NULL,
    chance REAL NOT NULL,
    FOREIGN KEY (event_a_id) REFERENCES synchronicity_events(id),
    FOREIGN KEY (event_b_id) REFERENCES synchronicity_events(id)
);
CREATE INDEX IF NOT EXISTS idx_res_ts ON resonances(timestamp);

-- L4: Meta Patterns
CREATE TABLE IF NOT EXISTS meta_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    glyph_cluster TEXT NOT NULL,
    event_ids TEXT NOT NULL,
    cluster_size INTEGER NOT NULL,
    significance_ratio REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_meta_ts ON meta_patterns(timestamp);

-- L4: Adaptive Thresholds
CREATE TABLE IF NOT EXISTS adaptive_thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    current_value REAL NOT NULL,
    default_value REAL NOT NULL,
    last_updated TEXT NOT NULL,
    burn_in_until TEXT
);

-- L4: Emergent Links
CREATE TABLE IF NOT EXISTS emergent_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    glyph_group TEXT NOT NULL,
    pattern_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    source_pattern_id INTEGER,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (source_pattern_id) REFERENCES meta_patterns(id)
);
CREATE INDEX IF NOT EXISTS idx_elink_active ON emergent_links(is_active);

-- L4: Salience Boost Log
CREATE TABLE IF NOT EXISTS salience_boosts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    old_salience_model REAL NOT NULL,
    new_salience_model REAL NOT NULL,
    source_pattern_id INTEGER,
    FOREIGN KEY (source_pattern_id) REFERENCES meta_patterns(id)
);
"""


def _sigmoid(x: float) -> float:
    """Sigmoid clamped to avoid overflow."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrendEngine:
    """Per-glyph trend analysis over a sliding event window."""

    def __init__(self, db_path: str = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._conn = self._connect()
        self._event_counter = 0

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    # ----------------------------------------------------------------
    # Event Logging
    # ----------------------------------------------------------------

    def log_tag_event(self, glyph: str, source: str, weight: float = 1.0):
        """Log a single tag event."""
        self._conn.execute(
            "INSERT INTO tag_events (glyph, timestamp, weight, source) VALUES (?, ?, ?, ?)",
            (glyph, _now_iso(), weight, source),
        )
        self._conn.commit()
        self._maybe_prune()

    def log_tag_events_batch(self, glyphs: List[str], source: str,
                             weight: float = 1.0):
        """Log tag events for multiple glyphs in one transaction."""
        if not glyphs:
            return
        ts = _now_iso()
        rows = [(g, ts, weight, source) for g in glyphs]
        self._conn.executemany(
            "INSERT INTO tag_events (glyph, timestamp, weight, source) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        self._event_counter += len(glyphs)
        if self._event_counter >= PRUNE_INTERVAL:
            self._maybe_prune()

    # ----------------------------------------------------------------
    # Trend Computation
    # ----------------------------------------------------------------

    def compute_glyph_trend(self, glyph: str,
                            now: Optional[datetime] = None) -> dict:
        """Compute trend vector for a single glyph.

        Returns {level, velocity, jerk, temperature, event_count}.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=TREND_WINDOW_HOURS)

        rows = self._conn.execute(
            "SELECT timestamp, weight FROM tag_events "
            "WHERE glyph = ? AND timestamp >= ? ORDER BY timestamp",
            (glyph, cutoff.isoformat()),
        ).fetchall()

        if not rows:
            return {"level": 0.0, "velocity": 0.0, "jerk": 0.0,
                    "temperature": 0.0, "event_count": 0}

        # Parse events into (age_hours, weight) pairs
        events = []
        for ts_str, weight in rows:
            try:
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_h = max((now - ts).total_seconds() / 3600.0, 0.0)
                events.append((age_h, weight))
            except (ValueError, TypeError):
                continue

        if not events:
            return {"level": 0.0, "velocity": 0.0, "jerk": 0.0,
                    "temperature": 0.0, "event_count": 0}

        # Level: sum of decayed weights
        level = sum(w * math.exp(-TREND_LAMBDA * age) for age, w in events)

        # Velocity: recent half vs older half
        half = TREND_WINDOW_HOURS / 2
        recent = sum(w * math.exp(-TREND_LAMBDA * age)
                     for age, w in events if age <= half)
        older = sum(w * math.exp(-TREND_LAMBDA * age)
                    for age, w in events if age > half)
        velocity = recent - older

        # Jerk: thirds finite-difference
        third = TREND_WINDOW_HOURS / 3
        t1 = sum(w * math.exp(-TREND_LAMBDA * age)
                 for age, w in events if age > 2 * third)
        t2 = sum(w * math.exp(-TREND_LAMBDA * age)
                 for age, w in events if third < age <= 2 * third)
        t3 = sum(w * math.exp(-TREND_LAMBDA * age)
                 for age, w in events if age <= third)
        jerk = t3 - 2 * t2 + t1

        return {
            "level": round(level, 4),
            "velocity": round(velocity, 4),
            "jerk": round(jerk, 4),
            "temperature": 0.0,  # computed in compute_all_trends with p90 normalization
            "event_count": len(events),
        }

    def compute_all_trends(self,
                           now: Optional[datetime] = None) -> Dict[str, dict]:
        """Compute trends for all glyphs with events in the window.

        Temperature is normalized via the 90th percentile of raw scores.
        """
        if now is None:
            now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=TREND_WINDOW_HOURS)

        # Find all active glyphs
        glyph_rows = self._conn.execute(
            "SELECT DISTINCT glyph FROM tag_events WHERE timestamp >= ?",
            (cutoff.isoformat(),),
        ).fetchall()

        if not glyph_rows:
            return {}

        glyphs = [r[0] for r in glyph_rows]
        trends = {}
        raw_scores = []

        for g in glyphs:
            t = self.compute_glyph_trend(g, now=now)
            trends[g] = t
            raw = (WEIGHT_LEVEL * t["level"]
                   + WEIGHT_VELOCITY * max(0.0, t["velocity"])
                   + WEIGHT_JERK * max(0.0, t["jerk"]))
            raw_scores.append((g, raw))

        # 90th percentile normalization
        if raw_scores:
            sorted_raw = sorted(r for _, r in raw_scores)
            p90_idx = max(0, int(len(sorted_raw) * 0.9) - 1)
            p90 = sorted_raw[p90_idx] if sorted_raw[p90_idx] > 0 else 1.0

            for g, raw in raw_scores:
                trends[g]["temperature"] = round(_sigmoid(raw / p90), 4)

        return trends

    def get_trending_glyphs(self, top_n: int = 10,
                            now: Optional[datetime] = None) -> List[dict]:
        """Return top N glyphs by trend temperature."""
        trends = self.compute_all_trends(now=now)
        if not trends:
            return []

        sorted_t = sorted(trends.items(),
                          key=lambda x: x[1]["temperature"], reverse=True)
        return [
            {"glyph": g, **data}
            for g, data in sorted_t[:top_n]
        ]

    # ----------------------------------------------------------------
    # Co-activation Data (for L3 Ramsey Lattice PCA)
    # ----------------------------------------------------------------

    def get_coactivation_data(self,
                              window_hours: float = TREND_WINDOW_HOURS
                              ) -> List[tuple]:
        """Get co-firing glyph pairs for lattice PCA.

        Groups tag_events by 1-second timestamp windows to find glyphs
        that fire together. Returns [(glyph_a, glyph_b, combined_weight), ...].
        """
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=window_hours)).isoformat()

        # Get all events in window, ordered by timestamp
        rows = self._conn.execute(
            "SELECT glyph, timestamp, weight FROM tag_events "
            "WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,),
        ).fetchall()

        if not rows:
            return []

        # Group by 1-second windows
        from collections import defaultdict
        windows = defaultdict(list)
        for glyph, ts, weight in rows:
            # Truncate to second precision for grouping
            sec_key = ts[:19]  # ISO format: YYYY-MM-DDTHH:MM:SS
            windows[sec_key].append((glyph, weight))

        # Build co-activation pairs
        pair_weights = defaultdict(float)
        for sec_key, events in windows.items():
            glyphs_in_window = events
            for i, (ga, wa) in enumerate(glyphs_in_window):
                for gb, wb in glyphs_in_window[i + 1:]:
                    if ga != gb:
                        pair = tuple(sorted([ga, gb]))
                        pair_weights[pair] += (wa + wb) / 2.0

        return [(a, b, w) for (a, b), w in pair_weights.items()]

    # ----------------------------------------------------------------
    # Maintenance
    # ----------------------------------------------------------------

    def _maybe_prune(self):
        """Delete events older than PRUNE_AGE_HOURS."""
        if self._event_counter < PRUNE_INTERVAL:
            return
        self._event_counter = 0
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=PRUNE_AGE_HOURS)).isoformat()
        self._conn.execute(
            "DELETE FROM tag_events WHERE timestamp < ?", (cutoff,),
        )
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()


class RetrievalTracker:
    """Logs retrieval events and computes per-glyph retrieval frequency."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------

    def log_retrieval(self, query_type: str, results: List[dict]) -> int:
        """Log a retrieval event with its result set.

        Returns the retrieval_event_id.
        """
        # Collect glyphs from all results
        all_glyphs = set()
        for r in results:
            for g in r.get("glyphs", []):
                all_glyphs.add(g)

        top_score = max((r.get("_recall_score", 0.0) for r in results),
                        default=0.0)

        cur = self._conn.execute(
            "INSERT INTO retrieval_events "
            "(timestamp, query_type, result_count, top_score, glyphs_in_results) "
            "VALUES (?, ?, ?, ?, ?)",
            (_now_iso(), query_type, len(results), top_score,
             json.dumps(list(all_glyphs), ensure_ascii=False)),
        )
        event_id = cur.lastrowid

        # Log individual memory hits
        hit_rows = []
        for rank, r in enumerate(results, 1):
            hit_rows.append((
                event_id,
                r.get("id", ""),
                rank,
                r.get("_recall_score", 0.0),
            ))
        if hit_rows:
            self._conn.executemany(
                "INSERT INTO retrieval_memory_hits "
                "(retrieval_event_id, memory_id, rank, score) "
                "VALUES (?, ?, ?, ?)",
                hit_rows,
            )

        self._conn.commit()
        return event_id

    # ----------------------------------------------------------------
    # Frequency Analysis
    # ----------------------------------------------------------------

    def get_glyph_retrieval_frequency(self, glyph: str,
                                       window_hours: float = TREND_WINDOW_HOURS
                                       ) -> float:
        """Normalized retrieval frequency for a glyph.

        Returns sigmoid(ln(raw_freq / base_rate)), mapped to [0, 1].
        """
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=window_hours)).isoformat()

        # How many retrieval events included this glyph?
        row = self._conn.execute(
            "SELECT COUNT(*) FROM retrieval_events "
            "WHERE timestamp >= ? AND glyphs_in_results LIKE ?",
            (cutoff, f'%"{glyph}"%'),
        ).fetchone()
        glyph_hits = row[0] if row else 0

        # Total retrievals and unique glyphs in window
        stats = self._conn.execute(
            "SELECT COUNT(*), GROUP_CONCAT(glyphs_in_results) "
            "FROM retrieval_events WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()
        total_retrievals = stats[0] if stats else 0

        if total_retrievals == 0 or glyph_hits == 0:
            return 0.0

        # Count unique glyphs across all retrievals
        all_glyphs_str = stats[1] or ""
        unique_glyphs = set()
        for chunk in all_glyphs_str.split(","):
            try:
                parsed = json.loads(chunk.strip()) if chunk.strip() else []
                unique_glyphs.update(parsed)
            except (json.JSONDecodeError, TypeError):
                continue
        num_active = max(len(unique_glyphs), 1)

        raw_freq = glyph_hits / window_hours
        base_rate = total_retrievals / (num_active * window_hours)

        if base_rate <= 0:
            return 0.5

        ratio = raw_freq / base_rate
        if ratio <= 0:
            return 0.0

        return round(_sigmoid(math.log(ratio)), 4)

    def get_retrieval_stats(self,
                            window_hours: float = TREND_WINDOW_HOURS) -> dict:
        """Summary statistics for the weather report."""
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=window_hours)).isoformat()

        row = self._conn.execute(
            "SELECT COUNT(*), AVG(result_count), AVG(top_score) "
            "FROM retrieval_events WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()

        total = row[0] if row and row[0] else 0
        avg_results = round(row[1], 1) if row and row[1] else 0.0
        avg_score = round(row[2], 4) if row and row[2] else 0.0

        # Most-retrieved glyphs
        glyph_rows = self._conn.execute(
            "SELECT glyphs_in_results FROM retrieval_events "
            "WHERE timestamp >= ?",
            (cutoff,),
        ).fetchall()

        from collections import Counter
        glyph_counts = Counter()
        for (glyphs_json,) in glyph_rows:
            if glyphs_json:
                try:
                    glyph_counts.update(json.loads(glyphs_json))
                except (json.JSONDecodeError, TypeError):
                    continue

        most_retrieved = [
            {"glyph": g, "count": c}
            for g, c in glyph_counts.most_common(5)
        ]

        return {
            "total_retrievals": total,
            "avg_results": avg_results,
            "avg_score": avg_score,
            "most_retrieved": most_retrieved,
        }


# ----------------------------
# Singletons
# ----------------------------
_trend_engine: Optional[TrendEngine] = None
_retrieval_tracker: Optional[RetrievalTracker] = None


def get_trend_engine() -> TrendEngine:
    """Get or create the singleton trend engine."""
    global _trend_engine
    if _trend_engine is None:
        _trend_engine = TrendEngine()
    return _trend_engine


def get_retrieval_tracker() -> RetrievalTracker:
    """Get or create the singleton retrieval tracker.

    Shares the same SQLite connection as the TrendEngine.
    """
    global _retrieval_tracker
    if _retrieval_tracker is None:
        engine = get_trend_engine()
        _retrieval_tracker = RetrievalTracker(engine._conn)
    return _retrieval_tracker
