# app/salience.py
"""
Layer 1: Dual-Track Salience Scoring for GPT-GU

Each memory carries two salience scores:
  - salience_model: system judgment of importance at storage time (0..1)
  - salience_usage: derived from access patterns over time (0..1)

These combine into a composite score that evolves with every retrieval:
  composite = 0.6 * model + 0.4 * usage
"""
from __future__ import annotations
import math
from datetime import datetime, timezone

# ----------------------------
# Constants
# ----------------------------
RECENCY_HALFLIFE_HOURS = 168.0          # 1 week
RECENCY_LAMBDA = math.log(2) / RECENCY_HALFLIFE_HOURS
SIGMOID_STEEPNESS = 6.0
SIGMOID_CENTER = 0.5
ACCESS_LOG_BASE = 101                   # log(1+count)/log(101) capped at 1.0
WEIGHT_ACCESS = 0.4
WEIGHT_RECENCY = 0.4
WEIGHT_SUCCESS = 0.2
COMPOSITE_MODEL_WEIGHT = 0.6
COMPOSITE_USAGE_WEIGHT = 0.4
SUCCESS_WINDOW_SECONDS = 300            # 5 minutes for retrieval-success heuristic


def _sigmoid(x: float, steepness: float = SIGMOID_STEEPNESS,
             center: float = SIGMOID_CENTER) -> float:
    """Standard sigmoid mapped through steepness and center."""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def compute_usage_salience(
    access_count: int,
    last_accessed_iso: str,
    success_count: int,
    total_count: int,
) -> float:
    """Compute usage-derived salience from access patterns.

    Combines three factors through a sigmoid:
      access_score  = min(log(1+count)/log(101), 1.0)    weight 40%
      recency_score = exp(-lambda * hours_since_access)   weight 40%
      success_score = success/total (or 0.5 if untested)  weight 20%
    """
    # Access frequency (logarithmic, saturates around 100 accesses)
    access_score = min(math.log(1 + access_count) / math.log(ACCESS_LOG_BASE), 1.0)

    # Recency (exponential decay with 1-week halflife)
    try:
        last_dt = datetime.fromisoformat(last_accessed_iso)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hours_since = max((now - last_dt).total_seconds() / 3600.0, 0.0)
    except (ValueError, TypeError):
        hours_since = RECENCY_HALFLIFE_HOURS  # default to halflife if unparseable
    recency_score = math.exp(-RECENCY_LAMBDA * hours_since)

    # Success rate
    if total_count > 0:
        success_score = success_count / total_count
    else:
        success_score = 0.5  # neutral prior

    raw = (WEIGHT_ACCESS * access_score
           + WEIGHT_RECENCY * recency_score
           + WEIGHT_SUCCESS * success_score)

    return _sigmoid(raw)


def compute_composite(salience_model: float, salience_usage: float) -> float:
    """Weighted blend of model judgment and usage data."""
    return (COMPOSITE_MODEL_WEIGHT * salience_model
            + COMPOSITE_USAGE_WEIGHT * salience_usage)


def compute_store_salience(total_heat: float) -> float:
    """System judgment of importance at storage time.

    Based on total cognitive intensity (sum of all glyph temperatures).
    Sigmoid-like scaling so salience is 0..1.
    """
    if total_heat <= 0:
        return 0.0
    return total_heat / (total_heat + 10.0)


def upgrade_record(record: dict) -> dict:
    """Add missing L1 fields to an old-format memory record.

    Idempotent — skips fields that already exist.
    Mutates the record in place and returns it.
    """
    if "salience_model" not in record:
        record["salience_model"] = record.get("store_salience", 0.5)

    if "salience_usage" not in record:
        record["salience_usage"] = 0.5

    if "salience_composite" not in record:
        record["salience_composite"] = round(
            compute_composite(record["salience_model"], record["salience_usage"]), 4
        )

    if "last_accessed" not in record:
        record["last_accessed"] = record.get("timestamp",
                                              datetime.now(timezone.utc).isoformat())

    if "retrieval_success_count" not in record:
        record["retrieval_success_count"] = 0

    if "retrieval_total_count" not in record:
        record["retrieval_total_count"] = record.get("activation_count", 0)

    if "is_archived" not in record:
        record["is_archived"] = False

    return record
