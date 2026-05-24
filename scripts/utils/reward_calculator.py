"""
Compatibility wrapper for reward utilities used by RL training scripts.
"""

from mocps.metrics import (
    NORM_STATS,
    build_cousin_ref,
    compute_cfd,
    compute_cis,
    compute_cousin,
    compute_csi,
    update_norm_stats,
    z_score,
)
from mocps.reward import RewardCalculator

__all__ = [
    "NORM_STATS",
    "RewardCalculator",
    "build_cousin_ref",
    "compute_cfd",
    "compute_cis",
    "compute_cousin",
    "compute_csi",
    "update_norm_stats",
    "z_score",
]
