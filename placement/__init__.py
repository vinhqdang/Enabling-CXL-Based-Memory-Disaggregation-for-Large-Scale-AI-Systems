"""Resident-layer placement for weight streaming (see paper/DESIGN.md)."""
from .model import (
    cycle_time,
    cycle_time_contention,
    lower_bound,
    schedule,
    schedule_contention,
)
from .placements import (
    balanced,
    brute_force_optimal,
    contiguous_first,
    contiguous_last,
    gaps,
    is_balanced,
    local_search,
    random_placement,
)
from .theory import budget_optimal_uniform, contiguous_uniform, lb_uniform

__all__ = [
    "cycle_time",
    "cycle_time_contention",
    "lower_bound",
    "schedule",
    "schedule_contention",
    "balanced",
    "brute_force_optimal",
    "contiguous_first",
    "contiguous_last",
    "gaps",
    "is_balanced",
    "local_search",
    "random_placement",
    "budget_optimal_uniform",
    "contiguous_uniform",
    "lb_uniform",
]
