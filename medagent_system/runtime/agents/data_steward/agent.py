from __future__ import annotations

from medagent_system.runtime.core.models import CPB


def validate_and_normalize_cpb(cpb: CPB) -> CPB:
    timeline_sorted = sorted(cpb.timeline, key=lambda e: e.t)
    if timeline_sorted != cpb.timeline:
        cpb.timeline = timeline_sorted
    return cpb
