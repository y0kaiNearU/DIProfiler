from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from models.models import PipelineRequest

type Vote[T] = tuple[T, float, str]
type Rule[T] = Callable[[PipelineRequest], Vote[T] | None]


@dataclass
class Tally:
    total_weight: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def add(self, weight: float, reason: str) -> None:
        self.total_weight += weight
        self.reasons.append(reason)


def aggregate_votes[T](
    request: PipelineRequest,
    rules: list[Rule[T]],
    candidates: set[T],
) -> list[tuple[T, float, str]]:
    """Run rules against request and tally weighted votes per candidate.

    Returns (candidate, confidence, reasoning) tuples for candidates with a
    nonzero tally, confidence normalized to sum to 1.0, sorted descending.
    """
    tallies: dict[T, Tally] = {c: Tally() for c in candidates}

    for rule in rules:
        vote = rule(request)
        if vote is not None:
            candidate, weight, reason = vote
            if candidate in tallies:
                tallies[candidate].add(weight, reason)

    total = sum(t.total_weight for t in tallies.values()) or 1.0

    scored = [
        (candidate, round(tally.total_weight / total, 3), "; ".join(tally.reasons) or "no applicable rules matched")
        for candidate, tally in tallies.items()
        if tally.total_weight > 0
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
