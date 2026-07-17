from __future__ import annotations

from typing import Any, Callable

import numpy as np


def score_with_model[T](
    model: Any,
    features: list[float],
    candidate_type: Callable[[str], T],
) -> list[tuple[T, float]]:
    """Run a fitted sklearn-compatible classifier on a single feature row.

    Returns (candidate, probability) pairs for each of the model's classes.
    """
    classes = [candidate_type(c) for c in model.classes_]
    probas = model.predict_proba(np.array([features], dtype=np.float32))[0]
    return list(zip(classes, probas))
