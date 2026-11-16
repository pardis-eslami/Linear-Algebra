"""Test fixtures and random matrix generation for the dashboard."""
from __future__ import annotations

import random

from linear_algebra import as_matrix, det_value

# Representative examples used by the engine and view tests.
EXAMPLES: dict[str, list[dict]] = {
    'det': [
        {'A': [[2, 0, 1], [3, -1, 4], [1, 2, 0]]},                       # has a zero to expand along
        {'A': [[1.5, 2, 0], [0, 1, 2.5], [2, 0, 1]]},                    # decimals
        {'A': [[1, 2, 0, 3], [0, 1, 4, 1], [2, 0, 1, 0], [1, 3, 2, 1]]},  # 4x4
        {'A': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},                        # singular
    ],
    'mul': [
        {'A': [[1, 2, 3], [4, 5, 6]], 'B': [[7, 8], [9, 10], [11, 12]]},
        {'A': [[1, 0, 2], [-1, 3, 1], [2, 1, 0]], 'B': [[1, 2, 0], [0, 1, 1], [3, 0, 1]]},
    ],
    'ref': [
        {'A': [[0, 2, 1, 4], [1, 1, 0, 3], [2, 4, 1, 9]]},               # needs a row swap
        {'A': [[1, 2, 1, 4], [2, 4, 3, 9], [3, 6, 4, 13]]},              # a column without pivot
        {'A': [[0.5, 1, 2], [1.5, 2.5, 4], [2, 1, 0.5]]},                # decimals
    ],
    'rref': [
        {'A': [[0, 2, 1, 4], [1, 1, 0, 3], [2, 4, 1, 9]]},
        {'A': [[1, 2, 1, 4], [2, 4, 3, 9], [3, 6, 4, 13]]},
        {'A': [[0.5, 1, 2], [1.5, 2.5, 4], [2, 1, 0.5]]},
    ],
    'inv': [
        {'A': [[2, 1, 1], [1, 3, 2], [1, 0, 0]]},
        {'A': [[4, 7], [2, 6]]},
        {'A': [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},                        # singular
    ],
    'sys': [
        {'A': [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], 'b': [8, -11, -3]},  # one solution
        {'A': [[1, 2, 1], [2, 4, 3]], 'b': [4, 9]},                       # infinitely many
        {'A': [[1, 1], [2, 2]], 'b': [1, 3]},                             # no solution
    ],
}


def random_matrix(rows: int, cols: int, rng: random.Random | None = None, *, lo: int = -5,
                  hi: int = 9, zero_p: float = 0.2, invertible: bool = False) -> list[list[int]]:
    """Small integers with a few zeros (so zero-handling shows up in the steps)."""
    rng = rng or random.Random()
    rows_ = [[0] * cols for _ in range(rows)]
    for _ in range(300):
        rows_ = [[0 if rng.random() < zero_p else rng.randint(lo, hi) for _ in range(cols)]
                 for _ in range(rows)]
        if not any(any(r) for r in rows_):
            continue
        if not invertible or det_value(as_matrix(rows_)) != 0:
            return rows_
    return rows_
