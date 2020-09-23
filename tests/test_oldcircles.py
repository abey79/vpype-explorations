import pytest

import numpy as np
from vpype_explorations.oldcircles import cut_line


def build_line(*arr):
    return np.array(arr, dtype=complex)


@pytest.mark.parametrize(
    ["line", "absc", "result"],
    [
        ([0, 10], 5, ([0, 5], [5, 10])),
        ([0, 10], 0, (None, [0, 10])),
        ([0, 10], 10, ([0, 10], None)),
        ([0, 10, 10 + 10j], 10, ([0, 10], [10, 10 + 10j])),
        ([0, 10, 10 + 10j], 15, ([0, 10, 10 + 5j], [10 + 5j, 10 + 10j])),
    ],
)
def test_cut_line(line, absc, result):
    actual = cut_line(build_line(*line), absc)
    assert (actual[0] is result[0] is None) or np.all(actual[0] == build_line(*result[0]))
    assert (actual[1] is result[1] is None) or np.all(actual[1] == build_line(*result[1]))
