import numpy as np
import pytest

from vpype_explorations.variablewidth import pixel_to_half_width


@pytest.mark.parametrize(
    ("pixel_value", "pitch", "pen_width", "black_level", "white_level", "expected"),
    [
        (0, 10, 1, 0, 1, 0.5 * 9),  # basic black
        (1, 10, 1, 0, 1, 0.0),  # basic white
        ([0, 0.01, 0.1], 10, 1, 0.1, 1, [4.5, 4.5, 4.5]),  # black level
        ([0.9, 0.96, 0.999], 10, 1, 0, 0.9, [0, 0, 0]),  # white level
    ],
)
def test_pixel_to_half_width(
    pixel_value, pitch, pen_width, black_level, white_level, expected
):

    if not isinstance(pixel_value, np.ndarray):
        pixel_value = np.array(pixel_value)
    if not isinstance(expected, np.ndarray):
        expected = np.array(expected)

    assert np.array_equal(
        pixel_to_half_width(pixel_value, pitch, pen_width, black_level, white_level), expected
    )
