import math
import random
from typing import Tuple, Optional, List

import click
import numpy as np
import pytest
from vpype import (
    LineCollection,
    Length,
    global_processor,
    VectorData,
    layer_processor,
    LayerType,
    single_to_layer_id,
)


def line_length(line: np.ndarray) -> float:
    return np.abs(np.diff(line)).sum()


def curvilinear_abscissa(line: np.ndarray) -> np.ndarray:
    """Computes the curvilinear abscissa of a line.

    Args:
        line: input line (dtype = complex)

    Return:
        curvilinear abscissa (dtype = float, same shape as input)
    """
    return np.hstack([0, np.cumsum(np.abs(np.diff(line)))])


def cut_line(
    line: np.ndarray, location: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Cut a line in two part at a location expressed as curvilinear abscissa.

    Args:
        line: line to be cut
        location: location at which to cut the line, must be between 0 and line's total length
            (see :func:`line_length`)

    Returns:
        tuple of lines
    """

    if location <= 0:
        return None, line

    curv_absc = curvilinear_abscissa(line)

    if location >= curv_absc[-1]:
        return line, None

    cut_idx = np.argmax(curv_absc > location)

    a1 = curv_absc[cut_idx - 1]
    a2 = curv_absc[cut_idx]

    r = (location - a1) / (a2 - a1)  # should be between 0 and 1

    if r == 0:
        return line[:cut_idx], line[cut_idx - 1 :]
    elif r == 1:
        return line[: cut_idx + 1], line[cut_idx:]
    else:
        c1 = line[cut_idx - 1]
        c2 = line[cut_idx]
        c = c1 + r * (c2 - c1)
        return np.hstack([line[:cut_idx], c]), np.hstack([c, line[cut_idx:]])


def punch_hole(
    line: np.ndarray, location: float, width: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Punch a hole of defined width and location in a line.

    Args:
        line: input line (dtype = complex)
        location: curvilinear abscissa of the hole center
        width: width (in curvilinear abscissa of the hole

    Returns:
        2-tuple of lines (dtype = complex) or None
    """

    ll = line_length(line)
    start = min(ll, max(0.0, location - width / 2))
    stop = min(ll, max(0.0, location + width / 2))

    l1, _ = cut_line(line, start)
    _, l2 = cut_line(line, stop)
    return l1, l2


def circle(r: float, quantization: float) -> np.ndarray:
    n = math.ceil(2 * math.pi * r / quantization)
    angle = np.array(list(range(n)) + [0]) / n * 2 * math.pi
    return r * (np.cos(angle) + 1j * np.sin(angle))


@click.command()
@click.argument("count", type=int)
@click.argument("delta", type=Length())
@click.option("-q", "--quantization", type=Length(), default="0.05mm")
@click.option("-lc", "--layer-count", type=int, default=1)
@click.option("-rl", "--random-layer", is_flag=True)
@click.option(
    "-l", "--layer", type=LayerType(accept_new=True), default="1", help="Starting layer ID."
)
@click.option(
    "-o",
    "--offset",
    type=Length(),
    nargs=2,
    default=(0, 0),
    help="Location of the cirles' center",
)
@global_processor
def circles(
    vector_data: VectorData,
    count,
    delta,
    quantization,
    layer_count,
    random_layer,
    layer,
    offset,
):

    start_layer_id = single_to_layer_id(layer, vector_data)
    for i in range(count):
        if random_layer:
            lid = start_layer_id + random.randint(0, layer_count - 1)
        else:
            lid = start_layer_id + (i % layer_count)

        vector_data.add(
            LineCollection(
                [circle((i + 1) * delta, quantization) + offset[0] + 1j * offset[1]]
            ),
            lid,
        )

    return vector_data


circles.help_group = "Plug-ins"


@click.command()
@click.option("-hs", "--hole-size", type=Length(), default="1mm")
@layer_processor
def holes(lines: LineCollection, hole_size):

    new_lines = LineCollection()
    for line in lines:
        ll = line_length(line)

        hole_arr = np.random.rand(math.floor(ll / 2 / hole_size)) < 0.2
        (hole_idx,) = np.where(hole_arr)
        locations = hole_idx * 2 * hole_size

        remain = line
        for loc in np.flip(locations):
            remain, to_add = punch_hole(remain, loc, hole_size)
            if to_add is not None:
                new_lines.append(to_add)
        if remain is not None:
            new_lines.append(remain)

    return new_lines


holes.help_group = "Plug-ins"
