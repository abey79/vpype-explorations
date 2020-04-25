import itertools
import math
import random
from typing import Sequence, Tuple, Optional, Dict, Any

import click
import numpy as np
from vpype import generator, LineCollection


def _wheelsonwheelsonwheels(
    k: int, n: Sequence[int], a: Sequence[float], s: Sequence[float]
) -> np.ndarray:
    """Implements the "wheels on wheels on wheels" geometry from Farris, 1996

        Returns an array of complex as follows:

            z(t) = sum_k(a_k  * exp(2πi * (n_k * t + s_k))   for t in [0, 1]

        Refs:
        - https://linuxgazette.net/133/luana.html
        - https://core.ac.uk/download/pdf/72850999.pdf

        Args:
            k: number of point in the generated trajectory
            n: relates to the rotation speed of each wheel
            a: relates to the radius of each wheel
            s: starting position of each wheel

        Returns:
            The generated trajectory as an array of complex number
    """

    n = np.array(n).reshape(len(n), 1)
    a = np.array(a).reshape(len(a), 1)
    s = np.array(s).reshape(len(s), 1)

    t = np.linspace(0, 1, k).reshape(1, k)

    return np.sum(a * np.exp(1j * 2 * np.pi * (np.dot(n, t) + s)), axis=0)


def random_symmetric(
    symmetry_order: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # g-fold rotational symmetry if all the pairwise differences |n_k-n_j| have g as their
    # greatest common divisor

    if symmetry_order:
        sym = symmetry_order
    else:
        sym = np.random.randint(2, 10)
    k = np.random.randint(3, 5)
    while True:
        n = np.arange(k) * sym + np.random.randint(-20, 10)
        n = n / np.gcd.reduce(n)
        if np.max(np.diff(n)) > 1:
            break
    a = 1 + np.random.randint(-2, 2, k) / 5
    # s = np.hstack((np.zeros(k-1), np.random.rand()))
    # np.random.shuffle(s)
    s = np.random.rand(k)
    return n, a, s


def random_symmetric_big(
    symmetry_order: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # g-fold rotational symmetry if all the pairwise differences |n_k-n_j| have g as their
    # greatest common divisor

    if symmetry_order:
        sym = symmetry_order
    else:
        sym = np.random.choice([2, 3, 5, 7, 11, 13])
    k = np.random.randint(3, 5)
    while True:
        n = np.arange(k) * sym + np.random.randint(-30, 30)
        n = n / np.gcd.reduce(n)
        if np.max(np.diff(n)) > 1:
            break
    np.random.shuffle(n)
    # a = 1 + np.random.randint(-2, 2, k) / 5
    a = 1 + np.random.randint(-2, 2, k) / 5
    s = np.random.rand(k)
    return n, a, s


@click.command()
@click.option("-n", "--count", type=int, default=10000, help="Number of segment to generate")
@generator
def whlfarris(count) -> LineCollection:
    """Generates figure 2 of Farris' 1996 paper.

    Ref: https://core.ac.uk/download/pdf/72850999.pdf
    """
    return LineCollection(
        [_wheelsonwheelsonwheels(count, [1, 7, -17], [1, 1 / 2, 1 / 3], [0, 0, 0.25])]
    )


whlfarris.help_group = "Plugins"


WHL_MODES = {
    "sym": random_symmetric,
    "sym_big": random_symmetric_big,
}


@click.command()
@click.option("-n", "--count", type=int, default=10000, help="Number of segment to generate")
@click.option("-m", "--mode", type=click.Choice(WHL_MODES.keys()), default="sym")
@click.option("-so", "--symmetry-order", type=int)
@click.option(
    "-ml", "--max-length", type=float, help="Limit the maximum length of the resulting path"
)
@generator
def whlrandom(count, mode, symmetry_order, max_length: Optional[float]) -> LineCollection:
    """Generate random 3-wheel spirograph curves."""

    func = WHL_MODES[mode]

    while True:
        line = _wheelsonwheelsonwheels(count, *func(symmetry_order))
        if max_length is None or np.sum(np.abs(np.diff(line))) < max_length:
            break

    return LineCollection([line])


whlrandom.help_group = "Plugins"


def _layout_line_collections(
    lc_map: Dict[Any, LineCollection], col_count: int, offset: Tuple[float, float]
) -> LineCollection:
    lc = LineCollection()
    for i, (key, line) in enumerate(lc_map.items()):
        lc.append(
            line + (i % col_count) * offset[0] + math.floor(i / col_count) * offset[1] * 1j
        )

    return lc


@click.command()
@click.option("-n", "--count", type=int, default=10000, help="Number of segment to generate")
@generator
def whlboard(count) -> LineCollection:
    """Systematic exploration (WIP)"""
    phase = [0, 0.25, 0.5, 0.75]
    radius = [1, 0.75, 0.5, 0.25]
    nn = [2, 3, 5, 7, 11, -1, -3, -5, -7]

    return _layout_line_collections(
        {
            # str(phases): _wheelsonwheelsonwheels(count, [1, 50, -48], [1, 1 / 2, 1 / 3], phases)
            # for phases in itertools.product(phase, phase, phase)
            # str(r): _wheelsonwheelsonwheels(count, [1, 50, -48], r, [0, 0, 0])
            # for r in itertools.product(radius, radius, radius)
            str(n): _wheelsonwheelsonwheels(count, n, [1, 1, 1], [0, 0, 0])
            for n in itertools.product(nn, nn, nn)
        },
        15,
        (8, 8),
    )


whlboard.help_group = "Plugins"
