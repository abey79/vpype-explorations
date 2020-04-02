import itertools
import logging
import math
import random
from typing import Tuple, Optional

import click
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import unary_union
from vpype import LineCollection, Length, global_processor, VectorData

RectType = Tuple[float, float, float, float]


def rect_to_polygon(rect: RectType) -> Polygon:
    return Polygon(
        [
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            (rect[0], rect[1] + rect[3]),
        ]
    )


def generate_fill(rect: RectType, pen_width: float) -> LineCollection:
    line_count = math.ceil(rect[3] / pen_width)

    base_seg = np.array([pen_width / 2, rect[2] - pen_width / 2]) + rect[0]
    y_start = rect[1] + (rect[3] - (line_count - 1) * pen_width) / 2

    segs = []
    for n in range(line_count):
        seg = base_seg + (y_start + pen_width * n) * 1j
        segs.append(seg if n % 2 == 0 else np.flip(seg))

    return LineCollection([np.hstack(segs)])


def generate_gradient(
    rect: RectType, line: np.ndarray, density: float = 1.0
) -> LineCollection:
    """Generate a random dots with a gradient density distribution. `density` is global average
    number of point per square pixel
    """

    n = int((rect[2] * rect[3]) * density)
    orig = np.random.uniform(rect[0], rect[0] + rect[2], n) + 1j * np.random.triangular(
        rect[1], rect[1], rect[1] + rect[3], n
    )
    lines = orig.reshape(n, 1) + line.reshape(1, len(line))
    return LineCollection(lines)


def generate_dot_gradient(
    rect: RectType, pen_width: float, density: float = 1.0
) -> LineCollection:
    """Generate a random dots with a gradient density distribution. `density` is global average
    number of point per square pixel
    """
    return generate_gradient(rect, np.array([0, pen_width * 0.05]), density)


def generate_big_dot_gradient(
    rect: RectType, pen_width: float, turns: float, density: float
) -> LineCollection:
    """Generate random big dots with a gradient density distribution."""

    s = np.cumsum(np.flip(np.linspace(0, 1, int(20 * (turns + 1))) ** 0.5))
    s /= s[-1]
    t = s / s[-1] * math.pi * 2 * (turns + 1)
    r = np.clip(t * pen_width / (2 * math.pi), a_min=0, a_max=turns * pen_width)
    line = r * (np.cos(t) + 1j * np.sin(t))
    lc = generate_gradient(rect, line, density)
    mls = lc.as_mls()
    return LineCollection(mls.intersection(rect_to_polygon(rect)))


def generate_star(rect: RectType, line_count: int = 20) -> LineCollection:
    """Generate a set of line from a random point."""

    orig_x = np.random.uniform(rect[0], rect[0] + rect[2])
    orig_y = np.random.uniform(rect[1], rect[1] + rect[3])
    r = math.hypot(rect[2], rect[3])
    angles = np.linspace(0, 2 * math.pi, num=line_count, endpoint=False)
    phase = np.random.normal(0, math.pi / 4)

    mls = MultiLineString(
        [
            ([orig_x, orig_y], [orig_x + r * math.cos(a), orig_y + r * math.sin(a)])
            for a in angles + phase
        ]
    )

    return LineCollection(mls.intersection(rect_to_polygon(rect)))


def generate_hatch(rect: RectType) -> LineCollection:
    """Generate hatching patter with random orientation and density"""

    angle = np.random.uniform(0, math.pi * 2)
    step = np.random.uniform(0.05 * min(rect[2:4]), 0.2 * min(rect[2:4]))
    r = math.hypot(rect[2], rect[3])
    ys = np.arange(-r, r, step)
    mls = translate(
        rotate(MultiLineString([[(-r, y), (r, y)] for y in ys]), angle, use_radians=True),
        rect[0] + 0.5 * rect[2],
        rect[1] + 0.5 * rect[3],
    )
    return LineCollection(mls.intersection(rect_to_polygon(rect)))


def distribute_widths(n, size):
    w = np.random.uniform(1, 10, n)
    return w / w.sum() * size


def check_default(rate: float, glob: float) -> float:
    if rate is None:
        return glob if glob is not None else 0
    else:
        return rate


def mls_parallel_offset(mls: MultiLineString, distance: float, side: str):
    return unary_union([line.parallel_offset(distance, side) for line in mls])


@click.command()
@click.option("-r", "--seed", type=int)
@click.option("-s", "--size", nargs=2, type=Length(), default=["10cm", "10cm"])
@click.option("-n", "--count", nargs=2, type=int, default=[5, 5])
@click.option("-pw", "--pen-width", type=Length(), default="0.3mm")
@click.option("-fg", "--fat-grid", is_flag=True, default=False)
@click.option("-g", "--global-rate", type=click.FloatRange(0.0, 1.0), default=0.1)
@click.option("-rf", "--rate-fill", type=click.FloatRange(0.0, 1.0), multiple=True)
@click.option("-rg", "--rate-gradient", type=click.FloatRange(0.0, 1.0))
@click.option("-rb", "--rate-bigdot", type=click.FloatRange(0.0, 1.0))
@click.option("-rs", "--rate-star", type=click.FloatRange(0.0, 1.0))
@click.option("-rh", "--rate-hatch", type=click.FloatRange(0.0, 1.0))
@global_processor
def mdgrid(
    vector_data: VectorData,
    seed: Optional[int],
    size,
    count,
    pen_width,
    fat_grid,
    global_rate,
    rate_fill,
    rate_gradient,
    rate_bigdot,
    rate_star,
    rate_hatch,
):
    """Create nice random grids with stuff in them.
    """

    if len(rate_fill) == 0 and global_rate is not None:
        rate_fill = [global_rate]
    rate_gradient = check_default(rate_gradient, global_rate)
    rate_bigdot = check_default(rate_bigdot, global_rate)
    rate_star = check_default(rate_star, global_rate)
    rate_hatch = check_default(rate_hatch, global_rate)

    logging.info(
        f"mdgrid: rates: fill = {rate_fill}, gradient = {rate_gradient}, "
        f"bigdot = {rate_bigdot}, star = {rate_star}, hatch = {rate_hatch}"
    )

    # handle seed
    if seed is None:
        seed = random.randint(0, 1e9)
        logging.info(f"mdgrid: no seed provided, generating one ({seed})")
    np.random.seed(seed)
    random.seed(seed)

    grid_lc = LineCollection()

    # build the grid
    col_widths = distribute_widths(count[0], size[0])
    row_widths = distribute_widths(count[1], size[1])
    col_seps = np.hstack([0, np.cumsum(col_widths)])
    row_seps = np.hstack([0, np.cumsum(row_widths)])

    # outer boundaries must be a single loop (for fat grid to work nicely)
    grid_lc.append(
        [
            col_seps[0] + row_seps[0] * 1j,
            col_seps[0] + row_seps[-1] * 1j,
            col_seps[-1] + row_seps[-1] * 1j,
            col_seps[-1] + row_seps[0] * 1j,
            col_seps[0] + row_seps[0] * 1j,
        ]
    )
    grid_lc.extend([x + row_seps[0] * 1j, x + row_seps[-1] * 1j] for x in col_seps)
    grid_lc.extend([y * 1j + col_seps[0], y * 1j + col_seps[-1]] for y in row_seps)

    # implement fat grid
    fat_grid_lc = LineCollection()
    if fat_grid:
        mls = grid_lc.as_mls()
        fat_grid_lc.extend(
            unary_union(
                [
                    mls_parallel_offset(mls, pen_width, "left"),
                    mls_parallel_offset(mls, pen_width, "right"),
                ]
            )
        )

    # generate content in each cell
    fill_lcs = [LineCollection() for _ in range(len(rate_fill))]
    grad_lc = LineCollection()
    bigdot_lc = LineCollection()
    star_lc = LineCollection()
    hatch_lc = LineCollection()
    for (x, y) in itertools.product(range(count[0]), range(count[1])):
        rect = (
            col_seps[x],
            row_seps[y],
            col_seps[x + 1] - col_seps[x],
            row_seps[y + 1] - row_seps[y],
        )

        filled = False
        for i, r in enumerate(rate_fill):
            if random.random() < r:
                fill_lcs[i].extend(generate_fill(rect, pen_width))
                filled = True
                break
        if not filled:
            if random.random() < rate_gradient:
                grad_lc.extend(generate_dot_gradient(rect, pen_width, density=0.3))
            elif random.random() < rate_bigdot:
                bigdot_lc.extend(generate_big_dot_gradient(rect, pen_width, 3, density=0.01))
            elif random.random() < rate_star:
                star_lc.extend(generate_star(rect, line_count=20))
            elif random.random() < rate_hatch:
                hatch_lc.extend(generate_hatch(rect))

    # populate vector data with layer content
    vector_data.add(grid_lc, 1)
    vector_data.add(fat_grid_lc, 2)

    vector_data.add(star_lc, 3)
    vector_data.add(hatch_lc, 4)

    vector_data.add(grad_lc, 5)

    vector_data.add(bigdot_lc, 6)

    for i, lc in enumerate(fill_lcs):
        vector_data.add(lc, 7 + i)

    return vector_data


mdgrid.help_group = "Plugins"
