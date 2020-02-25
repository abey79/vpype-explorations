import math
import random

import click
from shapely import affinity, ops
from shapely.geometry import MultiLineString
from vpype.model import LineCollection
from vpype.decorators import generator


def append_maybe(item, lst):
    if random.random() < 0.5:
        lst.append(item)


@click.command()
@click.option(
    "-N",
    "--count",
    nargs=2,
    default=(5, 6),
    type=int,
    help="Number of segments in X and Y direction",
)
@generator
def alien(count):
    """
    Generate an alien looking glyph based on segment connecting node of a regular grid.

    The grid size is controlled with the `--count N M` option. The generated grid has its
    origin on (0, 0) and a size of (N-1, M-1) (i.e. the grid's node lie on integer
    coordinates).

    Each neighbouring node is connected with horizontal and/or vertical segment with a
    probability of 50%. The pattern is mirrored.
    """

    segs = []

    # horizontal segments
    for i in range(math.floor(count[0] / 2)):
        for j in range(count[1]):
            append_maybe([(i, j), (i + 1, j)], segs)

    # add half horizontal segments
    if count[0] % 2 == 0:
        for j in range(count[1]):
            append_maybe([((count[0] / 2) - 1, j), (count[0] / 2, j)], segs)

    # add vertical segments
    for i in range(math.ceil(count[0] / 2)):
        for j in range(count[1] - 1):
            append_maybe([(i, j), (i, j + 1)], segs)

    half_mls = MultiLineString(segs)
    other_mls = affinity.translate(
        affinity.scale(half_mls, -1, 1, origin=(0, 0)), count[0] - 1, 0
    )

    lc = LineCollection(ops.linemerge(ops.unary_union([half_mls, other_mls])))

    return lc


alien.help_group = "Plugins"
