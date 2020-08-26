import math

import click
import numpy as np
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import unary_union
from vpype import layer_processor, LineCollection, Length, as_vector


def _generate_fill(poly: Polygon, pen_width: float) -> LineCollection:

    # nasty hack because unary_union() did something weird once
    poly = Polygon(poly.exterior)

    # we draw the boundary, accounting for pen width
    p = poly.buffer(-pen_width / 2)

    min_x, min_y, max_x, max_y = p.bounds
    height = max_y - min_y
    line_count = math.ceil(height / pen_width) + 1
    base_seg = np.array([min_x, max_x])
    y_start = min_y + (height - (line_count - 1) * pen_width) / 2

    segs = []
    for n in range(line_count):
        seg = base_seg + (y_start + pen_width * n) * 1j
        segs.append(seg if n % 2 == 0 else np.flip(seg))

    mls = MultiLineString([[(pt.real, pt.imag) for pt in seg] for seg in segs]).intersection(
        p.buffer(-pen_width / 2)
    )

    lc = LineCollection(mls)
    lc.merge(tolerance=pen_width*5, flip=True)
    print(p.geom_type)

    boundary = p.boundary
    if boundary.geom_type == "MultiLineString":
        lc.extend(boundary)
    else:
        lc.append(boundary)
    return lc


@click.command()
@click.option(
    "-pw", "--pen-width", type=Length(), default="0.3mm", help="Pen width (default: 0.3mm)"
)
@click.option(
    "-t",
    "--tolerance",
    type=Length(),
    default="0.01mm",
    help="Max distance between start and end point to consider a path closed "
    "(default: 0.01mm)",
)
@click.option("-k", "--keep-open", is_flag=True, help="Keep open paths")
@layer_processor
def fill(
    lines: LineCollection, pen_width: float, tolerance: float, keep_open: bool
) -> LineCollection:

    new_lines = LineCollection()
    for line in lines:
        if np.abs(line[0] - line[-1]) <= tolerance:
            p = Polygon([(pt.real, pt.imag) for pt in line])
            new_lines.extend(_generate_fill(p, pen_width))
        elif keep_open:
            new_lines.append(line)

    return new_lines


fill.help_group = "Plugins"
