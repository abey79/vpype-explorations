from __future__ import annotations

import math

import click
import numpy as np
import vpype as vp
import vpype_cli
from shapely.geometry import MultiLineString, Polygon
from shapely.ops import unary_union

DEFAULT_PEN_WIDTH = vp.convert_length("0.3mm")


def _generate_fill(poly: Polygon, pen_width: float) -> vp.LineCollection:

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

    lc = vp.LineCollection(mls)
    lc.merge(tolerance=pen_width * 5, flip=True)

    boundary = p.boundary
    if boundary.geom_type == "MultiLineString":
        lc.extend(boundary)
    else:
        lc.append(boundary)
    return lc


@click.command()
@click.option(
    "-pw",
    "--pen-width",
    type=vpype_cli.LengthType(),
    help="Pen width (default: current layer's pen width or 0.3mm)",
)
@click.option(
    "-t",
    "--tolerance",
    type=vpype_cli.LengthType(),
    default="0.01mm",
    help="Max distance between start and end point to consider a path closed "
    "(default: 0.01mm)",
)
@click.option("-k", "--keep-open", is_flag=True, help="Keep open paths")
@vpype_cli.layer_processor
def fill(
    lines: vp.LineCollection, pen_width: float | None, tolerance: float, keep_open: bool
) -> vp.LineCollection:
    """Horizontal hatch fill.

    If `--pen-width` is not used, the layer's pen width is used reverting, if unset, to 0.3mm.
    """
    if pen_width is None:
        pen_width = lines.property(vp.METADATA_FIELD_PEN_WIDTH) or DEFAULT_PEN_WIDTH

    new_lines = lines.clone()
    polys = []
    for line in lines:
        if np.abs(line[0] - line[-1]) <= tolerance:
            polys.append(Polygon([(pt.real, pt.imag) for pt in line]))
        elif keep_open:
            new_lines.append(line)

    # merge all polygons and fill the result
    mp = unary_union(polys)
    if mp.geom_type == "Polygon":
        mp = [mp]

    for p in mp:
        new_lines.extend(_generate_fill(p, pen_width))

    return new_lines


fill.help_group = "Plugins"


@click.command()
@click.option(
    "-pw",
    "--pen-width",
    type=vpype_cli.LengthType(),
    help="Pen width (default: current layer's pen width or 0.3mm)",
)
@click.option(
    "-t",
    "--tolerance",
    type=vpype_cli.LengthType(),
    default="0.01mm",
    help="Max distance between start and end point to consider a path closed "
    "(default: 0.01mm)",
)
@vpype_cli.layer_processor
def cfill(
    lines: vp.LineCollection, pen_width: float | None, tolerance: float
) -> vp.LineCollection:
    """Concentric fill.

    If `--pen-width` is not used, the layer's pen width is used reverting, if unset, to 0.3mm.
    """
    if pen_width is None:
        pen_width = lines.property(vp.METADATA_FIELD_PEN_WIDTH) or DEFAULT_PEN_WIDTH

    new_lines = lines.clone()
    for line in lines:
        new_lines.append(line)
        if np.abs(line[0] - line[-1]) <= tolerance:
            p = Polygon((pt.real, pt.imag) for pt in line)
            while True:
                p = p.buffer(-pen_width)
                if p.is_valid and len(p.exterior.coords) > 1:
                    new_lines.append(p.exterior)
                else:
                    break

    return new_lines


fill.help_group = "Plugins"
