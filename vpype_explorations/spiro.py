import math
from typing import Iterable

import click
import numpy as np
from shapely.geometry import Polygon, LineString
from vpype import LineCollection, layer_processor, interpolate_line


def interpolate(p0, p1, max_step):
    """Interpolate between p0 and p1 with steps close to, but smaller than, max_step, omitting
    p1
    """
    dp = np.array(p1) - np.array(p0)
    norm = np.linalg.norm(dp)
    step = math.ceil(norm / max_step)
    for i in range(step):
        yield p0 + i / step * dp


def interpolate_polygon(poly, max_step=0.1):
    interp = []
    for p1, p2 in circular_pairwise(poly):
        interp.extend(interpolate(p1, p2, max_step))
    return np.array(interp)


def circular_pairwise(arr):
    ln = len(arr)
    yield from zip(arr, (arr[(i + 1) % ln] for i in range(ln)))


def curvilinear_abscissa(arr):
    """
    Compute the curvilinear abscissa of an array of 2D points.

    """
    return np.cumsum(
        np.linalg.norm(np.diff(np.append(arr, [arr[0]], axis=0), axis=0), axis=1), axis=0,
    )


def spyro(
    template: Iterable[complex], k: int = 101, q: int = 11, d: float = 1, b: int = 1.2,
) -> np.ndarray:
    """
    Calculate a spirograph trajectory based on a template.
    :param template: CCW list of point of closed polygon to be used as template
    :param k: total number of spirograph rotation
    :param q: total number of template rotation (k and q should be prime to each other)
    :param d: distance to template shape
    :param b: generating relative radius (b = 1 cycloid)
    :return: computed spirograph trajectory
    """
    # Iterate over template and interpolate as required
    template_xy = [(x.real, x.imag) for x in template]
    if template_xy[0] == template_xy[-1]:
        base_shape = Polygon(template_xy)
    else:
        base_shape = LineString(template_xy)

    # convert trajectory back to complex
    trajectory_xy = np.array(base_shape.buffer(2 * d).buffer(-d).boundary)
    trajectory = np.array(trajectory_xy[:, 0], dtype=complex)
    trajectory.imag = trajectory_xy[:, 1]

    # interpolate the trajectory
    trajectory = interpolate_line(trajectory, step=0.1)

    # Compute template's curvilinear abscissa
    curv_absc = np.cumsum(np.hstack([0, np.abs(np.diff(trajectory))]))

    # Compute circumference of spirograph such that k complete rotations of the spirograph will
    # match q complete rotations of the template
    circum = q / k * curv_absc[-1]
    radius = circum / 2 / math.pi
    gen_radius = b * radius

    # generate
    last_angle_start = 0
    output_arr = []
    dp = np.diff(trajectory, prepend=trajectory[-1])
    delta_angle = np.cumsum(np.abs(dp)) / radius
    for _ in range(q):
        cur_angle = last_angle_start + delta_angle
        output_arr.append(
            trajectory + gen_radius * (np.cos(cur_angle) + 1j * np.sin(cur_angle))
        )
        last_angle_start = cur_angle[-1]

    return np.hstack(output_arr)


@click.command()
@layer_processor
def spiro(lines: LineCollection) -> LineCollection:
    """Generate a spirographic pattern around existing geometries
    """

    return LineCollection([spyro(line, 101, 100, 5, 1) for line in lines])
