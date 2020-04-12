from typing import Union, List

import click
import numpy as np
from vpype import global_processor, VectorData, LineCollection, LayerType


def _transform_line(line: np.ndarray, delta: float) -> np.ndarray:
    new_line = np.array(line, dtype=complex)
    new_line.imag = (
            -delta * new_line.real
            + new_line.imag
            + 2 * delta * new_line.real * new_line.imag
    )
    return new_line


@click.command()
@click.argument("delta", type=float)
@click.option(
    "-l",
    "--layer",
    type=LayerType(accept_multiple=True),
    default="all",
    help="Target layer(s).",
)
@global_processor
def fake3d(vector_data: VectorData, layer: Union[int, List[int]], delta: float):
    """
    Duplicate layers and distort them to emulate fake anaglyph 3D. The distortion works by
    mapping the unit square to a trapeze whose:
    - width is 1
    - left and right sides are parallel
    - left side length is 1
    - right side span from -delta to 1 + delta

    Based on https://math.stackexchange.com/a/863702
    CAUTION: highly hacky plugin, use at your own risk :)
    """

    layer_ids = LayerType.multiple_to_layer_ids(layer, vector_data)

    for layer_id in layer_ids:
        # copy the layer
        left_lc = LineCollection(vector_data[layer_id])

        # transform existing layer
        for idx, line in enumerate(vector_data[layer_id]):
            vector_data[layer_id].lines[idx] = _transform_line(line, delta)

        # compute data for new layer
        w = left_lc.width()
        left_lc.scale(-1, 1)
        left_lc.translate(w, 0)
        for idx, line in enumerate(left_lc):
            left_lc.lines[idx] = _transform_line(line, delta)
        left_lc.translate(-w, 0)
        left_lc.scale(-1, 1)

        vector_data[vector_data.free_id()] = left_lc

    return vector_data


fake3d.help_group = "Plugins"

