import click
import numpy as np
from vpype.decorators import generator
from vpype.model import LineCollection
from vpype.utils import Length


@click.command()
@click.option(
    "-s", "--size", nargs=2, default=[10.0, 10.0], type=Length(), help="",
)
@click.option(
    "-p", "--pitch", default=1, type=Length(), help="",
)
@generator
def fracture(size, pitch):
    """
    """

    width = size[0]
    height = size[1]
    count = round(height / pitch + 1)

    white_start = np.clip(
        np.random.normal(scale=0.1 * width, size=count) + 0.4 * width,
        0.05 * width,
        0.65 * width,
    )
    white_stop = np.clip(
        np.random.normal(scale=0.1 * width, size=count) + 0.6 * width,
        0.35 * width,
        0.95 * width,
    )

    # avoid any contact between left-hand and right-hand set of lines
    white_start = np.clip(white_start, a_min=None, a_max=white_stop - 0.05 * width)
    white_stop = np.clip(white_stop, a_min=white_start + 0.05 * width, a_max=None)
    white_start[1:] = np.clip(
        white_start[1:], a_min=None, a_max=white_stop[:-1] - 0.05 * width
    )
    white_stop[1:] = np.clip(white_stop[1:], a_min=white_start[:-1] + 0.05 * width, a_max=None)

    # generate line collection, taking care of line order and start
    return LineCollection(
        [np.array([0, white_start[i]]) + 1j * i * pitch for i in range(count)]
        + [np.array([width, white_stop[i]]) + 1j * i * pitch for i in range(count)]
    )


fracture.help_group = "Plugins"
