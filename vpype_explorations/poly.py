import click
import numpy as np
from vpype import LineCollection
from vpype_cli import LengthType, generator


@click.command()
@click.option(
    "-c",
    "--coords",
    type=LengthType(),
    nargs=2,
    multiple=True,
    help="X and Y coordinates of the point (can be used multiple time, accepts usual units)",
)
@generator
def poly(coords):
    """Generate a single path with coordinates provided by the sequence of `--cords` option.

    Example:

        vpype poly coord -c 0 0 -c 1 0 -c 0 1 show
    """
    if len(coords) == 0:
        return LineCollection()
    else:
        return LineCollection([np.array([c[0] + 1j * c[1] for c in coords])])
