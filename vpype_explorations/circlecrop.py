import click
import vpype as vp
from shapely.geometry import Polygon


@click.command()
@click.argument("X", type=vp.LengthType())
@click.argument("Y", type=vp.LengthType())
@click.argument("R", type=vp.LengthType())
@click.option(
    "-q",
    "--quantization",
    type=vp.LengthType(),
    default="0.1mm",
    help="Quantization used for the circular area",
)
@vp.layer_processor
def circlecrop(lines: vp.LineCollection, x: float, y: float, r: float, quantization: float):
    """Crop to a circular area."""

    circle = Polygon(vp.as_vector(vp.circle(x, y, r, quantization)))
    mls = lines.as_mls()
    return vp.LineCollection(mls.intersection(circle))
