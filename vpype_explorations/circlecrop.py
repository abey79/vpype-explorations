import click
import vpype as vp
import vpype_cli
from shapely.geometry import Polygon


@click.command()
@click.argument("X", type=vpype_cli.LengthType())
@click.argument("Y", type=vpype_cli.LengthType())
@click.argument("R", type=vpype_cli.LengthType())
@click.option(
    "-q",
    "--quantization",
    type=vpype_cli.LengthType(),
    default="0.1mm",
    help="Quantization used for the circular area",
)
@vpype_cli.layer_processor
def circlecrop(lines: vp.LineCollection, x: float, y: float, r: float, quantization: float):
    """Crop to a circular area."""

    circle = Polygon(vp.as_vector(vp.circle(x, y, r, quantization)))
    mls = lines.as_mls()
    return vp.LineCollection(mls.intersection(circle))
