import math
from typing import Iterable, Optional, Tuple, List

import click
import numpy as np
from PIL import Image
from vpype import generator, read_svg, Length, LineCollection

MSET_SUFFIX = [f"{i:04b}" for i in range(16)]
MSET_CONVERSIONS = {
    "0010": ("0001", math.pi / 2),
    "0100": ("0001", -math.pi),
    "0110": ("0011", math.pi / 2),
    "1000": ("0001", -math.pi / 2),
    "1001": ("0011", -math.pi / 2),
    "1010": ("0101", -math.pi / 2),
    "1011": ("0111", -math.pi / 2),
    "1100": ("0011", -math.pi),
    "1101": ("0111", -math.pi),
    "1110": ("0111", math.pi / 2),
}


def load_module_set(
    path: str, quantization: float
) -> Optional[Tuple[List[LineCollection], float, float]]:
    """
    Load a module set. If tiles are missing, they are created from existing tiles using
    rotation.
    """

    modules = {}
    for suffix in MSET_SUFFIX:
        try:
            modules[suffix] = read_svg(
                f"{path}_{suffix}.svg",
                quantization=quantization,
                simplify=True,
                return_size=True,
            )
        except FileNotFoundError:
            modules[suffix] = None

    # check equality on the sizes
    if not check_equality(
        m[1] for m in modules.values() if m is not None
    ) or not check_equality(m[2] for m in modules.values() if m is not None):
        raise RuntimeError(f"tile set `{path}` has inconsistent size")

    width = modules["0000"][1]
    height = modules["0000"][2]

    # check missing modules
    for suffix in modules:
        if modules[suffix] is None:
            conv = MSET_CONVERSIONS[suffix]
            lc = LineCollection(modules[conv[0]][0])
            lc.translate(-width / 2, -height / 2)
            lc.rotate(conv[1])
            lc.translate(width / 2, height / 2)
            modules[suffix] = (lc, width, height)

    return [modules[suffix][0] for suffix in MSET_SUFFIX], width, height


def check_equality(iterator: Iterable) -> bool:
    """Return True if all items of iterator are the same
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def bitmap_to_module(bmap: np.ndarray) -> np.ndarray:
    """Convert a bitmap (array of bool) to an array of module index (0-15) based on adjacency.
    In empty cells (False), -1 is returned
    """

    padded = np.pad(bmap, pad_width=1, mode="constant", constant_values=0).astype("int8")
    north = padded[0:-2, 1:-1]
    south = padded[2:, 1:-1]
    west = padded[1:-1, 0:-2]
    east = padded[1:-1, 2:]

    return np.where(bmap, east + 2 * south + 4 * west + 8 * north, -1)


def render_module_set(img: np.ndarray, mset_path: str, quantization: float) -> LineCollection:
    """
    Build a LineCollection from a 2-dimension bool Numpy array and a path to a module set
    """
    modules, tile_w, tile_h = load_module_set(mset_path, quantization)
    lc = LineCollection()
    for idx, mod_id in np.ndenumerate(bitmap_to_module(img)):
        if mod_id != -1:
            mod_lc = LineCollection(modules[mod_id])
            mod_lc.translate(idx[1] * tile_w, idx[0] * tile_h)
            lc.extend(mod_lc)

    return lc


@click.command()
@click.argument("mset", type=str)
@click.argument("bmap", type=click.Path(exists=True))
@click.option("-q", "--quantization", type=Length(), default="0.1mm")
@click.option(
    "-t",
    "--threshold",
    type=click.IntRange(0, 255, True),
    default=128,
    help="Threshold applied to the image",
)
@generator
def msimage(mset, bmap, quantization, threshold):
    """
    Render a bitmap image with complex module (P.2.3.6). The input image is first converted to
    grayscale and then a threshold at 128 is applied
    """

    img = np.array(Image.open(bmap).convert("L")) > threshold
    return render_module_set(img, mset, quantization)


msimage.help_group = "Module Set"


@click.command()
@click.argument("mset", type=str)
@click.option(
    "-n",
    "--size",
    nargs=2,
    type=int,
    default=(10, 10),
    help="Size of the random grid (default: 10x10)",
)
@click.option(
    "-d",
    "--density",
    type=click.FloatRange(0, 1, True),
    default=0.5,
    help="Occupancy probability ([0, 1], default: 0.5)",
)
@click.option("-q", "--quantization", type=Length(), default="0.1mm")
@generator
def msrandom(mset, size, density, quantization):
    """
    Render a grid with random occupancy with complex module (P.2.3.6).
    """

    img = np.random.rand(size[0], size[1]) > density
    return render_module_set(img, mset, quantization)


msrandom.help_group = "Module Set"
