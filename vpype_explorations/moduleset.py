import itertools
import logging
import math
import random
from typing import Iterable, Tuple, List, Union

import axi
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
# indicate if is appropriate to (randomly) swap the tile horizontally, resp. vertically.
MSET_MIRRORS = [
    [True, True],
    [False, True],
    [True, False],
    [False, False],
    [False, True],
    [False, True],
    [False, False],
    [True, False],
    [True, False],
    [False, False],
    [True, False],
    [False, True],
    [False, False],
    [True, False],
    [False, True],
    [True, True],
]


def load_module_set(
    path: str, quantization: float
) -> Tuple[List[LineCollection], float, float]:
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


def render_module_set(
    img: np.ndarray,
    mset_path: str,
    quantization: float,
    random_mirror: bool,
    return_sizes: bool = False,
) -> Union[LineCollection, Tuple[LineCollection, float, float]]:
    """
    Build a LineCollection from a 2-dimension bool Numpy array and a path to a module set
    """
    modules, tile_w, tile_h = load_module_set(mset_path, quantization)
    lc = LineCollection()
    for idx, mod_id in np.ndenumerate(bitmap_to_module(img)):
        if mod_id != -1:
            mod_lc = LineCollection(modules[mod_id])

            if random_mirror:
                mod_lc.translate(-tile_w / 2, -tile_h / 2)
                if MSET_MIRRORS[mod_id][0] and random.random() < 0.5:
                    mod_lc.scale(-1, 1)
                if MSET_MIRRORS[mod_id][1] and random.random() < 0.5:
                    mod_lc.scale(1, -1)
                mod_lc.translate(tile_w / 2, tile_h / 2)

            mod_lc.translate(idx[1] * tile_w, idx[0] * tile_h)

            lc.extend(mod_lc)

    if return_sizes:
        return lc, tile_w, tile_h
    else:
        return lc


def quantization_option(function):
    function = click.option(
        "-q",
        "--quantization",
        type=Length(),
        default="0.1mm",
        help="Quantization used when loading tiles (default: 0.1mm)",
    )(function)
    return function


def random_mirror_option(function):
    function = click.option(
        "-m",
        "--random-mirror",
        is_flag=True,
        help="Randomly mirror tiles in acceptable direction(s)",
    )(function)
    return function


def render_text(txt: str) -> LineCollection:
    lines = axi.text(txt, font=axi.hershey_fonts.FUTURAL)
    text = LineCollection()
    for line in lines:
        text.append([x + 1j * y for x, y in line])
    return text


@click.command()
@click.argument("mset", type=str)
@quantization_option
@random_mirror_option
@click.argument("bmap", type=click.Path(exists=True))
@click.option(
    "-t",
    "--threshold",
    type=click.IntRange(0, 255, True),
    default=128,
    help="Threshold applied to the image",
)
@generator
def msimage(mset, bmap, quantization, random_mirror, threshold):
    """
    Render a bitmap image with complex module (P.2.3.6). The input image is first converted to
    grayscale and then a threshold at 128 is applied
    """

    img = np.array(Image.open(bmap).convert("L")) > threshold
    return render_module_set(img, mset, quantization, random_mirror)


msimage.help_group = "Complex Modules"


@click.command()
@click.argument("mset", type=str)
@quantization_option
@random_mirror_option
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
@click.option("-s", "--symmetric", is_flag=True, help="Generate a symmetric pattern")
@click.option("-f", "--fingerprint", is_flag=True, help="Generate finger print")
@generator
def msrandom(mset, size, density, quantization, random_mirror, symmetric, fingerprint):
    """
    Render a grid with random occupancy with complex module (P.2.3.6).
    """

    img = np.random.rand(size[1], size[0]) < density

    if symmetric and size[0] > 1:
        n = math.floor(size[0] / 2)
        img[:, (size[0] - n) :] = img[:, (n - 1) :: -1]

    if fingerprint and random_mirror:
        # we need to record the seed used to generate tile swaps
        # we allow only seeds between 0 and 255
        seed = random.randint(0, 255)
        random.seed(seed)
    else:
        seed = None

    lc, tile_w, tile_h = render_module_set(
        img, mset, quantization, random_mirror, return_sizes=True
    )

    if fingerprint:
        byte_str = "".join(f"{x:02x}" for x in np.packbits(img.flatten("C")))
        txt = f"{size[0]}_{size[1]}_{byte_str}"
        if seed is not None:
            txt += f"_{seed:02x}"
        txt_lc = render_text(txt)
        txt_lc.scale(tile_h / 4 / 18)
        txt_lc.translate((size[0] * tile_w - txt_lc.width()) / 2, -tile_h / 3)
        lc.extend(txt_lc)

        logging.info(f"msrandom: fingerprint = {txt}")

    return lc


msrandom.help_group = "Complex Modules"


@click.command()
@click.argument("mset", type=str)
@click.argument("fingerprint", type=str)
@quantization_option
@generator
def msfingerprint(mset, quantization, fingerprint):
    """Generate geometries based on a previously generated fingerprint.
    """

    parts = fingerprint.split("_")
    if len(parts) < 3 or len(parts) > 4:
        logging.warning(f"msfingerprint: invalid fingerprint {fingerprint}")
        return LineCollection()

    size_x = int(parts[0])
    size_y = int(parts[1])
    data = bytearray.fromhex(parts[2])
    if len(parts) == 4:
        seed = int(parts[3], 16)
    else:
        seed = None

    img = (np.unpackbits(np.array(data), count=size_x * size_y) == 1).reshape((size_y, size_x))

    if seed is not None:
        random.seed(seed)

    return render_module_set(img, mset, quantization, random_mirror=seed is not None)


msfingerprint.help_group = "Complex Modules"


@click.command()
@generator
@click.argument("mset", type=str)
@quantization_option
@click.option("-c", "--crop-marks", is_flag=True)
def mstiles(mset, quantization, crop_marks) -> LineCollection:
    """
    Create a nice representation of all the module set as it would be used by other commands.
    """

    # load modules
    module_list, width, height = load_module_set(mset, quantization)

    # parameters
    ext_margin = 3
    line = 1
    int_margin = 0.5
    tile_width = 6

    border = ext_margin + line + int_margin
    step = 2 * border + tile_width

    # prepare marks
    if crop_marks:
        h_marks = np.array(
            [
                [ext_margin + 1j * border, ext_margin + line + 1j * border],
                [step - ext_margin - line + 1j * border, step - ext_margin + 1j * border],
                [
                    ext_margin + 1j * (border + tile_width),
                    ext_margin + line + 1j * (border + tile_width),
                ],
                [
                    step - ext_margin - line + 1j * (border + tile_width),
                    step - ext_margin + 1j * (border + tile_width),
                ],
            ],
            dtype=complex,
        )
        v_marks = np.empty(shape=h_marks.shape, dtype=complex)
        v_marks.imag = h_marks.real
        v_marks.real = h_marks.imag
        marks = np.concatenate([h_marks, v_marks])
    else:
        marks = np.array(
            [
                [
                    border + 1j * border,
                    border + tile_width + 1j * border,
                    border + tile_width + 1j * (border + tile_width),
                    border + 1j * (border + tile_width),
                    border + 1j * border,
                ]
            ]
        )

    # render everything
    lc = LineCollection()
    for i, j in itertools.product(range(4), range(4)):
        lc.extend(marks + i * step + j * 1j * step)
        idx = i + j * 4
        tile = LineCollection(module_list[idx])
        tile.scale(tile_width / width, tile_width / height)
        tile.translate(border + i * step, border + j * step)
        lc.extend(tile)

        # add title
        text = render_text(MSET_SUFFIX[idx])
        text.scale(1 / 18)
        bounds = text.bounds()
        text.translate(
            border + i * step + (tile_width - bounds[2] + bounds[0]) / 2,
            ext_margin + line / 2 + j * step,
        )
        lc.extend(text)

    return lc


mstiles.help_group = "Complex Modules"
