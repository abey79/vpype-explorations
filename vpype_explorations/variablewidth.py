import logging

import click
import cv2
import numpy as np
from scipy import interpolate
from shapely.affinity import translate
from shapely.geometry import LinearRing, Polygon, MultiLineString, LineString
from shapely.ops import unary_union
from skimage import measure
from vpype import LengthType, generator, LineCollection


def pixel_to_half_width(
    pixel_line: np.ndarray, pitch, pen_width, black_level, white_level
) -> np.ndarray:
    """ Convert pixel values to half stripe width
    """

    # converted_values = np.where(
    #    pixel_line < black_level, 1, 1 - pixel_line / (1 - black_level)
    # )

    # converted_values = (
    #     0.5
    #     * (pitch - pen_width)
    #     * (2 * black_level - white_level + pixel_line)
    #     / (black_level - white_level)
    # )
    # return np.clip(converted_values, 0.0, 0.5 * (pitch - pen_width))

    converted_value = np.clip(
        1 - ((pixel_line - black_level) / (white_level - black_level)), 0, 1
    )

    return converted_value * 0.5 * (pitch - pen_width)


def create_hatch_polygon(pixel_line, pitch, pen_width, black_level, white_level) -> Polygon:
    """ Create the horizontal outline of a line for a given linear array of pixel
    """
    half_width = pixel_to_half_width(pixel_line, pitch, pen_width, black_level, white_level)

    # ensure the polygon is not collapsed
    min_hw = 0.05 * pitch
    half_width[half_width < min_hw] = min_hw

    t = pitch * np.arange(len(pixel_line))
    spline = interpolate.splrep(t, half_width, s=0.5)
    xx = np.linspace(0, t[-1], 10 * (len(t) - 1))
    yy = interpolate.splev(xx, spline)
    coords = zip(list(xx) + list(reversed(xx)), list(yy) + list(reversed(-yy)))

    # a call to simplify is added here to reduce the number of points in the polygon and
    # optimize a bit the execution time (instead of calling linesimplify downstream)
    return Polygon(coords).simplify(tolerance=pen_width / 20)


def fill_polygon(p: Polygon, pen_width: float) -> MultiLineString:
    minx, miny, maxx, maxy = p.bounds

    result = []
    while not p.is_empty:
        if p.geom_type == "Polygon":
            result.append(p.boundary)
        elif p.geom_type == "MultiPolygon":
            result.extend(poly.boundary for poly in p)
        p = p.buffer(-pen_width)

    mls = unary_union(result)

    # we add a center line to be cut with a buffered version of the hatching lines
    c = 0.5 * (miny + maxy)
    ls = LineString([(minx, c), (maxx, c)])
    return mls.union(ls.difference(mls.buffer(0.55 * pen_width)))


def build_mask(cnt):
    lr = [LinearRing(p[:, [1, 0]]) for p in cnt if len(p) >= 4]

    mask = None
    for r in lr:
        if not r.is_valid:
            continue

        if mask is None:
            mask = Polygon(r)
        else:
            if r.is_ccw:
                mask = mask.union(Polygon(r).buffer(0.5))
            else:
                mask = mask.difference(Polygon(r).buffer(-0.5))

    return mask


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-s", "--scale", default=1.0, help="Scale factor to apply to the image size")
@click.option(
    "-p", "--pitch", default=1, type=LengthType(), help="Resulting size per pixel (default: 1.0)"
)
@click.option(
    "-pw",
    "--pen-width",
    default="0.3mm",
    type=LengthType(),
    help="Stroke width of the pen (default: 0.3mm)",
)
@click.option(
    "-bl", "--black-level", default=0.0, help="Black clipping level ([0-1], default: 0.0)"
)
@click.option(
    "-wl", "--white-level", default=1.0, help="White clipping level ([0-1], default: 1.0)"
)
@click.option("-d", "--delete-white", is_flag=True, help="Delete lines in white region")
@click.option(
    "-a",
    "--outline-alpha",
    count=True,
    help="Increase thickness of the boundary outline (can be repeated, uses PNG transparency)",
)
@click.option(
    "-i", "--invert", is_flag=True, default=False, help="Invert the image before processing"
)
@generator
def variablewidth(
    filename,
    scale,
    pitch,
    pen_width,
    black_level,
    white_level,
    delete_white,
    outline_alpha,
    invert,
):
    """Documentation todo

    """

    # load grayscale image data
    logging.info("variablewidth: loading image")
    orig_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED) / 255.0
    gray_img = np.average(orig_img[:, :, 0:3], axis=2)
    img = cv2.resize(np.array(gray_img, dtype=float), None, fx=scale, fy=scale)
    if invert:
        img = 1.0 - img
    logging.info(f"variablewidth: final image size: {img.shape[0]}x{img.shape[1]}")

    # load alpha layer if requested
    alpha_img = None
    if outline_alpha > 0:
        if orig_img.shape[2] == 4:
            alpha_img = orig_img[:, :, 3]
        else:
            logging.warning(
                "variablewidth: outline alpha requested but input image has no alpha layer"
            )

    # create base line work into a MultiLineString
    logging.info("variablewidth: creating line work")
    mls_arr = []
    for j in range(img.shape[0]):
        pixel_line = img[j]
        poly = translate(
            create_hatch_polygon(
                pixel_line,
                pitch=pitch,
                pen_width=pen_width,
                black_level=black_level,
                white_level=white_level,
            ),
            yoff=j * pitch,
        )
        if not poly.is_empty:
            mls_arr.append(fill_polygon(poly, pen_width))
    base_mls = unary_union(mls_arr)

    # deal with white
    if delete_white:
        logging.info("variablewidth: deleting white")
        white_cnt = [c * pitch for c in measure.find_contours(img, white_level)]
        white_mask = build_mask(white_cnt)
        base_mls = base_mls.difference(white_mask)

    # generate outline
    additional_mls = []
    if alpha_img is not None:
        logging.info("variablewidth: outlining alpha")
        cnt = [c * pitch * scale for c in measure.find_contours(alpha_img, 0.5) if len(c) >= 4]
        mask = build_mask(cnt)

        base_mls = base_mls.intersection(mask)

        # we want all boundaries, including possible holes
        bounds_mls = unary_union([mask.boundary] + [lr for lr in mask.interiors]).simplify(
            tolerance=pen_width / 2
        )
        additional_mls.append(bounds_mls)

        # if multiple
        additional_mls.extend(
            bounds_mls.parallel_offset((i + 1) * pen_width, side)
            for side in ("left", "right")
            for i in range(outline_alpha - 1)
        )

    lc = LineCollection(base_mls)
    for mls in additional_mls:
        lc.extend(mls)
    return lc


variablewidth.help_group = "Plugins"
