import time
import cv2
import numpy as np
from numba import njit, prange
from kiseki.logging import Profiler
from linefiller.trappedball_fill import (
    exclude_area,
    fast_where,
    get_unfilled_point,
    trapped_ball_fill_multi as original_trapped_ball_fill_multi,
    flood_fill_multi as original_flood_fill_multi,
    merge_fill as original_merge_fill,
)

try:
    from . import trappedballcpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("Warning: C++ module not available, using Python fallback")

if HAS_CPP:
    print(f"C++ module loaded from: {trappedballcpp.__file__}")
    print(f"Available functions: {[f for f in dir(trappedballcpp) if not f.startswith('_')]}")


def flood_fill_multi(image: np.ndarray, max_iter: int = 20000):
    """Perform multi flood fill operations until all valid areas are filled.
    This operation will fill all rest areas, which may result large amount of fills.

    # Arguments
        image: an image. the image should contain white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    if HAS_CPP:
        return trappedballcpp.flood_fill_multi(image, max_iter)
    else:
        return original_flood_fill_multi(image, max_iter)


def merge_fill(fillmap: np.ndarray, max_iter: int = 10):
    """Merge fill areas.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    if HAS_CPP:
        # C++ version returns list of lists, convert to numpy array
        result = np.array(trappedballcpp.merge_fill(fillmap.astype(np.int32), max_iter), dtype=fillmap.dtype)
        return result
    else:
        return original_merge_fill(fillmap, max_iter)


def build_fill_map(image: np.ndarray, fills: list):
    """Make an image(array) with each pixel(element) marked with fills' id. id of line is 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an array.
    """
    result = np.zeros(image.shape[:2], np.int_)

    for index, fill in enumerate(fills):
        result[fill] = index + 1

    return result


def show_fill_map(fillmap: np.ndarray):
    """Mark filled areas with colors. It is useful for visualization.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    # Generate color for each fill randomly.
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3))
    # Id of line is 0, and its color is black.
    colors[0] = [0, 0, 0]

    return colors[fillmap]


def mark_fill(image, fills):
    mask = np.zeros_like(image, dtype=bool)
    for fill in fills:
        mask[fill] = True
    image[mask] = 0
    return image  # Operates in-place


def trapped_ball_fill_multi(
    image: np.ndarray, radius: int, method="mean", max_iter=1000
):
    """Perform multi trapped ball fill operations until all valid areas are filled.

    # Arguments
        image: an image. The image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        radius: radius of ball shape.
        method: method for filtering the fills.
               'max' is usually with large radius for select large area such as background.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    if HAS_CPP and False:  # Disabled for now, needs proper implementation
        # C++ implementation with optimized trapped ball fill
        unfill_area = image
        filled_area, filled_area_size, result = [], [], []

        for _ in range(max_iter):
            points = get_unfilled_point(exclude_area(unfill_area, radius))

            if not len(points) > 0:
                break

            fill = trappedballcpp.trapped_ball_fill_single(
                unfill_area, (points[0][0], points[0][1]), radius
            )
            unfill_area = cv2.bitwise_and(unfill_area, fill)

            fill_points = trappedballcpp.fast_where(fill, 0)
            filled_area.append(fill_points)
            filled_area_size.append(len(fill_points[0]))

        filled_area_size = np.asarray(filled_area_size)

        if method == "max":
            area_size_filter = np.max(filled_area_size)
        elif method == "median":
            area_size_filter = np.median(filled_area_size)
        elif method == "mean":
            area_size_filter = np.mean(filled_area_size)
        else:
            area_size_filter = 0

        result_idx = np.where(filled_area_size >= area_size_filter)[0]

        for i in result_idx:
            result.append(filled_area[i])

        return result
    else:
        return original_trapped_ball_fill_multi(image, radius, method, max_iter)
