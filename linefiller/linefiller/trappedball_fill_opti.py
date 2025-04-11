import time
import cv2
import numpy as np
from numba import njit, prange
from linefiller.log.logger import logger
from linefiller.linefiller.trappedball_fill import (
    trapped_ball_fill_multi as original_trapped_ball_fill_multi,
    flood_fill_multi as original_flood_fill_multi,
)
import sys

sys.path.append("linefiller/linefiller")
import trappedballcpp


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
    return trappedballcpp.flood_fill_multi(image, max_iter)
    # return original_flood_fill_multi(image, max_iter)


def merge_fill(fillmap: np.ndarray, max_iter: int = 10):
    """Merge fill areas.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    return trappedballcpp.merge_fill(fillmap, max_iter)
    # return original_merge_fill(fillmap, max_iter)


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
    # return trappedballcpp.trapped_ball_fill_multi(image, radius, method, max_iter)
    return original_trapped_ball_fill_multi(image, radius, method, max_iter)
