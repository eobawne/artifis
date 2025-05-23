import numpy as np
from typing import Tuple


def apply_border(
    image: np.ndarray, border_type: str, padding: int = 10
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Applies a border to the image based on the specified border_type

    For "b_original", it returns the image unchanged
    For "b_circle", it pads the image to create a larger canvas,
    typically used when the final approximated area is intended to be circular
    or to provide space around the original image content

    :param image: The input image (NumPy array, HxW or HxWxC)
    :type image: np.ndarray
    :param border_type: The type of border to apply ("b_original" or "b_circle")
    :type border_type: str
    :param padding: Padding amount in pixels, used only for "b_circle" border type
    :type padding: int
    :raises ValueError: If an invalid `border_type` is provided
    :return: A tuple containing:
             - The image with the border applied (NumPy array)
             - The original image size (height, width) as a tuple
    :rtype: Tuple[np.ndarray, Tuple[int, int]]
    """
    original_shape = image.shape[:2]  # (height, width)

    if border_type == "b_original":
        # no border applied, return original image and its shape
        return image, original_shape
    elif border_type == "b_circle":
        height, width = original_shape
        # calculate dimensions of new image (with padding on all sides)
        new_height = height + 2 * padding
        new_width = width + 2 * padding

        # create new image with expanded size, filled with black (0)
        if image.ndim == 3:  # rgb or rgba image
            # new image needs same number of channels and dtype
            new_image = np.zeros(
                (new_height, new_width, image.shape[2]), dtype=image.dtype
            )
        else:  # assumes grayscale or other 2d image
            new_image = np.zeros((new_height, new_width), dtype=image.dtype)

        # copy original image into the center of new padded image
        # the slice `padding : padding + height` correctly places the original image
        new_image[padding : padding + height, padding : padding + width] = image

        return new_image, original_shape
    else:
        # handle unknown border types
        raise ValueError(f"Invalid border type: {border_type}")
