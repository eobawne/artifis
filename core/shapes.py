import numpy as np
from typing import Optional, Tuple
from numba import njit
from numba.core import types
from numba.typed import List as NumbaList

from numba.types import (
    ListType,
    unicode_type,
    uint8,
    int32,
    int64,
    float32,
    boolean,
    void,
    UniTuple,
)


@njit(int64[:](unicode_type, int32))
def get_coord_indices(shape_type: str, num_config_params: int) -> np.ndarray:
    """
    Returns the flat indices corresponding to coordinate parameters for a given shape type

    These indices are relative to the start of the standard parameter block
    (coordinates, color, alpha) which follows any config-defined parameters

    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param num_config_params: The number of parameters defined in the shape's JSON configuration
    :type num_config_params: int
    :return: A NumPy array of int64 indices for the coordinate parameters
    :rtype: np.ndarray
    """
    indices = np.empty(0, dtype=np.int64)
    if shape_type in ("circle", "rectangle", "triangle", "image"):
        # x, y
        indices = np.array(
            [num_config_params + 0, num_config_params + 1], dtype=np.int64
        )
    elif shape_type == "line":
        # x1, y1 (though often referred to as generic x, y in parameter maps)
        indices = np.array(
            [num_config_params + 0, num_config_params + 1], dtype=np.int64
        )
    return indices


@njit(
    types.Tuple((int32, int32, int32, int32, int32, int32, int32))(
        unicode_type, int32, boolean, boolean
    )
)
def _get_param_offsets(
    shape_type: str,
    num_config_params: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Calculates the starting index offsets and counts for different parameter groups within a shape's flat parameter array

    The order is: config-defined params, coordinates, fill color, alpha, stroke color

    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param num_config_params: Number of parameters from the shape's JSON config
    :type num_config_params: int
    :param shape_params_include_rgb: True if fill/stroke colors are RGB, False if grayscale
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if an alpha parameter is included
    :type shape_params_include_alpha: bool
    :return: A tuple containing:
             (coord_offset, num_coord_params, color_offset, num_fill_params,
             alpha_offset, stroke_color_offset, num_stroke_color_params)
             Offsets are -1 if the parameter group does not exist for the shape type
    :rtype: Tuple[int, int, int, int, int, int, int]
    """
    # coordinates always follow config params
    coord_offset = num_config_params
    num_coord_params = 0
    if shape_type in ("circle", "rectangle", "triangle", "image"):
        num_coord_params = 2  # x, y
    elif shape_type == "line":
        num_coord_params = 2  # x1, y1

    current_offset = coord_offset + num_coord_params

    # fill color params
    color_offset = -1
    num_fill_params = 0
    if shape_type not in (
        "image",
        "line",
    ):  # images have inherent color, lines have stroke color
        num_fill_params = 3 if shape_params_include_rgb else 1
        color_offset = current_offset
        current_offset += num_fill_params

    # alpha param
    alpha_offset = -1
    if shape_params_include_alpha:
        alpha_offset = current_offset
        current_offset += 1

    # stroke color params
    stroke_color_offset = -1
    num_stroke_color_params = 0
    if shape_type == "line":  # only lines currently have stroke color
        num_stroke_color_params = 3 if shape_params_include_rgb else 1
        stroke_color_offset = current_offset
        # current_offset += num_stroke_color_params # if more params followed stroke

    return (
        int32(coord_offset),
        int32(num_coord_params),
        int32(color_offset),
        int32(num_fill_params),
        int32(alpha_offset),
        int32(stroke_color_offset),
        int32(num_stroke_color_params),
    )


@njit
def get_total_shape_params(
    shape_type: str,
    param_names: NumbaList[str],
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
) -> int:
    """
    Calculates the total number of parameters per shape based on its type and color/alpha settings

    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param param_names: Numba typed list of names for config-defined parameters
    :type param_names: NumbaList[str]
    :param shape_params_include_rgb: True if shape colors are RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have an alpha parameter
    :type shape_params_include_alpha: bool
    :return: The total number of parameters for one shape of the given type
    :rtype: int
    """
    num_config_params = len(param_names)
    (
        _,  # coord_offset
        num_coord_params,
        _,  # color_offset
        num_fill_params,
        alpha_offset,  # used to determine if alpha param exists
        _,  # stroke_color_offset
        num_stroke_color_params,
    ) = _get_param_offsets(
        shape_type,
        num_config_params,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )

    num_alpha_params = 1 if alpha_offset != -1 else 0

    # sum components based on offsets/counts
    total_params = (
        num_config_params
        + num_coord_params
        + num_fill_params
        + num_alpha_params
        + num_stroke_color_params
    )
    return total_params


@njit(float32(float32[:], int32, float32))
def _safe_get_value(arr: np.ndarray, index: int, default: np.float32) -> np.float32:
    """
    Safely retrieves a value from a NumPy array by index

    Returns a default value if the index is out of bounds

    :param arr: The NumPy array to access
    :type arr: np.ndarray
    :param index: The index to retrieve from the array
    :type index: int
    :param default: The default value to return if index is out of bounds
    :type default: np.float32
    :return: The value at `arr[index]` or `default`
    :rtype: np.float32
    """
    if 0 <= index < len(arr):
        return arr[index]
    return default


# @njit(fastmath=true)
@njit(uint8[:](uint8[:], float32[:], float32), fastmath=True)
def _apply_alpha_rgb(image_pixel: np.ndarray, color_rgb: np.ndarray, alpha: np.float32):
    """
    Applies alpha blending to an RGB pixel

    `new_pixel = (1-alpha)*background + alpha*foreground`

    :param image_pixel: The background RGB pixel (NumPy array of 3 uint8 values)
    :type image_pixel: np.ndarray
    :param color_rgb: The foreground RGB color to blend (NumPy array of 3 float32 values, range [0,255])
    :type color_rgb: np.ndarray
    :param alpha: The alpha value for the foreground color (float32, range [0,1])
    :type alpha: np.float32
    :return: The resulting blended RGB pixel (NumPy array of 3 uint8 values)
    :rtype: np.ndarray
    """
    alpha_clamped = max(np.float32(0.0), min(np.float32(1.0), alpha))  # avoid clip
    one_minus_alpha = np.float32(1.0) - alpha_clamped
    new_pixel = (
        one_minus_alpha * image_pixel.astype(np.float32) + alpha_clamped * color_rgb
    )
    new_pixel_clamped = np.empty_like(new_pixel)
    for k in range(3):
        new_pixel_clamped[k] = max(
            np.float32(0.0), min(np.float32(255.0), new_pixel[k])
        )
    return new_pixel_clamped.astype(np.uint8)


# @njit(fastmath=True)
@njit(uint8(uint8, float32, float32), fastmath=True)
def _apply_alpha_gray(image_pixel: np.uint8, color_gray: np.float32, alpha: np.float32):
    """
    Applies alpha blending to a grayscale pixel

    :param image_pixel: The background grayscale pixel value (uint8)
    :type image_pixel: np.uint8
    :param color_gray: The foreground grayscale color to blend (float32, range [0,255])
    :type color_gray: np.float32
    :param alpha: The alpha value for the foreground color (float32, range [0,1])
    :type alpha: np.float32
    :return: The resulting blended grayscale pixel value (uint8)
    :rtype: np.uint8
    """
    alpha_clamped = max(np.float32(0.0), min(np.float32(1.0), alpha))
    one_minus_alpha = np.float32(1.0) - alpha_clamped
    new_pixel_f = one_minus_alpha * np.float32(image_pixel) + alpha_clamped * color_gray
    clamped_pixel = max(np.float32(0.0), min(np.float32(255.0), new_pixel_f))
    return np.uint8(clamped_pixel)


# @njit(int32(ListType(unicode_type), unicode_type))
@njit
def _find_param_index(param_names: NumbaList[str], name_to_find: str) -> int:
    """
    Finds the index of a parameter name in the list of config-defined parameter names

    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param name_to_find: The parameter name to search for
    :type name_to_find: str
    :return: The index of `name_to_find` in `param_names`, or -1 if not found
    :rtype: int
    """
    # note: this searches the list passed to the function, which should be config params
    for i in range(len(param_names)):
        if param_names[i] == name_to_find:
            return i
    return -1  # return -1 if not found


@njit(
    void(
        uint8[:, :, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _draw_circle_rgb(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a filled circle onto an RGB image canvas

    :param image: The RGB image canvas (NumPy array, HxWxC, uint8) to draw on
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this circle
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names for this shape type
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets from `_get_param_offsets`
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (coord_offset, _, color_offset, num_fill_params, alpha_offset, _, _) = offsets
    default_coord = np.float32(0.0)
    default_color = np.float32(0.5)
    default_alpha = np.float32(1.0)
    default_radius = np.float32(10.0)
    params_len = len(params_values)

    cx_idx, cy_idx = coord_offset + 0, coord_offset + 1
    cx = int32(
        round(params_values[cx_idx] if 0 <= cx_idx < params_len else default_coord)
    )
    cy = int32(
        round(params_values[cy_idx] if 0 <= cy_idx < params_len else default_coord)
    )

    radius_idx = _find_param_index(param_names, "radius")
    r_f = _safe_get_value(params_values, radius_idx, default_radius)
    r = int32(round(r_f))
    if r <= 0:  # if radius is zero or negative, nothing to draw
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    r_color_norm, g_color_norm, b_color_norm = (
        default_color,
        default_color,
        default_color,
    )

    if shape_params_include_rgb and num_fill_params == 3 and color_offset != -1:
        r_idx, g_idx, b_idx = color_offset + 0, color_offset + 1, color_offset + 2
        r_color_norm = (
            params_values[r_idx] if 0 <= r_idx < params_len else default_color
        )
        g_color_norm = (
            params_values[g_idx] if 0 <= g_idx < params_len else default_color
        )
        b_color_norm = (
            params_values[b_idx] if 0 <= b_idx < params_len else default_color
        )
    elif (
        not shape_params_include_rgb and num_fill_params == 1 and color_offset != -1
    ):  # grayscale color mode
        gray_idx = color_offset + 0
        gray_val_norm = (
            params_values[gray_idx] if 0 <= gray_idx < params_len else default_color
        )
        r_color_norm, g_color_norm, b_color_norm = (
            gray_val_norm,
            gray_val_norm,
            gray_val_norm,
        )

    color_to_blend = (
        np.array([r_color_norm, g_color_norm, b_color_norm], dtype=np.float32) * f_255
    )

    min_y, max_y = max(int32(0), cy - r), min(image.shape[0], cy + r + 1)
    min_x, max_x = max(int32(0), cx - r), min(image.shape[1], cx + r + 1)
    r_squared = r_f * r_f

    for y_coord in range(min_y, max_y):
        dy_val = np.float32(y_coord - cy)
        for x_coord in range(min_x, max_x):
            dx_val = np.float32(x_coord - cx)
            if (
                dx_val * dx_val + dy_val * dy_val <= r_squared
            ):  # check if pixel is inside circle
                image[y_coord, x_coord] = _apply_alpha_rgb(
                    image[y_coord, x_coord], color_to_blend, alpha
                )


@njit(
    void(
        uint8[:, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _draw_circle_grayscale(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a filled circle onto a grayscale image canvas

    :param image: The grayscale image canvas (NumPy array, HxW, uint8) to draw on
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this circle
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color parameter structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (coord_offset, _, color_offset, num_fill_params, alpha_offset, _, _) = offsets
    default_coord = np.float32(0.0)
    default_color = np.float32(0.5)
    default_alpha = np.float32(1.0)
    default_radius = np.float32(10.0)
    params_len = len(params_values)

    cx_idx, cy_idx = coord_offset + 0, coord_offset + 1
    cx = int32(
        round(params_values[cx_idx] if 0 <= cx_idx < params_len else default_coord)
    )
    cy = int32(
        round(params_values[cy_idx] if 0 <= cy_idx < params_len else default_coord)
    )

    radius_idx = _find_param_index(param_names, "radius")
    r_f = _safe_get_value(params_values, radius_idx, default_radius)
    r = int32(round(r_f))
    if r <= 0:
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    gray_val_norm = default_color

    if (
        not shape_params_include_rgb and num_fill_params == 1 and color_offset != -1
    ):  # grayscale color mode
        gray_idx = color_offset + 0
        gray_val_norm = (
            params_values[gray_idx] if 0 <= gray_idx < params_len else default_color
        )
    elif (
        shape_params_include_rgb and num_fill_params == 3 and color_offset != -1
    ):  # rgb color mode, convert to gray
        r_idx, g_idx, b_idx = color_offset + 0, color_offset + 1, color_offset + 2
        r_val_norm = params_values[r_idx] if 0 <= r_idx < params_len else default_color
        g_val_norm = params_values[g_idx] if 0 <= g_idx < params_len else default_color
        b_val_norm = params_values[b_idx] if 0 <= b_idx < params_len else default_color
        # luminosity conversion for grayscale
        gray_val_norm = (
            np.float32(0.299) * r_val_norm
            + np.float32(0.587) * g_val_norm
            + np.float32(0.114) * b_val_norm
        )

    color_to_blend = gray_val_norm * f_255

    min_y, max_y = max(int32(0), cy - r), min(image.shape[0], cy + r + 1)
    min_x, max_x = max(int32(0), cx - r), min(image.shape[1], cx + r + 1)
    r_squared = r_f * r_f

    for y_coord in range(min_y, max_y):
        dy_val = np.float32(y_coord - cy)
        for x_coord in range(min_x, max_x):
            dx_val = np.float32(x_coord - cx)
            if dx_val * dx_val + dy_val * dy_val <= r_squared:
                image[y_coord, x_coord] = _apply_alpha_gray(
                    image[y_coord, x_coord], color_to_blend, alpha
                )


@njit(
    void(
        uint8[:, :, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _draw_rectangle_rgb(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a filled rectangle onto an RGB image canvas

    :param image: The RGB image canvas (NumPy array, HxWxC, uint8)
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this rectangle
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (coord_offset, _, color_offset, num_fill_params, alpha_offset, _, _) = offsets
    default_coord = np.float32(0.0)
    default_size = np.float32(10.0)
    default_color = np.float32(0.5)
    default_alpha = np.float32(1.0)
    params_len = len(params_values)

    x0_idx, y0_idx = coord_offset + 0, coord_offset + 1  # top-left corner
    x0 = int32(
        round(params_values[x0_idx] if 0 <= x0_idx < params_len else default_coord)
    )
    y0 = int32(
        round(params_values[y0_idx] if 0 <= y0_idx < params_len else default_coord)
    )

    width_idx = _find_param_index(param_names, "width")
    height_idx = _find_param_index(param_names, "height")
    rect_width = int32(round(_safe_get_value(params_values, width_idx, default_size)))
    rect_height = int32(round(_safe_get_value(params_values, height_idx, default_size)))

    if rect_width <= 0 or rect_height <= 0:  # if no area, nothing to draw
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    r_color_norm, g_color_norm, b_color_norm = (
        default_color,
        default_color,
        default_color,
    )

    if shape_params_include_rgb and num_fill_params == 3 and color_offset != -1:
        r_idx, g_idx, b_idx = color_offset + 0, color_offset + 1, color_offset + 2
        r_color_norm = (
            params_values[r_idx] if 0 <= r_idx < params_len else default_color
        )
        g_color_norm = (
            params_values[g_idx] if 0 <= g_idx < params_len else default_color
        )
        b_color_norm = (
            params_values[b_idx] if 0 <= b_idx < params_len else default_color
        )
    elif not shape_params_include_rgb and num_fill_params == 1 and color_offset != -1:
        gray_idx = color_offset + 0
        gray_val_norm = (
            params_values[gray_idx] if 0 <= gray_idx < params_len else default_color
        )
        r_color_norm, g_color_norm, b_color_norm = (
            gray_val_norm,
            gray_val_norm,
            gray_val_norm,
        )

    color_to_blend = (
        np.array([r_color_norm, g_color_norm, b_color_norm], dtype=np.float32) * f_255
    )

    # define drawing bounds, clamped to image dimensions
    min_y, max_y = max(int32(0), y0), min(image.shape[0], y0 + rect_height)
    min_x, max_x = max(int32(0), x0), min(image.shape[1], x0 + rect_width)

    for y_coord in range(min_y, max_y):
        for x_coord in range(min_x, max_x):
            image[y_coord, x_coord] = _apply_alpha_rgb(
                image[y_coord, x_coord], color_to_blend, alpha
            )


@njit(
    void(
        uint8[:, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _draw_rectangle_grayscale(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a filled rectangle onto a grayscale image canvas

    :param image: The grayscale image canvas (NumPy array, HxW, uint8)
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this rectangle
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color parameter structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (coord_offset, _, color_offset, num_fill_params, alpha_offset, _, _) = offsets
    default_coord = np.float32(0.0)
    default_size = np.float32(10.0)
    default_color = np.float32(0.5)
    default_alpha = np.float32(1.0)
    params_len = len(params_values)

    x0_idx, y0_idx = coord_offset + 0, coord_offset + 1
    x0 = int32(
        round(params_values[x0_idx] if 0 <= x0_idx < params_len else default_coord)
    )
    y0 = int32(
        round(params_values[y0_idx] if 0 <= y0_idx < params_len else default_coord)
    )

    width_idx = _find_param_index(param_names, "width")
    height_idx = _find_param_index(param_names, "height")
    rect_width = int32(round(_safe_get_value(params_values, width_idx, default_size)))
    rect_height = int32(round(_safe_get_value(params_values, height_idx, default_size)))

    if rect_width <= 0 or rect_height <= 0:
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    gray_val_norm = default_color

    if not shape_params_include_rgb and num_fill_params == 1 and color_offset != -1:
        gray_idx = color_offset + 0
        gray_val_norm = (
            params_values[gray_idx] if 0 <= gray_idx < params_len else default_color
        )
    elif shape_params_include_rgb and num_fill_params == 3 and color_offset != -1:
        r_idx, g_idx, b_idx = color_offset + 0, color_offset + 1, color_offset + 2
        r_val_norm = params_values[r_idx] if 0 <= r_idx < params_len else default_color
        g_val_norm = params_values[g_idx] if 0 <= g_idx < params_len else default_color
        b_val_norm = params_values[b_idx] if 0 <= b_idx < params_len else default_color
        gray_val_norm = (
            np.float32(0.299) * r_val_norm
            + np.float32(0.587) * g_val_norm
            + np.float32(0.114) * b_val_norm
        )

    color_to_blend = gray_val_norm * f_255

    min_y, max_y = max(int32(0), y0), min(image.shape[0], y0 + rect_height)
    min_x, max_x = max(int32(0), x0), min(image.shape[1], x0 + rect_width)

    for y_coord in range(min_y, max_y):
        for x_coord in range(min_x, max_x):
            image[y_coord, x_coord] = _apply_alpha_gray(
                image[y_coord, x_coord], color_to_blend, alpha
            )


@njit(UniTuple(float32, 6)(float32[:], ListType(unicode_type), int32), fastmath=True)
def _calculate_triangle_vertices_from_array(
    params_values: np.ndarray, param_names: NumbaList[str], coord_offset: int
) -> Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]:
    """
    Calculates the three vertices (x1,y1, x2,y2, x3,y3) of a triangle

    Uses side lengths (a,b,c) and a base coordinate (x_base, y_base)
    Side 'c' is placed along the x-axis starting from (x_base, y_base)
    The third vertex is calculated using the law of cosines
    Includes triangle inequality adjustments to ensure valid triangles

    :param params_values: 1D NumPy array of shape parameters
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param coord_offset: Starting index of coordinate parameters in `params_values`
    :type coord_offset: int
    :return: A tuple of six float32 values: (x1, y1, x2, y2, x3, y3)
    :rtype: Tuple[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
    """
    default_side = np.float32(10.0)
    default_coord = np.float32(0.0)
    params_len = len(params_values)

    a_idx = _find_param_index(param_names, "side_a")
    b_idx = _find_param_index(param_names, "side_b")
    c_idx = _find_param_index(param_names, "side_c")
    a = _safe_get_value(params_values, a_idx, default_side)
    b = _safe_get_value(params_values, b_idx, default_side)
    c = _safe_get_value(params_values, c_idx, default_side)

    x_base_idx, y_base_idx = coord_offset + 0, coord_offset + 1
    x_base = (
        params_values[x_base_idx] if 0 <= x_base_idx < params_len else default_coord
    )
    y_base = (
        params_values[y_base_idx] if 0 <= y_base_idx < params_len else default_coord
    )

    # ensure sides are positive and attempt to satisfy triangle inequality
    epsilon = np.float32(1e-5)  # small tolerance for float comparisons
    min_side_length = np.float32(
        1e-4
    )  # minimum allowed side length to avoid degeneracy
    a = max(min_side_length, a)
    b = max(min_side_length, b)
    c = max(min_side_length, c)

    # adjust sides to satisfy triangle inequality (a+b > c, etc)
    # if a+b <= c, it means c is too long, so shorten c
    if a + b <= c + epsilon:
        c = max(min_side_length, a + b - epsilon)
    if a + c <= b + epsilon:
        b = max(min_side_length, a + c - epsilon)
    if b + c <= a + epsilon:
        a = max(min_side_length, b + c - epsilon)

    # define vertices
    x1, y1 = x_base, y_base  # first vertex at base coordinate
    x2, y2 = x_base + c, y_base  # second vertex along x-axis, distance c

    # calculate third vertex using cosine rule for angle A (opposite side a)
    cos_A_num = b * b + c * c - a * a
    cos_A_den = np.float32(2.0) * b * c

    if abs(cos_A_den) < 1e-9:  # if b or c is effectively zero, degenerate triangle
        # return a small, almost collinear triangle to avoid issues
        return x_base, y_base, x_base + c, y_base, x_base, y_base + min_side_length

    cos_A = max(
        np.float32(-1.0), min(np.float32(1.0), cos_A_num / cos_A_den)
    )  # clamp for arccos
    angle_A = np.arccos(cos_A)

    # third vertex is (b*cosA, b*sinA) relative to (x1,y1)
    x3 = x_base + b * np.cos(angle_A)
    y3 = y_base + b * np.sin(angle_A)  # sinA gives positive y for angles 0-pi

    return x1, y1, x2, y2, x3, y3


@njit(float32(float32, float32, float32, float32), fastmath=True)
def _cross_product_z(x1, y1, x2, y2):
    """
    Calculates the Z-component of the 2D cross product (v1 x v2)

    Used in point-in-triangle test (Barycentric coordinates)

    :param x1: x-component of the first vector
    :type x1: float32
    :param y1: y-component of the first vector
    :type y1: float32
    :param x2: x-component of the second vector
    :type x2: float32
    :param y2: y-component of the second vector
    :type y2: float32
    :return: The Z-component of the cross product
    :rtype: float32
    """
    return x1 * y2 - y1 * x2


@njit(
    void(
        uint8[:, :, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _draw_triangle_rgb(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a filled triangle onto an RGB image canvas using Barycentric coordinates test

    :param image: The RGB image canvas (NumPy array, HxWxC, uint8)
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this triangle
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (coord_offset, _, color_offset, num_fill_params, alpha_offset, _, _) = offsets
    default_color = np.float32(0.5)
    default_alpha = np.float32(1.0)
    params_len = len(params_values)

    x1f, y1f, x2f, y2f, x3f, y3f = _calculate_triangle_vertices_from_array(
        params_values, param_names, coord_offset
    )

    # check for degenerate triangle (collinear vertices) by checking area (using cross product)
    # if area is near zero, don't draw
    if abs((x2f - x1f) * (y3f - y1f) - (x3f - x1f) * (y2f - y1f)) < 1e-6:
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    r_color_norm, g_color_norm, b_color_norm = (
        default_color,
        default_color,
        default_color,
    )

    if shape_params_include_rgb and num_fill_params == 3 and color_offset != -1:
        r_idx, g_idx, b_idx = color_offset + 0, color_offset + 1, color_offset + 2
        r_color_norm = (
            params_values[r_idx] if 0 <= r_idx < params_len else default_color
        )
        g_color_norm = (
            params_values[g_idx] if 0 <= g_idx < params_len else default_color
        )
        b_color_norm = (
            params_values[b_idx] if 0 <= b_idx < params_len else default_color
        )
    elif not shape_params_include_rgb and num_fill_params == 1 and color_offset != -1:
        gray_idx = color_offset + 0
        gray_val_norm = (
            params_values[gray_idx] if 0 <= gray_idx < params_len else default_color
        )
        r_color_norm, g_color_norm, b_color_norm = (
            gray_val_norm,
            gray_val_norm,
            gray_val_norm,
        )

    color_to_blend = (
        np.array([r_color_norm, g_color_norm, b_color_norm], dtype=np.float32) * f_255
    )

    # convert float vertices to int for bounding box
    x1, y1 = int32(round(x1f)), int32(round(y1f))
    x2, y2 = int32(round(x2f)), int32(round(y2f))
    x3, y3 = int32(round(x3f)), int32(round(y3f))

    # determine bounding box of the triangle, clamped to image dimensions
    min_x_bb = max(int32(0), min(x1, x2, x3))
    max_x_bb = min(
        image.shape[1], max(x1, x2, x3) + 1
    )  # +1 for exclusive upper bound in range
    min_y_bb = max(int32(0), min(y1, y2, y3))
    max_y_bb = min(image.shape[0], max(y1, y2, y3) + 1)

    # vectors for edge functions (Barycentric coordinate test)
    # these are edges of the triangle: v1 = P2-P1, v2 = P3-P2, v3 = P1-P3
    v1x_edge, v1y_edge = x2f - x1f, y2f - y1f
    v2x_edge, v2y_edge = x3f - x2f, y3f - y2f
    v3x_edge, v3y_edge = x1f - x3f, y1f - y3f

    tolerance_bary = -1e-5  # small tolerance for edge cases with Barycentric test

    for y_coord_px in range(min_y_bb, max_y_bb):
        for x_coord_px in range(min_x_bb, max_x_bb):
            # use pixel center for more accurate rasterization
            px_center, py_center = (
                np.float32(x_coord_px) + 0.5,
                np.float32(y_coord_px) + 0.5,
            )

            # calculate Barycentric coordinates (or related signed areas)
            # w1 is related to area of triangle P-P1-P2 relative to P1-P2-P3
            w1_bary = _cross_product_z(
                v1x_edge, v1y_edge, px_center - x1f, py_center - y1f
            )
            w2_bary = _cross_product_z(
                v2x_edge, v2y_edge, px_center - x2f, py_center - y2f
            )
            w3_bary = _cross_product_z(
                v3x_edge, v3y_edge, px_center - x3f, py_center - y3f
            )

            # point is inside if all w_i have same sign (or are zero)
            # using tolerance helps with floating point inaccuracies at edges
            if (
                w1_bary >= tolerance_bary
                and w2_bary >= tolerance_bary
                and w3_bary >= tolerance_bary
            ) or (
                w1_bary <= -tolerance_bary
                and w2_bary <= -tolerance_bary
                and w3_bary <= -tolerance_bary
            ):
                image[y_coord_px, x_coord_px] = _apply_alpha_rgb(
                    image[y_coord_px, x_coord_px], color_to_blend, alpha
                )


@njit(
    void(
        uint8[:, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _draw_triangle_grayscale(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a filled triangle onto a grayscale image canvas

    :param image: The grayscale image canvas (NumPy array, HxW, uint8)
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this triangle
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color parameter structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (coord_offset, _, color_offset, num_fill_params, alpha_offset, _, _) = offsets
    default_color = np.float32(0.5)
    default_alpha = np.float32(1.0)
    params_len = len(params_values)

    x1f, y1f, x2f, y2f, x3f, y3f = _calculate_triangle_vertices_from_array(
        params_values, param_names, coord_offset
    )
    if abs((x2f - x1f) * (y3f - y1f) - (x3f - x1f) * (y2f - y1f)) < 1e-6:
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    gray_val_norm = default_color

    if not shape_params_include_rgb and num_fill_params == 1 and color_offset != -1:
        gray_idx = color_offset + 0
        gray_val_norm = (
            params_values[gray_idx] if 0 <= gray_idx < params_len else default_color
        )
    elif shape_params_include_rgb and num_fill_params == 3 and color_offset != -1:
        r_idx, g_idx, b_idx = color_offset + 0, color_offset + 1, color_offset + 2
        r_val_norm = params_values[r_idx] if 0 <= r_idx < params_len else default_color
        g_val_norm = params_values[g_idx] if 0 <= g_idx < params_len else default_color
        b_val_norm = params_values[b_idx] if 0 <= b_idx < params_len else default_color
        gray_val_norm = (
            np.float32(0.299) * r_val_norm
            + np.float32(0.587) * g_val_norm
            + np.float32(0.114) * b_val_norm
        )

    color_to_blend = gray_val_norm * f_255

    x1, y1 = int32(round(x1f)), int32(round(y1f))
    x2, y2 = int32(round(x2f)), int32(round(y2f))
    x3, y3 = int32(round(x3f)), int32(round(y3f))

    min_x_bb = max(int32(0), min(x1, x2, x3))
    max_x_bb = min(image.shape[1], max(x1, x2, x3) + 1)
    min_y_bb = max(int32(0), min(y1, y2, y3))
    max_y_bb = min(image.shape[0], max(y1, y2, y3) + 1)

    v1x_edge, v1y_edge = x2f - x1f, y2f - y1f
    v2x_edge, v2y_edge = x3f - x2f, y3f - y2f
    v3x_edge, v3y_edge = x1f - x3f, y1f - y3f
    tolerance_bary = -1e-5

    for y_coord_px in range(min_y_bb, max_y_bb):
        for x_coord_px in range(min_x_bb, max_x_bb):
            px_center, py_center = (
                np.float32(x_coord_px) + 0.5,
                np.float32(y_coord_px) + 0.5,
            )
            w1_bary = _cross_product_z(
                v1x_edge, v1y_edge, px_center - x1f, py_center - y1f
            )
            w2_bary = _cross_product_z(
                v2x_edge, v2y_edge, px_center - x2f, py_center - y2f
            )
            w3_bary = _cross_product_z(
                v3x_edge, v3y_edge, px_center - x3f, py_center - y3f
            )
            if (
                w1_bary >= tolerance_bary
                and w2_bary >= tolerance_bary
                and w3_bary >= tolerance_bary
            ) or (
                w1_bary <= -tolerance_bary
                and w2_bary <= -tolerance_bary
                and w3_bary <= -tolerance_bary
            ):
                image[y_coord_px, x_coord_px] = _apply_alpha_gray(
                    image[y_coord_px, x_coord_px], color_to_blend, alpha
                )


@njit(
    void(
        uint8[:, :, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def draw_line_rgb(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a line onto an RGB image canvas using (x1,y1), length, angle, and stroke_width

    Uses Bresenham's line algorithm for the centerline and a simple square brush
    for stroke width

    :param image: The RGB image canvas (NumPy array, HxWxC, uint8)
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this line
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape (stroke) color is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (
        coord_offset,
        _,
        _,
        _,
        alpha_offset,
        stroke_color_offset,
        num_stroke_color_params,
    ) = offsets
    params_len = len(params_values)
    default_coord = np.float32(0.0)
    default_length = np.float32(20.0)
    default_angle = np.float32(0.0)
    default_stroke_color = np.float32(0.0)  # black
    default_alpha = np.float32(1.0)
    default_stroke_width = np.float32(1.0)

    x1_idx, y1_idx = coord_offset + 0, coord_offset + 1
    x1f = params_values[x1_idx] if 0 <= x1_idx < params_len else default_coord
    y1f = params_values[y1_idx] if 0 <= y1_idx < params_len else default_coord

    length_idx = _find_param_index(param_names, "length")
    angle_idx = _find_param_index(param_names, "angle")
    sw_idx = _find_param_index(param_names, "stroke_width")
    length = _safe_get_value(params_values, length_idx, default_length)
    angle_deg = _safe_get_value(params_values, angle_idx, default_angle)
    stroke_width_f = _safe_get_value(params_values, sw_idx, default_stroke_width)
    stroke_width = max(
        int32(1), int32(round(stroke_width_f))
    )  # ensure at least 1px width

    if length <= 0:
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)

    # calculate endpoint (x2, y2) from (x1,y1), length, and angle
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x2f = x1f + length * cos_a
    y2f = y1f + length * sin_a

    r_color_norm, g_color_norm, b_color_norm = (
        default_stroke_color,
        default_stroke_color,
        default_stroke_color,
    )
    if (
        shape_params_include_rgb
        and num_stroke_color_params == 3
        and stroke_color_offset != -1
    ):
        r_idx, g_idx, b_idx = (
            stroke_color_offset + 0,
            stroke_color_offset + 1,
            stroke_color_offset + 2,
        )
        r_color_norm = (
            params_values[r_idx] if 0 <= r_idx < params_len else default_stroke_color
        )
        g_color_norm = (
            params_values[g_idx] if 0 <= g_idx < params_len else default_stroke_color
        )
        b_color_norm = (
            params_values[b_idx] if 0 <= b_idx < params_len else default_stroke_color
        )
    elif (
        not shape_params_include_rgb
        and num_stroke_color_params == 1
        and stroke_color_offset != -1
    ):
        gray_idx = stroke_color_offset + 0
        gray_val_norm = (
            params_values[gray_idx]
            if 0 <= gray_idx < params_len
            else default_stroke_color
        )
        r_color_norm, g_color_norm, b_color_norm = (
            gray_val_norm,
            gray_val_norm,
            gray_val_norm,
        )

    color_to_blend = (
        np.array([r_color_norm, g_color_norm, b_color_norm], dtype=np.float32) * f_255
    )

    # bresenham's line algorithm
    x1_bres, y1_bres = int32(round(x1f)), int32(round(y1f))
    x2_bres, y2_bres = int32(round(x2f)), int32(round(y2f))
    dx_bres, dy_bres = abs(x2_bres - x1_bres), abs(y2_bres - y1_bres)
    sx_bres = int32(1) if x1_bres < x2_bres else int32(-1)
    sy_bres = int32(1) if y1_bres < y2_bres else int32(-1)
    err_bres = dx_bres - dy_bres
    radius_stroke = stroke_width // 2  # for square brush centered on line pixel
    img_h_val, img_w_val = image.shape[0], image.shape[1]

    while True:
        # draw a square "brush" of stroke_width around current line pixel (x1_bres, y1_bres)
        min_py_stroke = max(0, y1_bres - radius_stroke)
        max_py_stroke = min(img_h_val, y1_bres + radius_stroke + 1)
        min_px_stroke = max(0, x1_bres - radius_stroke)
        max_px_stroke = min(img_w_val, x1_bres + radius_stroke + 1)
        for py_stroke in range(min_py_stroke, max_py_stroke):
            for px_stroke in range(min_px_stroke, max_px_stroke):
                image[py_stroke, px_stroke] = _apply_alpha_rgb(
                    image[py_stroke, px_stroke], color_to_blend, alpha
                )

        if x1_bres == x2_bres and y1_bres == y2_bres:  # if endpoint reached
            break
        e2_bres = 2 * err_bres
        if e2_bres > -dy_bres:  # slope < 1
            err_bres -= dy_bres
            x1_bres += sx_bres
        if e2_bres < dx_bres:  # slope > 1
            err_bres += dx_bres
            y1_bres += sy_bres


@njit(
    void(
        uint8[:, :],
        float32[:],
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def draw_line_grayscale(
    image: np.ndarray,
    params_values: np.ndarray,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Draws a line onto a grayscale image canvas

    Similar to `draw_line_rgb` but for grayscale images

    :param image: The grayscale image canvas (NumPy array, HxW, uint8)
    :type image: np.ndarray
    :param params_values: 1D NumPy array of parameters for this line
    :type params_values: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape (stroke) color parameter structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shape has alpha
    :type shape_params_include_alpha: bool
    """
    (
        coord_offset,
        _,
        _,
        _,
        alpha_offset,
        stroke_color_offset,
        num_stroke_color_params,
    ) = offsets
    params_len = len(params_values)
    default_coord = np.float32(0.0)
    default_length = np.float32(20.0)
    default_angle = np.float32(0.0)
    default_stroke_color = np.float32(0.0)
    default_alpha = np.float32(1.0)
    default_stroke_width = np.float32(1.0)

    x1_idx, y1_idx = coord_offset + 0, coord_offset + 1
    x1f = params_values[x1_idx] if 0 <= x1_idx < params_len else default_coord
    y1f = params_values[y1_idx] if 0 <= y1_idx < params_len else default_coord

    length_idx = _find_param_index(param_names, "length")
    angle_idx = _find_param_index(param_names, "angle")
    sw_idx = _find_param_index(param_names, "stroke_width")
    length = _safe_get_value(params_values, length_idx, default_length)
    angle_deg = _safe_get_value(params_values, angle_idx, default_angle)
    stroke_width_f = _safe_get_value(params_values, sw_idx, default_stroke_width)
    stroke_width = max(int32(1), int32(round(stroke_width_f)))

    if length <= 0:
        return

    alpha = (
        params_values[alpha_offset]
        if alpha_offset != -1 and 0 <= alpha_offset < params_len
        else default_alpha
    )
    f_255 = np.float32(255.0)
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x2f = x1f + length * cos_a
    y2f = y1f + length * sin_a

    gray_val_norm = default_stroke_color
    if (
        not shape_params_include_rgb
        and num_stroke_color_params == 1
        and stroke_color_offset != -1
    ):
        gray_idx = stroke_color_offset + 0
        gray_val_norm = (
            params_values[gray_idx]
            if 0 <= gray_idx < params_len
            else default_stroke_color
        )
    elif (
        shape_params_include_rgb
        and num_stroke_color_params == 3
        and stroke_color_offset != -1
    ):
        r_idx, g_idx, b_idx = (
            stroke_color_offset + 0,
            stroke_color_offset + 1,
            stroke_color_offset + 2,
        )
        r_val_norm = (
            params_values[r_idx] if 0 <= r_idx < params_len else default_stroke_color
        )
        g_val_norm = (
            params_values[g_idx] if 0 <= g_idx < params_len else default_stroke_color
        )
        b_val_norm = (
            params_values[b_idx] if 0 <= b_idx < params_len else default_stroke_color
        )
        gray_val_norm = (
            np.float32(0.299) * r_val_norm
            + np.float32(0.587) * g_val_norm
            + np.float32(0.114) * b_val_norm
        )

    color_to_blend = gray_val_norm * f_255

    x1_bres, y1_bres = int32(round(x1f)), int32(round(y1f))
    x2_bres, y2_bres = int32(round(x2f)), int32(round(y2f))
    dx_bres, dy_bres = abs(x2_bres - x1_bres), abs(y2_bres - y1_bres)
    sx_bres = int32(1) if x1_bres < x2_bres else int32(-1)
    sy_bres = int32(1) if y1_bres < y2_bres else int32(-1)
    err_bres = dx_bres - dy_bres
    radius_stroke = stroke_width // 2
    img_h_val, img_w_val = image.shape[0], image.shape[1]

    while True:
        min_py_stroke = max(0, y1_bres - radius_stroke)
        max_py_stroke = min(img_h_val, y1_bres + radius_stroke + 1)
        min_px_stroke = max(0, x1_bres - radius_stroke)
        max_px_stroke = min(img_w_val, x1_bres + radius_stroke + 1)
        for py_stroke in range(min_py_stroke, max_py_stroke):
            for px_stroke in range(min_px_stroke, max_px_stroke):
                image[py_stroke, px_stroke] = _apply_alpha_gray(
                    image[py_stroke, px_stroke], color_to_blend, alpha
                )
        if x1_bres == x2_bres and y1_bres == y2_bres:
            break
        e2_bres = 2 * err_bres
        if e2_bres > -dy_bres:
            err_bres -= dy_bres
            x1_bres += sx_bres
        if e2_bres < dx_bres:
            err_bres += dx_bres
            y1_bres += sy_bres


@njit(
    void(
        uint8[:, :],
        float32[:, :],
        unicode_type,
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _render_shapes_grayscale(
    image: np.ndarray,
    shapes_to_draw: np.ndarray,
    shape_type: str,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Renders a collection of shapes onto a preexisting grayscale image canvas

    Dispatches to the appropriate `_draw_shape_grayscale` function based on `shape_type`

    :param image: The 2D grayscale image canvas (NumPy array, HxW, uint8) to draw on
    :type image: np.ndarray
    :param shapes_to_draw: 2D NumPy array (num_shapes x num_params) of shape parameters
    :type shapes_to_draw: np.ndarray
    :param shape_type: String identifier for the shape type to draw
    :type shape_type: str
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color parameter structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have alpha
    :type shape_params_include_alpha: bool
    """
    num_shapes = shapes_to_draw.shape[0]
    for i in range(num_shapes):
        shape_params = shapes_to_draw[i]  # parameters for current shape
        if shape_type == "circle":
            _draw_circle_grayscale(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        elif shape_type == "rectangle":
            _draw_rectangle_grayscale(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        elif shape_type == "triangle":
            _draw_triangle_grayscale(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        elif shape_type == "line":
            draw_line_grayscale(  # line has different naming convention currently
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )


@njit(
    void(
        uint8[:, :, :],
        float32[:, :],
        unicode_type,
        ListType(unicode_type),
        UniTuple(int32, 7),
        boolean,
        boolean,
    ),
    fastmath=True,
)
def _render_shapes_rgb(
    image: np.ndarray,
    shapes_to_draw: np.ndarray,
    shape_type: str,
    param_names: NumbaList[str],
    offsets: Tuple,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
):
    """
    Renders a collection of shapes onto a preexisting RGB image canvas

    Dispatches to the appropriate `_draw_shape_rgb` function based on `shape_type`

    :param image: The 3D RGB image canvas (NumPy array, HxWxC, uint8) to draw on
    :type image: np.ndarray
    :param shapes_to_draw: 2D NumPy array (num_shapes x num_params) of shape parameters
    :type shapes_to_draw: np.ndarray
    :param shape_type: String identifier for the shape type to draw
    :type shape_type: str
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param offsets: Tuple of parameter group offsets
    :type offsets: Tuple
    :param shape_params_include_rgb: True if shape color is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have alpha
    :type shape_params_include_alpha: bool
    """
    num_shapes = shapes_to_draw.shape[0]
    for i in range(num_shapes):
        shape_params = shapes_to_draw[i]
        if shape_type == "circle":
            _draw_circle_rgb(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        elif shape_type == "rectangle":
            _draw_rectangle_rgb(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        elif shape_type == "triangle":
            _draw_triangle_rgb(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        elif shape_type == "line":
            draw_line_rgb(
                image,
                shape_params,
                param_names,
                offsets,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )


@njit(fastmath=True)
def initialize_shapes(
    num_shapes: int,
    shape_type: str,
    image_shape: Tuple[int, int],
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    coord_init_type: str,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    param_init_type: str,
    initial_coords: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Initializes parameters for a given number of shapes

    Handles different initialization strategies for coordinates (random, grid, zero, PDF-based)
    and for other configurable parameters (midpoint, min, max, random)

    :param num_shapes: Number of shapes to initialize
    :type num_shapes: int
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width) of the image canvas
    :type image_shape: Tuple[int, int]
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: NumPy array of min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: NumPy array of max bounds for config parameters
    :type param_maxs: np.ndarray
    :param coord_init_type: Strategy for initializing coordinates ("random", "grid", etc)
    :type coord_init_type: str
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param param_init_type: Strategy for initializing non-coordinate config parameters ("midpoint", "random", etc)
    :type param_init_type: str
    :param initial_coords: Optional NumPy array (num_shapes x 2) of presampled (x,y) coordinates
                           Used if `coord_init_type` was PDF-based and sampling succeeded
    :type initial_coords: Optional[np.ndarray]
    :return: A 2D NumPy array (num_shapes x num_total_params) of initialized shape parameters
    :rtype: np.ndarray
    """
    height, width = image_shape
    num_config_params = len(param_names)
    total_params_per_shape = get_total_shape_params(
        shape_type, param_names, shape_params_include_rgb, shape_params_include_alpha
    )
    shapes_arr = np.zeros((num_shapes, total_params_per_shape), dtype=np.float32)
    f_height, f_width = np.float32(height), np.float32(width)

    # grid initialization setup (if coord_init_type is "grid")
    grid_cols_val = int32(0)
    grid_rows_val = int32(0)
    cell_width_val = np.float32(0.0)
    cell_height_val = np.float32(0.0)
    if (
        coord_init_type == "grid" and initial_coords is None
    ):  # only if not using presampled coords
        f_num_shapes_val = np.float32(num_shapes)
        grid_cols_f_val = np.ceil(np.sqrt(f_num_shapes_val))
        grid_cols_val = int32(grid_cols_f_val) if grid_cols_f_val > 0 else int32(1)
        grid_rows_val = (
            int32(np.ceil(f_num_shapes_val / max(1.0, grid_cols_f_val)))
            if grid_cols_val > 0
            else int32(0)
        )
        cell_width_val = (
            f_width / np.float32(grid_cols_val) if grid_cols_val > 0 else f_width
        )
        cell_height_val = (
            f_height / np.float32(grid_rows_val) if grid_rows_val > 0 else f_height
        )

    (
        coord_offset,
        num_coord_params,
        color_offset,
        num_fill_params,
        alpha_offset,
        stroke_color_offset,
        num_stroke_color_params,
    ) = _get_param_offsets(
        shape_type,
        num_config_params,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    default_alpha_val = np.float32(0.5)  # default alpha to semi-transparent

    # initialize each shape
    for i_shape in range(num_shapes):
        # 1. config-defined parameters (e.g., radius, width, angle)
        for k_param_config in range(num_config_params):
            value_to_assign_config = np.float32(0.0)
            if k_param_config < len(param_mins) and k_param_config < len(param_maxs):
                min_val_cfg = param_mins[k_param_config]
                max_val_cfg = param_maxs[k_param_config]
                f_min_cfg = (
                    np.float32(min_val_cfg)
                    if np.isfinite(min_val_cfg)
                    else np.float32(0.0)
                )
                f_max_cfg = (
                    np.float32(max_val_cfg)
                    if np.isfinite(max_val_cfg)
                    else np.float32(1.0)
                )
                if f_min_cfg > f_max_cfg:
                    f_min_cfg, f_max_cfg = f_max_cfg, f_min_cfg  # ensure min <= max

                if param_init_type == "random":
                    value_to_assign_config = np.random.uniform(f_min_cfg, f_max_cfg)
                elif param_init_type == "min":
                    value_to_assign_config = f_min_cfg
                elif param_init_type == "max":
                    value_to_assign_config = f_max_cfg
                else:  # midpoint or fallback
                    value_to_assign_config = (f_min_cfg + f_max_cfg) / np.float32(2.0)
            else:  # fallback if bounds missing for a config param
                value_to_assign_config = (
                    np.float32(0.5)
                    if param_init_type != "random"
                    else np.random.uniform(np.float32(0.0), np.float32(1.0))
                )
            shapes_arr[i_shape, k_param_config] = value_to_assign_config

        # 2. coordinate parameters
        if (
            coord_offset >= 0
            and coord_offset + num_coord_params <= total_params_per_shape
        ):
            x_idx_coord = coord_offset
            y_idx_coord = coord_offset + 1  # assumes at least 2 coord params if any

            if (
                initial_coords is not None and i_shape < initial_coords.shape[0]
            ):  # use presampled coords
                if num_coord_params >= 1 and initial_coords.shape[1] > 0:
                    shapes_arr[i_shape, x_idx_coord] = initial_coords[i_shape, 0]
                if num_coord_params >= 2 and initial_coords.shape[1] > 1:
                    shapes_arr[i_shape, y_idx_coord] = initial_coords[i_shape, 1]
            elif coord_init_type == "random":
                if num_coord_params >= 1:
                    shapes_arr[i_shape, x_idx_coord] = np.random.uniform(
                        np.float32(0.0), f_width
                    )
                if num_coord_params >= 2:
                    shapes_arr[i_shape, y_idx_coord] = np.random.uniform(
                        np.float32(0.0), f_height
                    )
            elif coord_init_type == "grid":
                center_x_grid, center_y_grid = np.float32(0.0), np.float32(0.0)
                if grid_cols_val > 0 and grid_rows_val > 0:
                    row_grid = i_shape // grid_cols_val
                    col_grid = i_shape % grid_cols_val
                    center_x_grid = (
                        np.float32(col_grid) + np.float32(0.5)
                    ) * cell_width_val
                    center_y_grid = (
                        np.float32(row_grid) + np.float32(0.5)
                    ) * cell_height_val

                # for shapes like rectangle, grid places top-left corner, not center
                if shape_type == "rectangle":
                    width_idx_rect = _find_param_index(param_names, "width")
                    height_idx_rect = _find_param_index(param_names, "height")
                    rect_w_f = _safe_get_value(
                        shapes_arr[i_shape], width_idx_rect, np.float32(10.0)
                    )
                    rect_h_f = _safe_get_value(
                        shapes_arr[i_shape], height_idx_rect, np.float32(10.0)
                    )
                    x_origin_rect = center_x_grid - (rect_w_f / np.float32(2.0))
                    y_origin_rect = center_y_grid - (rect_h_f / np.float32(2.0))
                    if num_coord_params >= 1:
                        shapes_arr[i_shape, x_idx_coord] = x_origin_rect
                    if num_coord_params >= 2:
                        shapes_arr[i_shape, y_idx_coord] = y_origin_rect
                elif (
                    shape_type == "triangle"
                ):  # for triangle, (x,y) is one vertex, usually a base
                    # estimate an offset to somewhat center the triangle in its grid cell
                    # this is a heuristic
                    a_idx_tri = _find_param_index(param_names, "side_a")
                    b_idx_tri = _find_param_index(param_names, "side_b")
                    c_idx_tri = _find_param_index(param_names, "side_c")
                    side_a_tri = _safe_get_value(
                        shapes_arr[i_shape], a_idx_tri, np.float32(30.0)
                    )
                    side_b_tri = _safe_get_value(
                        shapes_arr[i_shape], b_idx_tri, np.float32(30.0)
                    )
                    side_c_tri = _safe_get_value(
                        shapes_arr[i_shape], c_idx_tri, np.float32(30.0)
                    )
                    avg_side_tri = (side_a_tri + side_b_tri + side_c_tri) / np.float32(
                        3.0
                    )
                    offset_factor_tri = np.float32(0.25)  # shift base point slightly
                    x_base_tri = center_x_grid - avg_side_tri * offset_factor_tri
                    y_base_tri = center_y_grid - avg_side_tri * offset_factor_tri
                    if num_coord_params >= 1:
                        shapes_arr[i_shape, x_idx_coord] = x_base_tri
                    if num_coord_params >= 2:
                        shapes_arr[i_shape, y_idx_coord] = y_base_tri
                else:  # for other shapes (circle, line), (x,y) is center or one endpoint
                    if num_coord_params >= 1:
                        shapes_arr[i_shape, x_idx_coord] = center_x_grid
                    if num_coord_params >= 2:
                        shapes_arr[i_shape, y_idx_coord] = center_y_grid
            elif coord_init_type == "zero":
                if num_coord_params >= 1:
                    shapes_arr[i_shape, x_idx_coord] = np.float32(0.0)
                if num_coord_params >= 2:
                    shapes_arr[i_shape, y_idx_coord] = np.float32(0.0)

            # clamp coordinates to be within image bounds
            if num_coord_params >= 1:
                shapes_arr[i_shape, x_idx_coord] = max(
                    np.float32(0.0), min(f_width, shapes_arr[i_shape, x_idx_coord])
                )
            if num_coord_params >= 2:
                shapes_arr[i_shape, y_idx_coord] = max(
                    np.float32(0.0), min(f_height, shapes_arr[i_shape, y_idx_coord])
                )

        # 3. fill color parameters [0,1]
        if (
            color_offset != -1
            and color_offset + num_fill_params <= total_params_per_shape
        ):
            for k_color in range(num_fill_params):
                shapes_arr[i_shape, color_offset + k_color] = np.float32(
                    np.random.random()
                )

        # 4. alpha parameter [0,1]
        if alpha_offset != -1 and alpha_offset < total_params_per_shape:
            shapes_arr[i_shape, alpha_offset] = default_alpha_val

        # 5. stroke color parameters [0,1]
        if (
            stroke_color_offset != -1
            and stroke_color_offset + num_stroke_color_params <= total_params_per_shape
        ):
            for k_stroke_color in range(num_stroke_color_params):
                shapes_arr[i_shape, stroke_color_offset + k_stroke_color] = np.float32(
                    np.random.random()
                )

    return shapes_arr
