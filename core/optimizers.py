import numpy as np
from numba import njit, int32, float32, prange, boolean
from numba.typed import List as NumbaList, Dict as NumbaDict
from numba import int32, int64, float32, boolean
from numba.core import types as numba_types
from core.shapes import (
    _render_shapes_rgb,
    _render_shapes_grayscale,
    get_coord_indices,
    _get_param_offsets,
)

from core.metrics import mse, psnr, ssim, gmsd
from typing import Tuple, Generator, Optional


@njit(
    numba_types.Tuple((float32, float32))(
        int32,
        int32,
        int64[:],
        float32[:],
        float32[:],
        numba_types.Tuple((int32, int32)),
        boolean,
        boolean,
        numba_types.unicode_type,
    ),
    fastmath=True,
)
def get_param_bounds(
    param_index: int,
    num_config_params: int,
    coord_indices: np.ndarray,
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    image_shape: Tuple[int, int],
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    shape_type: str,
) -> Tuple[np.float32, np.float32]:
    """
    Determines the min and max bounds for a given parameter index

    Bounds depend on whether the parameter is a config-defined param,
    a coordinate, a color component, or alpha

    :param param_index: The flat index of the parameter within a shape's parameter array
    :type param_index: int
    :param num_config_params: Number of parameters defined in the shape's JSON configuration
    :type num_config_params: int
    :param coord_indices: NumPy array of indices that correspond to coordinate parameters
    :type coord_indices: np.ndarray
    :param param_mins: NumPy array of minimum bounds for config-specific parameters
    :type param_mins: np.ndarray
    :param param_maxs: NumPy array of maximum bounds for config-specific parameters
    :type param_maxs: np.ndarray
    :param image_shape: Tuple (height, width) of the target image canvas
    :type image_shape: Tuple[int, int]
    :param shape_params_include_rgb: True if shape colors are RGB, False if grayscale
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have an alpha parameter
    :type shape_params_include_alpha: bool
    :param shape_type: String identifier for the shape type ("circle", "line", etc)
    :type shape_type: str
    :return: A tuple (min_bound, max_bound) for the specified parameter
    :rtype: Tuple[np.float32, np.float32]
    """
    height, width = image_shape
    f_height, f_width = np.float32(height), np.float32(width)

    # check if param_index falls within range of config-defined parameters
    if 0 <= param_index < num_config_params:
        if param_index < len(param_mins) and param_index < len(param_maxs):
            min_b, max_b = param_mins[param_index], param_maxs[param_index]
            # handle potential -inf/inf from config, convert to large float for numba
            min_b_f = np.float32(-1e18) if min_b == -np.inf else np.float32(min_b)
            max_b_f = np.float32(1e18) if max_b == np.inf else np.float32(max_b)
            # ensure min <= max, swapping if necessary
            return (max_b_f, min_b_f) if min_b_f > max_b_f else (min_b_f, max_b_f)
        else:
            # fallback if param_index is in config range but bounds arrays are too short
            return np.float32(0.0), np.float32(1.0)

    # check if param_index is a coordinate
    is_coord, coord_sub_index = False, -1
    for idx in range(len(coord_indices)):
        if coord_indices[idx] == param_index:
            is_coord, coord_sub_index = True, idx
            break

    if is_coord:
        # bounds for coordinates depend on shape type and image dimensions
        if shape_type in ("circle", "rectangle", "triangle", "image"):
            if coord_sub_index == 0:
                return np.float32(0.0), f_width
            if coord_sub_index == 1:
                return np.float32(0.0), f_height
        elif shape_type == "line":
            if coord_sub_index == 0:
                return np.float32(0.0), f_width
            if coord_sub_index == 1:
                return np.float32(0.0), f_height
        return np.float32(0.0), np.float32(1.0)

    # get offsets for color, alpha, stroke_color to identify these parameter types
    (
        _,
        _,
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

    # check if param_index is a fill color component (r,g,b or gray)
    if (
        color_offset != -1
        and color_offset <= param_index < color_offset + num_fill_params
    ):
        return np.float32(0.0), np.float32(1.0)

    # check if param_index is alpha parameter
    if alpha_offset != -1 and param_index == alpha_offset:
        return np.float32(0.0), np.float32(1.0)

    # check if param_index is a stroke color component (for lines)
    if (
        stroke_color_offset != -1
        and stroke_color_offset
        <= param_index
        < stroke_color_offset + num_stroke_color_params
    ):
        return np.float32(0.0), np.float32(1.0)

    # fallback for any other parameter types not explicitly handled
    return np.float32(0.0), np.float32(1.0)


@njit(
    numba_types.DictType(numba_types.unicode_type, int32)(
        numba_types.unicode_type,
        numba_types.ListType(numba_types.unicode_type),
        boolean,
        boolean,
    )
)
def _get_param_index_map(
    shape_type: str,
    config_param_names: NumbaList[str],
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
) -> NumbaDict:
    """
    Creates a map from parameter names to their flat indices in a shape's parameter array

    This map includes config-defined parameters, coordinates, color, and alpha

    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param config_param_names: Numba typed list of names for config-defined parameters
    :type config_param_names: NumbaList[str]
    :param shape_params_include_rgb: True if shape colors are RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have an alpha parameter
    :type shape_params_include_alpha: bool
    :return: A NumbaDict mapping parameter name (unicode_type) to its index (int32)
    :rtype: NumbaDict
    """
    param_index_map = NumbaDict.empty(
        key_type=numba_types.unicode_type, value_type=numba_types.int32
    )
    num_config_params = len(config_param_names)

    # first, map config-defined parameters
    for i in range(num_config_params):
        param_index_map[config_param_names[i]] = int32(i)

    # get offsets for standard parameter groups (coords, color, alpha, stroke)
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

    # map coordinate parameters (x, y or x1, y1)
    if num_coord_params >= 1:
        param_index_map["x"] = int32(coord_offset + 0)
    if num_coord_params >= 2:
        param_index_map["y"] = int32(coord_offset + 1)

    # map fill color parameters
    if color_offset != -1:
        if not shape_params_include_rgb:
            if num_fill_params >= 1:
                param_index_map["gray"] = int32(color_offset + 0)
        else:
            if num_fill_params >= 1:
                param_index_map["r"] = int32(color_offset + 0)
            if num_fill_params >= 2:
                param_index_map["g"] = int32(color_offset + 1)
            if num_fill_params >= 3:
                param_index_map["b"] = int32(color_offset + 2)

    # map alpha parameter
    if alpha_offset != -1:
        param_index_map["alpha"] = int32(alpha_offset)

    # map stroke color parameters (currently only for 'line' shape)
    if stroke_color_offset != -1:
        if not shape_params_include_rgb:
            if num_stroke_color_params >= 1:
                param_index_map["stroke_gray"] = int32(stroke_color_offset + 0)
        else:
            if num_stroke_color_params >= 1:
                param_index_map["stroke_r"] = int32(stroke_color_offset + 0)
            if num_stroke_color_params >= 2:
                param_index_map["stroke_g"] = int32(stroke_color_offset + 1)
            if num_stroke_color_params >= 3:
                param_index_map["stroke_b"] = int32(stroke_color_offset + 2)
    return param_index_map


@njit(fastmath=True)
def mutate_single_parameter(
    shape: np.ndarray,
    param_index: int,
    min_val: np.float32,
    max_val: np.float32,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    mutation_strength: float = 0.1,
    full_range_prob: float = 0.3,
) -> np.ndarray:
    """
    Mutates a single parameter of a shape, respecting bounds and fixed/frozen status

    If parameter is fixed to a specific value, it's set to that value
    If parameter is a coordinate frozen to its initial value, it's not changed
    Otherwise, it's mutated either by Gaussian noise or a full-range random value

    :param shape: 1D NumPy array of a single shape's parameters
    :type shape: np.ndarray
    :param param_index: Index of the parameter to mutate within the `shape` array
    :type param_index: int
    :param min_val: Minimum allowed value for this parameter
    :type min_val: np.float32
    :param max_val: Maximum allowed value for this parameter
    :type max_val: np.float32
    :param fixed_indices_specific: NumPy array of parameter indices fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: NumPy array of values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: NumPy array of coordinate parameter indices frozen to initial values
    :type coord_indices_to_freeze: np.ndarray
    :param mutation_strength: Factor determining std dev of Gaussian mutation relative to param range
    :type mutation_strength: float
    :param full_range_prob: Probability of choosing a new value uniformly from the entire valid range
    :type full_range_prob: float
    :return: The `shape` array with the specified parameter potentially mutated
    :rtype: np.ndarray
    """
    # check if parameter is fixed to a specific value
    is_fixed_specific = False
    fixed_value_to_set = np.float32(0.0)
    for i in range(len(fixed_indices_specific)):
        if fixed_indices_specific[i] == param_index:
            is_fixed_specific = True
            fixed_value_to_set = fixed_values_specific[i]
            break
    if is_fixed_specific:
        shape[param_index] = fixed_value_to_set
        return shape

    # check if parameter is a coordinate frozen to its initial value
    is_frozen_coord = False
    for i in range(len(coord_indices_to_freeze)):
        if coord_indices_to_freeze[i] == param_index:
            is_frozen_coord = True
            break
    if is_frozen_coord:
        return shape

    # proceed with mutation if not fixed or frozen
    current_value = shape[param_index]
    f_min_val, f_max_val = np.float32(min_val), np.float32(max_val)
    f_mutation_strength, f_full_range_prob = np.float32(mutation_strength), np.float32(
        full_range_prob
    )

    # define effective range, avoiding issues with inf bounds for sigma calculation
    large_val_threshold = np.float32(1e5)
    effective_min_bound = (
        f_min_val if f_min_val > -large_val_threshold else -large_val_threshold
    )
    effective_max_bound = (
        f_max_val if f_max_val < large_val_threshold else large_val_threshold
    )
    effective_param_range = max(
        np.float32(0.0), effective_max_bound - effective_min_bound
    )

    new_value_mutated: np.float32
    # decide mutation type: full range random or Gaussian perturbation
    if (
        effective_param_range <= np.float32(1e-9)
        or np.isinf(effective_param_range)
        or np.random.rand() < f_full_range_prob
    ):
        # mutate to a random value within effective (clamped) bounds
        sample_min_val = effective_min_bound
        sample_max_val = max(sample_min_val + np.float32(1e-9), effective_max_bound)
        new_value_mutated = np.float32(
            np.random.uniform(sample_min_val, sample_max_val)
        )
    else:
        # mutate using Gaussian noise proportional to range and strength
        sigma = max(np.float32(1e-9), effective_param_range * f_mutation_strength)
        new_value_mutated = np.float32(np.random.normal(current_value, sigma))

    # clamp mutated value to original min_val, max_val
    clamped_value = max(f_min_val, min(f_max_val, new_value_mutated))
    shape[param_index] = clamped_value
    return shape


@njit(fastmath=True)
def _calculate_combined_metric(
    img1_u8: np.ndarray,
    img2_u8: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
) -> np.float32:
    """
    Calculates a weighted combined metric score from a list of specified metrics

    Each individual metric should be normalized to [0,1] where 1 is better

    :param img1_u8: First image (NumPy array, uint8)
    :type img1_u8: np.ndarray
    :param img2_u8: Second image (NumPy array, uint8)
    :type img2_u8: np.ndarray
    :param metric_names: Numba typed list of metric names ("mse", "ssim", etc)
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of weights corresponding to `metric_names`
    :type metric_weights: np.ndarray
    :return: The combined weighted metric score, normalized to [0,1]
    :rtype: np.float32
    """
    combined_score_val = np.float32(0.0)
    num_metrics_used = len(metric_names)
    if num_metrics_used == 0 or len(metric_weights) != num_metrics_used:
        # invalid input if no metrics or mismatch between names and weights
        return np.float32(0.0)

    total_weight_applied = np.float32(0.0)
    for i in range(num_metrics_used):
        metric_name_str = metric_names[i]
        weight_val = metric_weights[i]
        if weight_val <= 0:
            continue

        score_val = np.float32(0.0)
        # dispatch to appropriate normalized metric function
        if metric_name_str == "mse":
            score_val = mse(img1_u8, img2_u8)
        elif metric_name_str == "psnr":
            score_val = psnr(img1_u8, img2_u8)
        elif metric_name_str == "ssim":
            score_val = ssim(img1_u8, img2_u8)
        elif metric_name_str == "gmsd":
            score_val = gmsd(img1_u8, img2_u8)

        combined_score_val += weight_val * score_val
        total_weight_applied += weight_val

    # if total weight applied isn't 1
    if total_weight_applied > np.float32(1e-6) and not np.isclose(
        total_weight_applied, np.float32(1.0)
    ):
        combined_score_val /= total_weight_applied

    # final score should be clipped to [0,1] as safeguard
    final_score_val = max(np.float32(0.0), min(np.float32(1.0), combined_score_val))
    return final_score_val


@njit(fastmath=True)
def _simulated_annealing_grayscale(
    initial_shapes: np.ndarray,
    shape_type: str,
    image_shape: Tuple[int, int],
    target_image: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    init_temp: float,
    cooling_rate: float,
    iterations: int,
    callback_interval: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]:
    """
    Core Numba JIT-compiled Simulated Annealing loop for grayscale images

    Optimizes shape parameters to match a target grayscale image
    Respects fixed/frozen parameters and can use an initial canvas

    :param initial_shapes: NumPy array of initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width) of the image canvas
    :type image_shape: Tuple[int, int]
    :param target_image: Target grayscale image (NumPy array, uint8, HxW)
    :type target_image: np.ndarray
    :param metric_names: Numba typed list of metric names to use
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of weights for metrics
    :type metric_weights: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: NumPy array of min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: NumPy array of max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_indices_specific: NumPy array of indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: NumPy array of values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: NumPy array of coordinate indices frozen to their initial values
    :type coord_indices_to_freeze: np.ndarray
    :param init_temp: Initial temperature for SA
    :type init_temp: float
    :param cooling_rate: Cooling rate for SA (e.g., 0.95)
    :type cooling_rate: float
    :param iterations: Total number of iterations for SA
    :type iterations: int
    :param callback_interval: Number of iterations between yielding intermediate results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered canvas (NumPy array, uint8, HxW)
                           If provided, new shapes are rendered on top of this
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: A 1-element NumPy array; if stop_flag[0] becomes 1, the loop terminates
    :type stop_flag: np.ndarray
    :return: A generator yielding (current_best_image, current_best_metric, current_best_shapes)
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]
    """
    height, width = image_shape
    h_i32, w_i32 = int32(height), int32(width)
    num_shapes, total_params = initial_shapes.shape
    num_config_params = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params)
    offsets_render = _get_param_offsets(
        shape_type,
        num_config_params,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )

    current_shapes_arr = initial_shapes.copy()
    # apply any specific fixed values to initial shapes
    if len(fixed_indices_specific) > 0:
        for i in range(len(fixed_indices_specific)):
            idx_fixed = fixed_indices_specific[i]
            if 0 <= idx_fixed < total_params:
                current_shapes_arr[:, idx_fixed] = fixed_values_specific[i]

    f_init_temp_val = np.float32(init_temp)
    f_cooling_rate_val = np.float32(cooling_rate)
    temperature_val = f_init_temp_val

    # initial evaluation of starting shapes
    current_image_u8_render: np.ndarray
    if initial_canvas is not None:
        current_image_u8_render = initial_canvas.copy()
    else:
        current_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)

    _render_shapes_grayscale(
        current_image_u8_render,
        current_shapes_arr,
        shape_type,
        param_names,
        offsets_render,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )

    current_metric_score = _calculate_combined_metric(
        current_image_u8_render, target_image, metric_names, metric_weights
    )
    best_shapes_arr = current_shapes_arr.copy()
    best_metric_score = current_metric_score
    best_image_u8_render = current_image_u8_render.copy()

    # determine which parameters are actually mutable
    all_param_indices_list = np.arange(total_params, dtype=np.int64)
    mutable_mask_arr = np.ones(total_params, dtype=np.bool_)
    if len(fixed_indices_specific) > 0:
        valid_fixed_indices_list = fixed_indices_specific[
            (fixed_indices_specific >= 0) & (fixed_indices_specific < total_params)
        ]
        if len(valid_fixed_indices_list) > 0:
            mutable_mask_arr[valid_fixed_indices_list] = False
    if len(coord_indices_to_freeze) > 0:
        valid_freeze_indices_list = coord_indices_to_freeze[
            (coord_indices_to_freeze >= 0) & (coord_indices_to_freeze < total_params)
        ]
        if len(valid_freeze_indices_list) > 0:
            mutable_mask_arr[valid_freeze_indices_list] = False
    mutable_param_indices_list = all_param_indices_list[mutable_mask_arr]
    num_mutable_params_available = len(mutable_param_indices_list)

    if num_mutable_params_available == 0:
        yield best_image_u8_render, np.float64(best_metric_score), best_shapes_arr
        return

    # sa main loop
    for i_iter in range(iterations):
        if stop_flag[0] == 1:
            break

        shape_idx_to_mutate = np.random.randint(0, num_shapes)
        param_idx_in_shape_to_mutate = mutable_param_indices_list[
            np.random.randint(0, num_mutable_params_available)
        ]

        min_bound_val, max_bound_val = get_param_bounds(
            int(param_idx_in_shape_to_mutate),
            num_config_params,
            coord_indices_for_bounds_calc,
            param_mins,
            param_maxs,
            image_shape,
            shape_params_include_rgb,
            shape_params_include_alpha,
            shape_type,
        )

        candidate_shapes_arr = current_shapes_arr.copy()
        mutate_single_parameter(
            candidate_shapes_arr[shape_idx_to_mutate],
            int(param_idx_in_shape_to_mutate),
            min_bound_val,
            max_bound_val,
            fixed_indices_specific,
            fixed_values_specific,
            coord_indices_to_freeze,
        )

        candidate_image_u8_render: np.ndarray
        if initial_canvas is not None:
            candidate_image_u8_render = initial_canvas.copy()
        else:
            candidate_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)

        _render_shapes_grayscale(
            candidate_image_u8_render,
            candidate_shapes_arr,
            shape_type,
            param_names,
            offsets_render,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
        candidate_metric_score = _calculate_combined_metric(
            candidate_image_u8_render, target_image, metric_names, metric_weights
        )

        delta_metric_val = candidate_metric_score - current_metric_score
        accept_candidate = False
        temp_epsilon_val = np.float32(1e-9)
        if delta_metric_val > 0:
            accept_candidate = True
        elif temperature_val > temp_epsilon_val:
            acceptance_probability = np.exp(
                np.float64(delta_metric_val) / np.float64(temperature_val)
            )
            if np.random.rand() < acceptance_probability:
                accept_candidate = True

        if accept_candidate:
            current_shapes_arr = candidate_shapes_arr
            current_metric_score = candidate_metric_score
            if current_metric_score > best_metric_score:
                best_shapes_arr = current_shapes_arr.copy()
                best_metric_score = current_metric_score
                best_image_u8_render = candidate_image_u8_render.copy()

        temperature_val *= f_cooling_rate_val

        if i_iter > 0 and i_iter % callback_interval == 0:
            yield best_image_u8_render.copy(), np.float64(
                best_metric_score
            ), best_shapes_arr.copy()

    yield best_image_u8_render.copy(), np.float64(
        best_metric_score
    ), best_shapes_arr.copy()


@njit(fastmath=True)
def _simulated_annealing_rgb(
    initial_shapes: np.ndarray,
    shape_type: str,
    image_shape: Tuple[int, int],
    target_image: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    init_temp: float,
    cooling_rate: float,
    iterations: int,
    callback_interval: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]:
    """
    Core Numba JIT-compiled Simulated Annealing loop for RGB images

    Optimizes shape parameters to match a target RGB image
    Functionally similar to the grayscale version but uses RGB rendering

    :param initial_shapes: NumPy array of initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width) of the image canvas
    :type image_shape: Tuple[int, int]
    :param target_image: Target RGB image (NumPy array, uint8, HxWxC)
    :type target_image: np.ndarray
    :param metric_names: Numba typed list of metric names to use
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of weights for metrics
    :type metric_weights: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: NumPy array of min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: NumPy array of max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_indices_specific: NumPy array of indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: NumPy array of values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: NumPy array of coordinate indices frozen to their initial values
    :type coord_indices_to_freeze: np.ndarray
    :param init_temp: Initial temperature for SA
    :type init_temp: float
    :param cooling_rate: Cooling rate for SA
    :type cooling_rate: float
    :param iterations: Total number of iterations for SA
    :type iterations: int
    :param callback_interval: Number of iterations between yielding intermediate results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered RGB canvas (NumPy array, uint8, HxWxC)
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: A 1-element NumPy array; if stop_flag[0] becomes 1, the loop terminates
    :type stop_flag: np.ndarray
    :return: A generator yielding (current_best_image, current_best_metric, current_best_shapes)
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]
    """
    height, width = image_shape
    h_i32, w_i32 = int32(height), int32(width)
    num_shapes, total_params = initial_shapes.shape
    num_config_params = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params)
    offsets_render = _get_param_offsets(
        shape_type,
        num_config_params,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    current_shapes_arr = initial_shapes.copy()
    if len(fixed_indices_specific) > 0:
        for i in range(len(fixed_indices_specific)):
            idx_fixed = fixed_indices_specific[i]
            if 0 <= idx_fixed < total_params:
                current_shapes_arr[:, idx_fixed] = fixed_values_specific[i]

    f_init_temp_val = np.float32(init_temp)
    f_cooling_rate_val = np.float32(cooling_rate)
    temperature_val = f_init_temp_val

    current_image_u8_render: np.ndarray
    if initial_canvas is not None:
        current_image_u8_render = initial_canvas.copy()
    else:
        current_image_u8_render = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)

    _render_shapes_rgb(
        current_image_u8_render,
        current_shapes_arr,
        shape_type,
        param_names,
        offsets_render,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    current_metric_score = _calculate_combined_metric(
        current_image_u8_render, target_image, metric_names, metric_weights
    )
    best_shapes_arr = current_shapes_arr.copy()
    best_metric_score = current_metric_score
    best_image_u8_render = current_image_u8_render.copy()

    all_param_indices_list = np.arange(total_params, dtype=np.int64)
    mutable_mask_arr = np.ones(total_params, dtype=np.bool_)
    if len(fixed_indices_specific) > 0:
        valid_fixed_indices_list = fixed_indices_specific[
            (fixed_indices_specific >= 0) & (fixed_indices_specific < total_params)
        ]
        if len(valid_fixed_indices_list) > 0:
            mutable_mask_arr[valid_fixed_indices_list] = False
    if len(coord_indices_to_freeze) > 0:
        valid_freeze_indices_list = coord_indices_to_freeze[
            (coord_indices_to_freeze >= 0) & (coord_indices_to_freeze < total_params)
        ]
        if len(valid_freeze_indices_list) > 0:
            mutable_mask_arr[valid_freeze_indices_list] = False
    mutable_param_indices_list = all_param_indices_list[mutable_mask_arr]
    num_mutable_params_available = len(mutable_param_indices_list)

    if num_mutable_params_available == 0:
        yield best_image_u8_render, np.float64(best_metric_score), best_shapes_arr
        return

    for i_iter in range(iterations):
        if stop_flag[0] == 1:
            break
        shape_idx_to_mutate = np.random.randint(0, num_shapes)
        param_idx_in_shape_to_mutate = mutable_param_indices_list[
            np.random.randint(0, num_mutable_params_available)
        ]
        min_bound_val, max_bound_val = get_param_bounds(
            int(param_idx_in_shape_to_mutate),
            num_config_params,
            coord_indices_for_bounds_calc,
            param_mins,
            param_maxs,
            image_shape,
            shape_params_include_rgb,
            shape_params_include_alpha,
            shape_type,
        )
        candidate_shapes_arr = current_shapes_arr.copy()
        mutate_single_parameter(
            candidate_shapes_arr[shape_idx_to_mutate],
            int(param_idx_in_shape_to_mutate),
            min_bound_val,
            max_bound_val,
            fixed_indices_specific,
            fixed_values_specific,
            coord_indices_to_freeze,
        )

        candidate_image_u8_render: np.ndarray
        if initial_canvas is not None:
            candidate_image_u8_render = initial_canvas.copy()
        else:
            candidate_image_u8_render = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)

        _render_shapes_rgb(
            candidate_image_u8_render,
            candidate_shapes_arr,
            shape_type,
            param_names,
            offsets_render,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
        candidate_metric_score = _calculate_combined_metric(
            candidate_image_u8_render, target_image, metric_names, metric_weights
        )

        delta_metric_val = candidate_metric_score - current_metric_score
        accept_candidate = False
        temp_epsilon_val = np.float32(1e-9)
        if delta_metric_val > 0:
            accept_candidate = True
        elif temperature_val > temp_epsilon_val:
            acceptance_probability = np.exp(
                np.float64(delta_metric_val) / np.float64(temperature_val)
            )
            if np.random.rand() < acceptance_probability:
                accept_candidate = True

        if accept_candidate:
            current_shapes_arr = candidate_shapes_arr
            current_metric_score = candidate_metric_score
            if current_metric_score > best_metric_score:
                best_shapes_arr = current_shapes_arr.copy()
                best_metric_score = current_metric_score
                best_image_u8_render = candidate_image_u8_render.copy()
        temperature_val *= f_cooling_rate_val

        if i_iter > 0 and i_iter % callback_interval == 0:
            yield best_image_u8_render.copy(), np.float64(
                best_metric_score
            ), best_shapes_arr.copy()

    yield best_image_u8_render.copy(), np.float64(
        best_metric_score
    ), best_shapes_arr.copy()


@njit(fastmath=True)
def _hill_climbing_grayscale(
    initial_shapes: np.ndarray,
    shape_type: str,
    image_shape: Tuple[int, int],
    target_image: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    iterations: int,
    callback_interval: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]:
    """
    Core Numba JIT-compiled Hill Climbing loop for grayscale images

    Iteratively mutates parameters and accepts changes if they improve or maintain the metric score

    :param initial_shapes: NumPy array of initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width) of the image canvas
    :type image_shape: Tuple[int, int]
    :param target_image: Target grayscale image (NumPy array, uint8, HxW)
    :type target_image: np.ndarray
    :param metric_names: Numba typed list of metric names to use
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of weights for metrics
    :type metric_weights: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: NumPy array of min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: NumPy array of max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_indices_specific: NumPy array of indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: NumPy array of values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: NumPy array of coordinate indices frozen to their initial values
    :type coord_indices_to_freeze: np.ndarray
    :param iterations: Total number of iterations for Hill Climbing
    :type iterations: int
    :param callback_interval: Number of iterations between yielding intermediate results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered canvas (NumPy array, uint8, HxW)
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: A 1-element NumPy array; if stop_flag[0] becomes 1, the loop terminates
    :type stop_flag: np.ndarray
    :return: A generator yielding (current_best_image, current_best_metric, current_best_shapes)
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]
    """
    height, width = image_shape
    h_i32, w_i32 = int32(height), int32(width)
    num_shapes, total_params = initial_shapes.shape
    num_config_params = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params)
    offsets_render = _get_param_offsets(
        shape_type,
        num_config_params,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    current_shapes_arr = initial_shapes.copy()
    if len(fixed_indices_specific) > 0:
        for i in range(len(fixed_indices_specific)):
            idx_fixed = fixed_indices_specific[i]
            if 0 <= idx_fixed < total_params:
                current_shapes_arr[:, idx_fixed] = fixed_values_specific[i]

    current_image_u8_render: np.ndarray
    if initial_canvas is not None:
        current_image_u8_render = initial_canvas.copy()
    else:
        current_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)

    _render_shapes_grayscale(
        current_image_u8_render,
        current_shapes_arr,
        shape_type,
        param_names,
        offsets_render,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    current_metric_score = _calculate_combined_metric(
        current_image_u8_render, target_image, metric_names, metric_weights
    )
    best_shapes_arr = current_shapes_arr.copy()
    best_metric_score = current_metric_score
    best_image_u8_render = current_image_u8_render.copy()

    all_param_indices_list = np.arange(total_params, dtype=np.int64)
    mutable_mask_arr = np.ones(total_params, dtype=np.bool_)
    if len(fixed_indices_specific) > 0:
        valid_fixed_indices_list = fixed_indices_specific[
            (fixed_indices_specific >= 0) & (fixed_indices_specific < total_params)
        ]
        if len(valid_fixed_indices_list) > 0:
            mutable_mask_arr[valid_fixed_indices_list] = False
    if len(coord_indices_to_freeze) > 0:
        valid_freeze_indices_list = coord_indices_to_freeze[
            (coord_indices_to_freeze >= 0) & (coord_indices_to_freeze < total_params)
        ]
        if len(valid_freeze_indices_list) > 0:
            mutable_mask_arr[valid_freeze_indices_list] = False
    mutable_param_indices_list = all_param_indices_list[mutable_mask_arr]
    num_mutable_params_available = len(mutable_param_indices_list)

    if num_mutable_params_available == 0:
        yield best_image_u8_render, np.float64(best_metric_score), best_shapes_arr
        return

    for i_iter in range(iterations):
        if stop_flag[0] == 1:
            break
        shape_idx_to_mutate = np.random.randint(0, num_shapes)
        param_idx_in_shape_to_mutate = mutable_param_indices_list[
            np.random.randint(0, num_mutable_params_available)
        ]
        min_bound_val, max_bound_val = get_param_bounds(
            int(param_idx_in_shape_to_mutate),
            num_config_params,
            coord_indices_for_bounds_calc,
            param_mins,
            param_maxs,
            image_shape,
            shape_params_include_rgb,
            shape_params_include_alpha,
            shape_type,
        )
        candidate_shapes_arr = current_shapes_arr.copy()
        mutate_single_parameter(
            candidate_shapes_arr[shape_idx_to_mutate],
            int(param_idx_in_shape_to_mutate),
            min_bound_val,
            max_bound_val,
            fixed_indices_specific,
            fixed_values_specific,
            coord_indices_to_freeze,
        )

        candidate_image_u8_render: np.ndarray
        if initial_canvas is not None:
            candidate_image_u8_render = initial_canvas.copy()
        else:
            candidate_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)

        _render_shapes_grayscale(
            candidate_image_u8_render,
            candidate_shapes_arr,
            shape_type,
            param_names,
            offsets_render,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
        candidate_metric_score = _calculate_combined_metric(
            candidate_image_u8_render, target_image, metric_names, metric_weights
        )

        if candidate_metric_score >= current_metric_score:
            current_shapes_arr = candidate_shapes_arr
            current_metric_score = candidate_metric_score
            if current_metric_score > best_metric_score:
                best_shapes_arr = current_shapes_arr.copy()
                best_metric_score = current_metric_score
                best_image_u8_render = candidate_image_u8_render.copy()

        if i_iter > 0 and i_iter % callback_interval == 0:
            yield best_image_u8_render.copy(), np.float64(
                best_metric_score
            ), best_shapes_arr.copy()

    yield best_image_u8_render.copy(), np.float64(
        best_metric_score
    ), best_shapes_arr.copy()


@njit(fastmath=True)
def _hill_climbing_rgb(
    initial_shapes: np.ndarray,
    shape_type: str,
    image_shape: Tuple[int, int],
    target_image: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    iterations: int,
    callback_interval: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
) -> Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]:
    """
    Core Numba JIT-compiled Hill Climbing loop for RGB images

    Functionally similar to the grayscale version but uses RGB rendering

    :param initial_shapes: NumPy array of initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width) of the image canvas
    :type image_shape: Tuple[int, int]
    :param target_image: Target RGB image (NumPy array, uint8, HxWxC)
    :type target_image: np.ndarray
    :param metric_names: Numba typed list of metric names to use
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of weights for metrics
    :type metric_weights: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: NumPy array of min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: NumPy array of max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_indices_specific: NumPy array of indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: NumPy array of values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: NumPy array of coordinate indices frozen to their initial values
    :type coord_indices_to_freeze: np.ndarray
    :param iterations: Total number of iterations for Hill Climbing
    :type iterations: int
    :param callback_interval: Number of iterations between yielding intermediate results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered RGB canvas (NumPy array, uint8, HxWxC)
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: A 1-element NumPy array; if stop_flag[0] becomes 1, the loop terminates
    :type stop_flag: np.ndarray
    :return: A generator yielding (current_best_image, current_best_metric, current_best_shapes)
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]
    """
    height, width = image_shape
    h_i32, w_i32 = int32(height), int32(width)
    num_shapes, total_params = initial_shapes.shape
    num_config_params = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params)
    offsets_render = _get_param_offsets(
        shape_type,
        num_config_params,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    current_shapes_arr = initial_shapes.copy()
    if len(fixed_indices_specific) > 0:
        for i in range(len(fixed_indices_specific)):
            idx_fixed = fixed_indices_specific[i]
            if 0 <= idx_fixed < total_params:
                current_shapes_arr[:, idx_fixed] = fixed_values_specific[i]

    current_image_u8_render: np.ndarray
    if initial_canvas is not None:
        current_image_u8_render = initial_canvas.copy()
    else:
        current_image_u8_render = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)

    _render_shapes_rgb(
        current_image_u8_render,
        current_shapes_arr,
        shape_type,
        param_names,
        offsets_render,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    current_metric_score = _calculate_combined_metric(
        current_image_u8_render, target_image, metric_names, metric_weights
    )
    best_shapes_arr = current_shapes_arr.copy()
    best_metric_score = current_metric_score
    best_image_u8_render = current_image_u8_render.copy()

    all_param_indices_list = np.arange(total_params, dtype=np.int64)
    mutable_mask_arr = np.ones(total_params, dtype=np.bool_)
    if len(fixed_indices_specific) > 0:
        valid_fixed_indices_list = fixed_indices_specific[
            (fixed_indices_specific >= 0) & (fixed_indices_specific < total_params)
        ]
        if len(valid_fixed_indices_list) > 0:
            mutable_mask_arr[valid_fixed_indices_list] = False
    if len(coord_indices_to_freeze) > 0:
        valid_freeze_indices_list = coord_indices_to_freeze[
            (coord_indices_to_freeze >= 0) & (coord_indices_to_freeze < total_params)
        ]
        if len(valid_freeze_indices_list) > 0:
            mutable_mask_arr[valid_freeze_indices_list] = False
    mutable_param_indices_list = all_param_indices_list[mutable_mask_arr]
    num_mutable_params_available = len(mutable_param_indices_list)

    if num_mutable_params_available == 0:
        yield best_image_u8_render, np.float64(best_metric_score), best_shapes_arr
        return

    for i_iter in range(iterations):
        if stop_flag[0] == 1:
            break
        shape_idx_to_mutate = np.random.randint(0, num_shapes)
        param_idx_in_shape_to_mutate = mutable_param_indices_list[
            np.random.randint(0, num_mutable_params_available)
        ]
        min_bound_val, max_bound_val = get_param_bounds(
            int(param_idx_in_shape_to_mutate),
            num_config_params,
            coord_indices_for_bounds_calc,
            param_mins,
            param_maxs,
            image_shape,
            shape_params_include_rgb,
            shape_params_include_alpha,
            shape_type,
        )
        candidate_shapes_arr = current_shapes_arr.copy()
        mutate_single_parameter(
            candidate_shapes_arr[shape_idx_to_mutate],
            int(param_idx_in_shape_to_mutate),
            min_bound_val,
            max_bound_val,
            fixed_indices_specific,
            fixed_values_specific,
            coord_indices_to_freeze,
        )

        candidate_image_u8_render: np.ndarray
        if initial_canvas is not None:
            candidate_image_u8_render = initial_canvas.copy()
        else:
            candidate_image_u8_render = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)
        _render_shapes_rgb(
            candidate_image_u8_render,
            candidate_shapes_arr,
            shape_type,
            param_names,
            offsets_render,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
        candidate_metric_score = _calculate_combined_metric(
            candidate_image_u8_render, target_image, metric_names, metric_weights
        )

        if candidate_metric_score >= current_metric_score:
            current_shapes_arr = candidate_shapes_arr
            current_metric_score = candidate_metric_score
            if current_metric_score > best_metric_score:
                best_shapes_arr = current_shapes_arr.copy()
                best_metric_score = current_metric_score
                best_image_u8_render = candidate_image_u8_render.copy()

        if i_iter > 0 and i_iter % callback_interval == 0:
            yield best_image_u8_render.copy(), np.float64(
                best_metric_score
            ), best_shapes_arr.copy()

    yield best_image_u8_render.copy(), np.float64(
        best_metric_score
    ), best_shapes_arr.copy()


@njit(fastmath=True)
def _initialize_pso_particles(
    initial_shapes: np.ndarray,
    swarm_size: int,
    shape_type: str,
    image_shape: Tuple[int, int],
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    perturbation_strength_factor: float = 0.05,
    min_perturbation_sigma: float = 1e-4,
) -> np.ndarray:
    """
    Initializes PSO particles based on a provided set of `initial_shapes`

    Each particle is a slight perturbation of `initial_shapes`, respecting bounds
    and fixed/frozen parameters

    :param initial_shapes: Base NumPy array of shape parameters to perturb from
    :type initial_shapes: np.ndarray
    :param swarm_size: Number of particles in the swarm
    :type swarm_size: int
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
    :param fixed_indices_specific: NumPy array of indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: NumPy array of values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: NumPy array of coordinate indices frozen to their initial values
    :type coord_indices_to_freeze: np.ndarray
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param perturbation_strength_factor: Factor for perturbation std dev relative to param range
    :type perturbation_strength_factor: float
    :param min_perturbation_sigma: Minimum std dev for Gaussian perturbation
    :type min_perturbation_sigma: float
    :return: A 3D NumPy array representing particle positions (swarm_size x num_shapes x num_params)
    :rtype: np.ndarray
    """
    num_shapes_val, num_params_val = initial_shapes.shape
    num_config_params_val = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params_val)
    particles_arr = np.zeros(
        (swarm_size, num_shapes_val, num_params_val), dtype=np.float32
    )
    f_perturb_factor_val = np.float32(perturbation_strength_factor)
    f_min_sigma_val = np.float32(min_perturbation_sigma)

    for i_particle in range(swarm_size):
        for s_shape in range(num_shapes_val):
            for p_param in range(num_params_val):
                initial_param_value = initial_shapes[s_shape, p_param]
                particle_param_value = initial_param_value

                is_fixed_specific_val = False
                fixed_value_to_set = np.float32(0.0)
                for fi_idx in range(len(fixed_indices_specific)):
                    if fixed_indices_specific[fi_idx] == p_param:
                        is_fixed_specific_val = True
                        fixed_value_to_set = fixed_values_specific[fi_idx]
                        break
                is_frozen_initial_val = False
                for ci_idx in range(len(coord_indices_to_freeze)):
                    if coord_indices_to_freeze[ci_idx] == p_param:
                        is_frozen_initial_val = True
                        break

                if is_fixed_specific_val:
                    particle_param_value = fixed_value_to_set
                elif is_frozen_initial_val:
                    particle_param_value = initial_param_value
                else:
                    min_b_val, max_b_val = get_param_bounds(
                        p_param,
                        num_config_params_val,
                        coord_indices_for_bounds_calc,
                        param_mins,
                        param_maxs,
                        image_shape,
                        shape_params_include_rgb,
                        shape_params_include_alpha,
                        shape_type,
                    )
                    param_range_val = max(np.float32(0.0), max_b_val - min_b_val)
                    sigma_val = f_min_sigma_val
                    if param_range_val > 1e-9 and np.isfinite(param_range_val):
                        sigma_val = max(
                            f_min_sigma_val, param_range_val * f_perturb_factor_val
                        )

                    perturbation_val = np.float32(np.random.normal(0.0, sigma_val))
                    perturbed_value = initial_param_value + perturbation_val
                    particle_param_value = max(
                        min_b_val, min(max_b_val, perturbed_value)
                    )
                particles_arr[i_particle, s_shape, p_param] = particle_param_value
    return particles_arr


@njit(fastmath=True)
def _pso_grayscale(
    initial_shapes: np.ndarray,
    shape_type: str,
    image_shape: Tuple[int, int],
    target_image: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    swarm_size: int,
    cognitive_param: float,
    social_param: float,
    inertia_weight: float,
    iterations: int,
    callback_interval: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    offsets_render_tuple: Tuple,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
) -> Generator[
    Tuple[np.ndarray, np.float64, np.ndarray, NumbaList[np.ndarray]], None, None
]:
    """
    Core Numba JIT-compiled Particle Swarm Optimization loop for grayscale images

    :param initial_shapes: Base shapes to initialize particles from
    :type initial_shapes: np.ndarray
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width)
    :type image_shape: Tuple[int, int]
    :param target_image: Target grayscale image (NumPy array, uint8, HxW)
    :type target_image: np.ndarray
    :param metric_names: Numba typed list of metric names
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of metric weights
    :type metric_weights: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: Min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: Max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_indices_specific: Indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: Values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: Coordinate indices frozen to initial values
    :type coord_indices_to_freeze: np.ndarray
    :param swarm_size: Number of particles in the swarm
    :type swarm_size: int
    :param cognitive_param: PSO cognitive component weight (c1)
    :type cognitive_param: float
    :param social_param: PSO social component weight (c2)
    :type social_param: float
    :param inertia_weight: PSO inertia weight (w)
    :type inertia_weight: float
    :param iterations: Total number of iterations for PSO
    :type iterations: int
    :param callback_interval: Iterations between yielding results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param offsets_render_tuple: Precomputed tuple of parameter offsets for rendering
    :type offsets_render_tuple: Tuple
    :param initial_canvas: Optional prerendered canvas (NumPy array, uint8, HxW)
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: A 1-element NumPy array for stopping the loop
    :type stop_flag: np.ndarray
    :return: A generator yielding (gbest_image, gbest_score, gbest_position, pbest_scores_history)
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray, NumbaList[np.ndarray]], None, None]
    """
    height, width = image_shape
    h_i32, w_i32 = int32(height), int32(width)
    num_shapes_val, num_params_val = initial_shapes.shape
    num_config_params_val = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params_val)

    f_cognitive_param_val = np.float32(cognitive_param)
    f_social_param_val = np.float32(social_param)
    f_inertia_weight_val = np.float32(inertia_weight)

    particles_arr = _initialize_pso_particles(
        initial_shapes,
        swarm_size,
        shape_type,
        image_shape,
        param_names,
        param_mins,
        param_maxs,
        fixed_indices_specific,
        fixed_values_specific,
        coord_indices_to_freeze,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    velocities_arr = np.zeros(
        (swarm_size, num_shapes_val, num_params_val), dtype=np.float32
    )

    pbest_positions_arr = particles_arr.copy()
    pbest_scores_arr = np.full(swarm_size, -np.inf, dtype=np.float32)

    initial_best_idx_val = 0
    initial_best_score_val = -np.inf
    temp_scores_arr = np.empty(swarm_size, dtype=np.float32)

    # prange enables parallel evaluation of initial swarm
    for i_particle in prange(swarm_size):
        img_eval_init: np.ndarray
        if initial_canvas is not None:
            img_eval_init = initial_canvas.copy()
        else:
            img_eval_init = np.full((h_i32, w_i32), 255, dtype=np.uint8)
        _render_shapes_grayscale(
            img_eval_init,
            particles_arr[i_particle],
            shape_type,
            param_names,
            offsets_render_tuple,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
        score_init_val = _calculate_combined_metric(
            img_eval_init, target_image, metric_names, metric_weights
        )
        temp_scores_arr[i_particle] = np.float32(score_init_val)
        pbest_scores_arr[i_particle] = temp_scores_arr[i_particle]

    for i_particle_idx in range(swarm_size):
        if temp_scores_arr[i_particle_idx] > initial_best_score_val:
            initial_best_score_val = temp_scores_arr[i_particle_idx]
            initial_best_idx_val = i_particle_idx

    gbest_position_arr = particles_arr[initial_best_idx_val].copy()
    gbest_score_val = initial_best_score_val
    gbest_image_u8_render = np.zeros(0, dtype=np.uint8)
    pbest_scores_history_list = NumbaList.empty_list(numba_types.float32[:])
    pbest_scores_history_list.append(pbest_scores_arr.copy())

    if initial_canvas is not None:
        gbest_image_u8_render = initial_canvas.copy()
    else:
        gbest_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)
    _render_shapes_grayscale(
        gbest_image_u8_render,
        gbest_position_arr,
        shape_type,
        param_names,
        offsets_render_tuple,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )

    current_scores_eval_arr = np.empty(swarm_size, dtype=np.float32)
    for iteration_num in range(iterations):
        if stop_flag[0] == 1:
            break

        # prange for parallel update of particle velocities and positions
        for i_particle_pso in prange(swarm_size):
            for s_shape_pso in range(num_shapes_val):
                original_particle_s_p_vals = particles_arr[
                    i_particle_pso, s_shape_pso
                ].copy()
                r1_vals = np.random.rand(num_params_val).astype(np.float32)
                r2_vals = np.random.rand(num_params_val).astype(np.float32)
                cognitive_update_vals = (
                    f_cognitive_param_val
                    * r1_vals
                    * (
                        pbest_positions_arr[i_particle_pso, s_shape_pso]
                        - particles_arr[i_particle_pso, s_shape_pso]
                    )
                )
                social_update_vals = (
                    f_social_param_val
                    * r2_vals
                    * (
                        gbest_position_arr[s_shape_pso]
                        - particles_arr[i_particle_pso, s_shape_pso]
                    )
                )
                new_velocity_vals = (
                    f_inertia_weight_val * velocities_arr[i_particle_pso, s_shape_pso]
                    + cognitive_update_vals
                    + social_update_vals
                )
                velocities_arr[i_particle_pso, s_shape_pso] = new_velocity_vals
                candidate_position_s_vals = (
                    particles_arr[i_particle_pso, s_shape_pso]
                    + velocities_arr[i_particle_pso, s_shape_pso]
                )

                for p_param_pso in range(num_params_val):
                    is_fixed_val, fixed_val_set = False, np.float32(0.0)
                    for fi_idx_pso in range(len(fixed_indices_specific)):
                        if fixed_indices_specific[fi_idx_pso] == p_param_pso:
                            is_fixed_val, fixed_val_set = (
                                True,
                                fixed_values_specific[fi_idx_pso],
                            )
                            break
                    if is_fixed_val:
                        candidate_position_s_vals[p_param_pso] = fixed_val_set
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] = 0.0
                        continue

                    is_frozen_val = False
                    for ci_idx_pso in range(len(coord_indices_to_freeze)):
                        if coord_indices_to_freeze[ci_idx_pso] == p_param_pso:
                            is_frozen_val = True
                            break
                    if is_frozen_val:
                        candidate_position_s_vals[p_param_pso] = (
                            original_particle_s_p_vals[p_param_pso]
                        )
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] = 0.0
                        continue

                    min_b_pso, max_b_pso = get_param_bounds(
                        p_param_pso,
                        num_config_params_val,
                        coord_indices_for_bounds_calc,
                        param_mins,
                        param_maxs,
                        image_shape,
                        shape_params_include_rgb,
                        shape_params_include_alpha,
                        shape_type,
                    )
                    if candidate_position_s_vals[p_param_pso] < min_b_pso:
                        candidate_position_s_vals[p_param_pso] = min_b_pso
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] *= -0.5
                    elif candidate_position_s_vals[p_param_pso] > max_b_pso:
                        candidate_position_s_vals[p_param_pso] = max_b_pso
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] *= -0.5
                particles_arr[i_particle_pso, s_shape_pso] = candidate_position_s_vals

        # prange for parallel evaluation of new particle positions
        for i_particle_eval in prange(swarm_size):
            img_eval_loop: np.ndarray
            if initial_canvas is not None:
                img_eval_loop = initial_canvas.copy()
            else:
                img_eval_loop = np.full((h_i32, w_i32), 255, dtype=np.uint8)
            _render_shapes_grayscale(
                img_eval_loop,
                particles_arr[i_particle_eval],
                shape_type,
                param_names,
                offsets_render_tuple,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
            score_eval = _calculate_combined_metric(
                img_eval_loop, target_image, metric_names, metric_weights
            )
            current_scores_eval_arr[i_particle_eval] = np.float32(score_eval)

        gbest_changed_this_iter_flag = False
        best_score_in_iter_val = gbest_score_val
        best_idx_in_iter_val = -1

        # this loop must be serial for correct gbest update
        for i_particle_update in range(swarm_size):
            if (
                current_scores_eval_arr[i_particle_update]
                > pbest_scores_arr[i_particle_update]
            ):
                pbest_scores_arr[i_particle_update] = current_scores_eval_arr[
                    i_particle_update
                ]
                pbest_positions_arr[i_particle_update] = particles_arr[
                    i_particle_update
                ].copy()
            if pbest_scores_arr[i_particle_update] > best_score_in_iter_val:
                best_score_in_iter_val = pbest_scores_arr[i_particle_update]
                best_idx_in_iter_val = i_particle_update
        if best_idx_in_iter_val != -1:
            gbest_score_val = best_score_in_iter_val
            gbest_position_arr = pbest_positions_arr[best_idx_in_iter_val].copy()
            gbest_changed_this_iter_flag = True

        pbest_scores_history_list.append(pbest_scores_arr.copy())

        if iteration_num > 0 and iteration_num % callback_interval == 0:
            if gbest_changed_this_iter_flag:
                if initial_canvas is not None:
                    gbest_image_u8_render = initial_canvas.copy()
                else:
                    gbest_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)
                _render_shapes_grayscale(
                    gbest_image_u8_render,
                    gbest_position_arr,
                    shape_type,
                    param_names,
                    offsets_render_tuple,
                    shape_params_include_rgb,
                    shape_params_include_alpha,
                )
            yield gbest_image_u8_render.copy(), np.float64(
                gbest_score_val
            ), gbest_position_arr.copy(), pbest_scores_history_list

    if gbest_changed_this_iter_flag or iteration_num % callback_interval != 0:
        if initial_canvas is not None:
            gbest_image_u8_render = initial_canvas.copy()
        else:
            gbest_image_u8_render = np.full((h_i32, w_i32), 255, dtype=np.uint8)
        _render_shapes_grayscale(
            gbest_image_u8_render,
            gbest_position_arr,
            shape_type,
            param_names,
            offsets_render_tuple,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
    yield gbest_image_u8_render.copy(), np.float64(
        gbest_score_val
    ), gbest_position_arr.copy(), pbest_scores_history_list


@njit(fastmath=True)
def _pso_rgb(
    initial_shapes: np.ndarray,
    shape_type: str,
    image_shape: Tuple[int, int],
    target_image: np.ndarray,
    metric_names: NumbaList[str],
    metric_weights: np.ndarray,
    param_names: NumbaList[str],
    param_mins: np.ndarray,
    param_maxs: np.ndarray,
    fixed_indices_specific: np.ndarray,
    fixed_values_specific: np.ndarray,
    coord_indices_to_freeze: np.ndarray,
    swarm_size: int,
    cognitive_param: float,
    social_param: float,
    inertia_weight: float,
    iterations: int,
    callback_interval: int,
    shape_params_include_rgb: bool,
    shape_params_include_alpha: bool,
    offsets_render_tuple: Tuple,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
) -> Generator[
    Tuple[np.ndarray, np.float64, np.ndarray, NumbaList[np.ndarray]], None, None
]:
    """
    Core Numba JIT-compiled Particle Swarm Optimization loop for RGB images

    Functionally similar to the grayscale version but uses RGB rendering

    :param initial_shapes: Base shapes to initialize particles from
    :type initial_shapes: np.ndarray
    :param shape_type: String identifier for the shape type
    :type shape_type: str
    :param image_shape: Tuple (height, width)
    :type image_shape: Tuple[int, int]
    :param target_image: Target RGB image (NumPy array, uint8, HxWxC)
    :type target_image: np.ndarray
    :param metric_names: Numba typed list of metric names
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of metric weights
    :type metric_weights: np.ndarray
    :param param_names: Numba typed list of config-defined parameter names
    :type param_names: NumbaList[str]
    :param param_mins: Min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: Max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_indices_specific: Indices for parameters fixed to specific values
    :type fixed_indices_specific: np.ndarray
    :param fixed_values_specific: Values for `fixed_indices_specific`
    :type fixed_values_specific: np.ndarray
    :param coord_indices_to_freeze: Coordinate indices frozen to initial values
    :type coord_indices_to_freeze: np.ndarray
    :param swarm_size: Number of particles in the swarm
    :type swarm_size: int
    :param cognitive_param: PSO cognitive component weight (c1)
    :type cognitive_param: float
    :param social_param: PSO social component weight (c2)
    :type social_param: float
    :param inertia_weight: PSO inertia weight (w)
    :type inertia_weight: float
    :param iterations: Total number of iterations for PSO
    :type iterations: int
    :param callback_interval: Iterations between yielding results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape color structure is RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes include an alpha parameter
    :type shape_params_include_alpha: bool
    :param offsets_render_tuple: Precomputed tuple of parameter offsets for rendering
    :type offsets_render_tuple: Tuple
    :param initial_canvas: Optional prerendered RGB canvas (NumPy array, uint8, HxWxC)
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: A 1-element NumPy array for stopping the loop
    :type stop_flag: np.ndarray
    :return: A generator yielding (gbest_image, gbest_score, gbest_position, pbest_scores_history)
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray, NumbaList[np.ndarray]], None, None]
    """
    height, width = image_shape
    h_i32, w_i32 = int32(height), int32(width)
    num_shapes_val, num_params_val = initial_shapes.shape
    num_config_params_val = len(param_names)
    coord_indices_for_bounds_calc = get_coord_indices(shape_type, num_config_params_val)
    f_cognitive_param_val = np.float32(cognitive_param)
    f_social_param_val = np.float32(social_param)
    f_inertia_weight_val = np.float32(inertia_weight)
    particles_arr = _initialize_pso_particles(
        initial_shapes,
        swarm_size,
        shape_type,
        image_shape,
        param_names,
        param_mins,
        param_maxs,
        fixed_indices_specific,
        fixed_values_specific,
        coord_indices_to_freeze,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    velocities_arr = np.zeros(
        (swarm_size, num_shapes_val, num_params_val), dtype=np.float32
    )
    pbest_positions_arr = particles_arr.copy()
    pbest_scores_arr = np.full(swarm_size, -np.inf, dtype=np.float32)
    initial_best_idx_val = 0
    initial_best_score_val = -np.inf
    temp_scores_arr = np.empty(swarm_size, dtype=np.float32)

    for i_particle in prange(swarm_size):
        img_eval_init: np.ndarray
        if initial_canvas is not None:
            img_eval_init = initial_canvas.copy()
        else:
            img_eval_init = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)
        _render_shapes_rgb(
            img_eval_init,
            particles_arr[i_particle],
            shape_type,
            param_names,
            offsets_render_tuple,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
        score_init_val = _calculate_combined_metric(
            img_eval_init, target_image, metric_names, metric_weights
        )
        temp_scores_arr[i_particle] = np.float32(score_init_val)
        pbest_scores_arr[i_particle] = temp_scores_arr[i_particle]

    for i_particle_idx in range(swarm_size):
        if temp_scores_arr[i_particle_idx] > initial_best_score_val:
            initial_best_score_val = temp_scores_arr[i_particle_idx]
            initial_best_idx_val = i_particle_idx

    gbest_position_arr = particles_arr[initial_best_idx_val].copy()
    gbest_score_val = initial_best_score_val
    gbest_image_u8_render = np.zeros(0, dtype=np.uint8)
    pbest_scores_history_list = NumbaList.empty_list(numba_types.float32[:])
    pbest_scores_history_list.append(pbest_scores_arr.copy())

    if initial_canvas is not None:
        gbest_image_u8_render = initial_canvas.copy()
    else:
        gbest_image_u8_render = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)
    _render_shapes_rgb(
        gbest_image_u8_render,
        gbest_position_arr,
        shape_type,
        param_names,
        offsets_render_tuple,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )

    current_scores_eval_arr = np.empty(swarm_size, dtype=np.float32)
    for iteration_num in range(iterations):
        if stop_flag[0] == 1:
            break
        for i_particle_pso in prange(swarm_size):
            for s_shape_pso in range(num_shapes_val):
                original_particle_s_p_vals = particles_arr[
                    i_particle_pso, s_shape_pso
                ].copy()
                r1_vals = np.random.rand(num_params_val).astype(np.float32)
                r2_vals = np.random.rand(num_params_val).astype(np.float32)
                cognitive_update_vals = (
                    f_cognitive_param_val
                    * r1_vals
                    * (
                        pbest_positions_arr[i_particle_pso, s_shape_pso]
                        - particles_arr[i_particle_pso, s_shape_pso]
                    )
                )
                social_update_vals = (
                    f_social_param_val
                    * r2_vals
                    * (
                        gbest_position_arr[s_shape_pso]
                        - particles_arr[i_particle_pso, s_shape_pso]
                    )
                )
                new_velocity_vals = (
                    f_inertia_weight_val * velocities_arr[i_particle_pso, s_shape_pso]
                    + cognitive_update_vals
                    + social_update_vals
                )
                velocities_arr[i_particle_pso, s_shape_pso] = new_velocity_vals
                candidate_position_s_vals = (
                    particles_arr[i_particle_pso, s_shape_pso]
                    + velocities_arr[i_particle_pso, s_shape_pso]
                )
                for p_param_pso in range(num_params_val):
                    is_fixed_val, fixed_val_set = False, np.float32(0.0)
                    for fi_idx_pso in range(len(fixed_indices_specific)):
                        if fixed_indices_specific[fi_idx_pso] == p_param_pso:
                            is_fixed_val, fixed_val_set = (
                                True,
                                fixed_values_specific[fi_idx_pso],
                            )
                            break
                    if is_fixed_val:
                        candidate_position_s_vals[p_param_pso] = fixed_val_set
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] = 0.0
                        continue
                    is_frozen_val = False
                    for ci_idx_pso in range(len(coord_indices_to_freeze)):
                        if coord_indices_to_freeze[ci_idx_pso] == p_param_pso:
                            is_frozen_val = True
                            break
                    if is_frozen_val:
                        candidate_position_s_vals[p_param_pso] = (
                            original_particle_s_p_vals[p_param_pso]
                        )
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] = 0.0
                        continue
                    min_b_pso, max_b_pso = get_param_bounds(
                        p_param_pso,
                        num_config_params_val,
                        coord_indices_for_bounds_calc,
                        param_mins,
                        param_maxs,
                        image_shape,
                        shape_params_include_rgb,
                        shape_params_include_alpha,
                        shape_type,
                    )
                    if candidate_position_s_vals[p_param_pso] < min_b_pso:
                        candidate_position_s_vals[p_param_pso] = min_b_pso
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] *= -0.5
                    elif candidate_position_s_vals[p_param_pso] > max_b_pso:
                        candidate_position_s_vals[p_param_pso] = max_b_pso
                        velocities_arr[i_particle_pso, s_shape_pso, p_param_pso] *= -0.5
                particles_arr[i_particle_pso, s_shape_pso] = candidate_position_s_vals

        for i_particle_eval in prange(swarm_size):
            img_eval_loop: np.ndarray
            if initial_canvas is not None:
                img_eval_loop = initial_canvas.copy()
            else:
                img_eval_loop = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)
            _render_shapes_rgb(
                img_eval_loop,
                particles_arr[i_particle_eval],
                shape_type,
                param_names,
                offsets_render_tuple,
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
            score_eval = _calculate_combined_metric(
                img_eval_loop, target_image, metric_names, metric_weights
            )
            current_scores_eval_arr[i_particle_eval] = np.float32(score_eval)

        gbest_changed_this_iter_flag = False
        best_score_in_iter_val = gbest_score_val
        best_idx_in_iter_val = -1
        for i_particle_update in range(swarm_size):
            if (
                current_scores_eval_arr[i_particle_update]
                > pbest_scores_arr[i_particle_update]
            ):
                pbest_scores_arr[i_particle_update] = current_scores_eval_arr[
                    i_particle_update
                ]
                pbest_positions_arr[i_particle_update] = particles_arr[
                    i_particle_update
                ].copy()
            if pbest_scores_arr[i_particle_update] > best_score_in_iter_val:
                best_score_in_iter_val = pbest_scores_arr[i_particle_update]
                best_idx_in_iter_val = i_particle_update
        if best_idx_in_iter_val != -1:
            gbest_score_val = best_score_in_iter_val
            gbest_position_arr = pbest_positions_arr[best_idx_in_iter_val].copy()
            gbest_changed_this_iter_flag = True

        pbest_scores_history_list.append(pbest_scores_arr.copy())

        if iteration_num > 0 and iteration_num % callback_interval == 0:
            if gbest_changed_this_iter_flag:
                if initial_canvas is not None:
                    gbest_image_u8_render = initial_canvas.copy()
                else:
                    gbest_image_u8_render = np.full(
                        (h_i32, w_i32, 3), 255, dtype=np.uint8
                    )
                _render_shapes_rgb(
                    gbest_image_u8_render,
                    gbest_position_arr,
                    shape_type,
                    param_names,
                    offsets_render_tuple,
                    shape_params_include_rgb,
                    shape_params_include_alpha,
                )
            yield gbest_image_u8_render.copy(), np.float64(
                gbest_score_val
            ), gbest_position_arr.copy(), pbest_scores_history_list

    if gbest_changed_this_iter_flag or iteration_num % callback_interval != 0:
        if initial_canvas is not None:
            gbest_image_u8_render = initial_canvas.copy()
        else:
            gbest_image_u8_render = np.full((h_i32, w_i32, 3), 255, dtype=np.uint8)
        _render_shapes_rgb(
            gbest_image_u8_render,
            gbest_position_arr,
            shape_type,
            param_names,
            offsets_render_tuple,
            shape_params_include_rgb,
            shape_params_include_alpha,
        )
    yield gbest_image_u8_render.copy(), np.float64(
        gbest_score_val
    ), gbest_position_arr.copy(), pbest_scores_history_list


def simulated_annealing(
    initial_shapes,
    shape_type,
    image_shape,
    target_image,
    metric_names,
    metric_weights,
    param_names,
    param_mins,
    param_maxs,
    fixed_params_specific_dict: NumbaDict,
    coord_indices_to_freeze_arr: np.ndarray,
    init_temp,
    cooling_rate,
    iterations,
    callback_interval,
    shape_params_include_rgb,
    shape_params_include_alpha,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
):
    """
    Wrapper for Simulated Annealing optimizer

    Prepares Numba-compatible structures for fixed parameters and calls
    the appropriate grayscale or RGB core SA function

    :param initial_shapes: Initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: Shape type string
    :type shape_type: str
    :param image_shape: (height, width) of target image
    :type image_shape: Tuple[int, int]
    :param target_image: Target image NumPy array (uint8)
    :type target_image: np.ndarray
    :param metric_names: NumbaList of metric names
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of metric weights
    :type metric_weights: np.ndarray
    :param param_names: NumbaList of config parameter names
    :type param_names: NumbaList[str]
    :param param_mins: Min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: Max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_params_specific_dict: NumbaDict mapping param name to specific fixed value
    :type fixed_params_specific_dict: NumbaDict
    :param coord_indices_to_freeze_arr: NumPy array of coordinate indices to freeze to initial values
    :type coord_indices_to_freeze_arr: np.ndarray
    :param init_temp: Initial SA temperature
    :type init_temp: float
    :param cooling_rate: SA cooling rate
    :type cooling_rate: float
    :param iterations: Number of iterations
    :type iterations: int
    :param callback_interval: Interval for yielding results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape colors are RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have alpha
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered canvas
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: Stop flag array
    :type stop_flag: np.ndarray
    :return: Generator yielding optimization results
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]
    """
    param_index_map_dict = _get_param_index_map(
        shape_type, param_names, shape_params_include_rgb, shape_params_include_alpha
    )
    fixed_indices_specific_list = []
    fixed_values_specific_list = []
    if fixed_params_specific_dict:
        for name, value in fixed_params_specific_dict.items():
            if name in param_index_map_dict:
                fixed_indices_specific_list.append(param_index_map_dict[name])
                fixed_values_specific_list.append(np.float32(value))
    fixed_indices_specific_arr_np = np.array(
        fixed_indices_specific_list, dtype=np.int64
    )
    fixed_values_specific_arr_np = np.array(
        fixed_values_specific_list, dtype=np.float32
    )

    target_u8_img = target_image
    is_target_grayscale_bool = target_u8_img.ndim == 2

    if is_target_grayscale_bool:
        yield from _simulated_annealing_grayscale(
            initial_shapes,
            shape_type,
            image_shape,
            target_u8_img,
            metric_names,
            metric_weights,
            param_names,
            param_mins,
            param_maxs,
            fixed_indices_specific_arr_np,
            fixed_values_specific_arr_np,
            coord_indices_to_freeze_arr,
            init_temp,
            cooling_rate,
            iterations,
            callback_interval,
            shape_params_include_rgb,
            shape_params_include_alpha,
            initial_canvas,
            stop_flag,
        )
    else:
        yield from _simulated_annealing_rgb(
            initial_shapes,
            shape_type,
            image_shape,
            target_u8_img,
            metric_names,
            metric_weights,
            param_names,
            param_mins,
            param_maxs,
            fixed_indices_specific_arr_np,
            fixed_values_specific_arr_np,
            coord_indices_to_freeze_arr,
            init_temp,
            cooling_rate,
            iterations,
            callback_interval,
            shape_params_include_rgb,
            shape_params_include_alpha,
            initial_canvas,
            stop_flag,
        )


def pso(
    initial_shapes,
    shape_type,
    image_shape,
    target_image,
    metric_names,
    metric_weights,
    param_names,
    param_mins,
    param_maxs,
    fixed_params_specific_dict: NumbaDict,
    coord_indices_to_freeze_arr: np.ndarray,
    swarm_size,
    cognitive_param,
    social_param,
    inertia_weight,
    iterations,
    callback_interval,
    shape_params_include_rgb,
    shape_params_include_alpha,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
):
    """
    Wrapper for Particle Swarm Optimizer

    Prepares Numba-compatible args and calls appropriate grayscale or RGB core PSO function

    :param initial_shapes: Initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: Shape type string
    :type shape_type: str
    :param image_shape: (height, width) of target image
    :type image_shape: Tuple[int, int]
    :param target_image: Target image NumPy array (uint8)
    :type target_image: np.ndarray
    :param metric_names: NumbaList of metric names
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of metric weights
    :type metric_weights: np.ndarray
    :param param_names: NumbaList of config parameter names
    :type param_names: NumbaList[str]
    :param param_mins: Min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: Max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_params_specific_dict: NumbaDict mapping param name to specific fixed value
    :type fixed_params_specific_dict: NumbaDict
    :param coord_indices_to_freeze_arr: NumPy array of coordinate indices to freeze
    :type coord_indices_to_freeze_arr: np.ndarray
    :param swarm_size: Number of particles
    :type swarm_size: int
    :param cognitive_param: PSO c1 parameter
    :type cognitive_param: float
    :param social_param: PSO c2 parameter
    :type social_param: float
    :param inertia_weight: PSO w parameter
    :type inertia_weight: float
    :param iterations: Number of iterations
    :type iterations: int
    :param callback_interval: Interval for yielding results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape colors are RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have alpha
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered canvas
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: Stop flag array
    :type stop_flag: np.ndarray
    :return: Generator yielding optimization results and PSO history
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray, NumbaList[np.ndarray]], None, None]
    """
    num_config_params_val = len(param_names)
    py_offsets_tuple = _get_param_offsets(
        shape_type,
        num_config_params_val,
        shape_params_include_rgb,
        shape_params_include_alpha,
    )
    param_index_map_dict = _get_param_index_map(
        shape_type, param_names, shape_params_include_rgb, shape_params_include_alpha
    )
    fixed_indices_specific_list = []
    fixed_values_specific_list = []
    if fixed_params_specific_dict:
        for name, value in fixed_params_specific_dict.items():
            if name in param_index_map_dict:
                fixed_indices_specific_list.append(param_index_map_dict[name])
                fixed_values_specific_list.append(np.float32(value))
    fixed_indices_specific_arr_np = np.array(
        fixed_indices_specific_list, dtype=np.int64
    )
    fixed_values_specific_arr_np = np.array(
        fixed_values_specific_list, dtype=np.float32
    )

    target_u8_img = target_image
    is_target_grayscale_bool = target_u8_img.ndim == 2

    if is_target_grayscale_bool:
        yield from _pso_grayscale(
            initial_shapes,
            shape_type,
            image_shape,
            target_u8_img,
            metric_names,
            metric_weights,
            param_names,
            param_mins,
            param_maxs,
            fixed_indices_specific_arr_np,
            fixed_values_specific_arr_np,
            coord_indices_to_freeze_arr,
            swarm_size,
            cognitive_param,
            social_param,
            inertia_weight,
            iterations,
            callback_interval,
            shape_params_include_rgb,
            shape_params_include_alpha,
            py_offsets_tuple,
            initial_canvas,
            stop_flag,
        )
    else:
        yield from _pso_rgb(
            initial_shapes,
            shape_type,
            image_shape,
            target_u8_img,
            metric_names,
            metric_weights,
            param_names,
            param_mins,
            param_maxs,
            fixed_indices_specific_arr_np,
            fixed_values_specific_arr_np,
            coord_indices_to_freeze_arr,
            swarm_size,
            cognitive_param,
            social_param,
            inertia_weight,
            iterations,
            callback_interval,
            shape_params_include_rgb,
            shape_params_include_alpha,
            py_offsets_tuple,
            initial_canvas,
            stop_flag,
        )


def hill_climbing(
    initial_shapes,
    shape_type,
    image_shape,
    target_image,
    metric_names,
    metric_weights,
    param_names,
    param_mins,
    param_maxs,
    fixed_params_specific_dict: NumbaDict,
    coord_indices_to_freeze_arr: np.ndarray,
    iterations,
    callback_interval,
    shape_params_include_rgb,
    shape_params_include_alpha,
    initial_canvas: Optional[np.ndarray],
    stop_flag: np.ndarray,
):
    """
    Wrapper for Hill Climbing optimizer

    Prepares Numba-compatible args and calls appropriate grayscale or RGB core HC function

    :param initial_shapes: Initial shape parameters
    :type initial_shapes: np.ndarray
    :param shape_type: Shape type string
    :type shape_type: str
    :param image_shape: (height, width) of target image
    :type image_shape: Tuple[int, int]
    :param target_image: Target image NumPy array (uint8)
    :type target_image: np.ndarray
    :param metric_names: NumbaList of metric names
    :type metric_names: NumbaList[str]
    :param metric_weights: NumPy array of metric weights
    :type metric_weights: np.ndarray
    :param param_names: NumbaList of config parameter names
    :type param_names: NumbaList[str]
    :param param_mins: Min bounds for config parameters
    :type param_mins: np.ndarray
    :param param_maxs: Max bounds for config parameters
    :type param_maxs: np.ndarray
    :param fixed_params_specific_dict: NumbaDict mapping param name to specific fixed value
    :type fixed_params_specific_dict: NumbaDict
    :param coord_indices_to_freeze_arr: NumPy array of coordinate indices to freeze
    :type coord_indices_to_freeze_arr: np.ndarray
    :param iterations: Number of iterations
    :type iterations: int
    :param callback_interval: Interval for yielding results
    :type callback_interval: int
    :param shape_params_include_rgb: True if shape colors are RGB
    :type shape_params_include_rgb: bool
    :param shape_params_include_alpha: True if shapes have alpha
    :type shape_params_include_alpha: bool
    :param initial_canvas: Optional prerendered canvas
    :type initial_canvas: Optional[np.ndarray]
    :param stop_flag: Stop flag array
    :type stop_flag: np.ndarray
    :return: Generator yielding optimization results
    :rtype: Generator[Tuple[np.ndarray, np.float64, np.ndarray], None, None]
    """
    param_index_map_dict = _get_param_index_map(
        shape_type, param_names, shape_params_include_rgb, shape_params_include_alpha
    )
    fixed_indices_specific_list = []
    fixed_values_specific_list = []
    if fixed_params_specific_dict:
        for name, value in fixed_params_specific_dict.items():
            if name in param_index_map_dict:
                fixed_indices_specific_list.append(param_index_map_dict[name])
                fixed_values_specific_list.append(np.float32(value))
    fixed_indices_specific_arr_np = np.array(
        fixed_indices_specific_list, dtype=np.int64
    )
    fixed_values_specific_arr_np = np.array(
        fixed_values_specific_list, dtype=np.float32
    )

    target_u8_img = target_image
    is_target_grayscale_bool = target_u8_img.ndim == 2

    if is_target_grayscale_bool:
        yield from _hill_climbing_grayscale(
            initial_shapes,
            shape_type,
            image_shape,
            target_u8_img,
            metric_names,
            metric_weights,
            param_names,
            param_mins,
            param_maxs,
            fixed_indices_specific_arr_np,
            fixed_values_specific_arr_np,
            coord_indices_to_freeze_arr,
            iterations,
            callback_interval,
            shape_params_include_rgb,
            shape_params_include_alpha,
            initial_canvas,
            stop_flag,
        )
    else:
        yield from _hill_climbing_rgb(
            initial_shapes,
            shape_type,
            image_shape,
            target_u8_img,
            metric_names,
            metric_weights,
            param_names,
            param_mins,
            param_maxs,
            fixed_indices_specific_arr_np,
            fixed_values_specific_arr_np,
            coord_indices_to_freeze_arr,
            iterations,
            callback_interval,
            shape_params_include_rgb,
            shape_params_include_alpha,
            initial_canvas,
            stop_flag,
        )
