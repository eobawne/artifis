import sys
import os
import datetime
import numpy as np
import traceback

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThreadPool
from numba.typed import List as NumbaList, Dict as NumbaDict
from numba.core import types as numba_types

from utils.config_loader import ConfigLoader
import core.image_utils as image_utils
import core.shapes as shapes
import core.borders as borders
import core.optimizers as optimizers
import core.metrics as metrics
import core.gif_creator as gif_creator
from utils.shape_io import (
    save_results_and_shapes,
    load_results_and_shapes,
)
from typing import Dict, Any, Callable, Generator, Tuple, List, Optional


class App:
    """
    Main application class for Artifis.

    Manages the overall application flow, including GUI and CLI operations,
    data preparation, running the approximation algorithms, and handling results.
    """

    def __init__(self):
        """
        Initializes the App class.

        Sets up available metrics, a thread pool for background tasks,
        and initial states for error messages, metrics data, and a stop flag
        for optimization processes.
        """
        self.available_metrics = metrics.AVAILABLE_METRICS
        self.threadpool = QThreadPool()
        self.error_message = ""
        self.best_metrics_data: List[float] = []
        self.pso_pbest_scores_history: Optional[List[np.ndarray]] = None
        self.final_metrics_orig: Dict[str, Any] = {}
        # stop_flag is used to signal optimization loops to terminate early if requested by the user.
        # 0 = run, 1 = stop requested. using numpy array for numba compatibility.
        self.stop_flag = np.array([0], dtype=np.int8)

    def run_gui(self):
        """
        Runs the application in Graphical User Interface (GUI) mode.

        Initializes and shows the main UI window.
        """
        from gui.main_window import ImageApproximationUI

        # this import is done here to avoid importing qt if only cli is used

        self.app = QApplication(sys.argv)
        self.main_window = ImageApproximationUI(self)
        self.main_window.show()
        sys.exit(self.app.exec())

    def run_cli(self, args):
        """
        Runs the application in Command Line Interface (CLI) mode.

        :param args: Parsed command line arguments.
        :type args: argparse.Namespace
        """
        params = self.collect_parameters_cli(args)
        if params is None:
            print("CLI parameter collection failed.")
            return
        self.config_loader = ConfigLoader()
        # cli run doesn't use progress emitter (prints to console) or gui stop flag interaction directly
        result_tuple = self.run_approximation(params, progress_emitter=None)
        final_image = (
            result_tuple[0] if result_tuple and len(result_tuple) > 0 else None
        )
        if final_image is None and self.error_message:
            print(f"Error during CLI run: {self.error_message}")
        elif final_image is not None:
            print("CLI run finished successfully.")
        else:
            print(
                "CLI run finished, but no final image was produced (check logs/errors)."
            )

    def collect_parameters_cli(self, args) -> Optional[Dict[str, Any]]:
        """
        Collects and structures parameters from parsed CLI arguments for the approximation process.

        This includes handling shape loading from files, validating loaded data,
        and setting up optimization method specific parameters.

        :param args: Parsed command line arguments from `argparse`.
        :type args: argparse.Namespace
        :return: A dictionary of parameters for the approximation run, or None if an error occurs.
        :rtype: Optional[Dict[str, Any]]
        """
        self.error_message = ""
        try:
            args_dict = vars(args)

            # initialize variables for loaded shapes
            loaded_shapes_data = (
                None  # will hold raw data from json if shapes are loaded
            )
            loaded_shapes_array = None  # will hold numpy array of shapes if loaded
            loaded_param_names = (
                []
            )  # parameter names corresponding to loaded_shapes_array
            use_loaded_shapes = (
                False  # flag to indicate if loaded shapes should be used
            )
            num_loaded_n = 0  # number of shapes actually loaded from file

            if args_dict.get("load_shapes"):
                json_path = args_dict["load_shapes"]
                print(f"Attempting to load shapes from: {json_path}")
                loaded_results_data, loaded_shapes_array_raw = load_results_and_shapes(
                    json_path
                )

                if loaded_results_data is None:
                    self.error_message = (
                        f"Failed to load or parse results json: {json_path}"
                    )
                    print(f"Error: {self.error_message}")
                    return None
                if loaded_shapes_array_raw is None:
                    self.error_message = f"No shape data found or loaded from associated .npy for: {json_path}"
                    print(f"Error: {self.error_message}")
                    return None

                # validate essential metadata from the loaded json
                required_meta = [
                    "shape_type",
                    "num_shapes",
                    "color_scheme",
                    "param_names_config",
                ]
                if not all(k in loaded_results_data for k in required_meta):
                    self.error_message = "Loaded JSON missing required metadata fields."
                    print(f"Error: {self.error_message}")
                    return None

                loaded_shape_type = loaded_results_data["shape_type"]
                loaded_num_shapes_meta = loaded_results_data["num_shapes"]
                loaded_color_scheme = loaded_results_data["color_scheme"]
                loaded_param_names_meta = loaded_results_data["param_names_config"]

                # validate consistency between loaded array and its metadata
                if (
                    loaded_shapes_array_raw.ndim != 2
                    or loaded_shapes_array_raw.shape[0] != loaded_num_shapes_meta
                ):
                    self.error_message = (
                        "Loaded shapes array dimensions mismatch metadata."
                    )
                    print(f"Error: {self.error_message}")
                    return None

                n_loaded, p_loaded = loaded_shapes_array_raw.shape
                # calculate expected number of parameters based on loaded metadata to ensure integrity
                expected_p = shapes.get_total_shape_params(
                    loaded_shape_type,
                    loaded_param_names_meta,  # use param names from loaded file
                    loaded_color_scheme in ("rgb", "both"),
                    True,  # assume alpha is always included in shape data structure
                )
                if p_loaded != expected_p:
                    self.error_message = f"Loaded array params ({p_loaded}) mismatch expected ({expected_p})."
                    print(f"Error: {self.error_message}")
                    return None

                print(
                    f"Successfully loaded {n_loaded} '{loaded_shape_type}' shapes ({loaded_color_scheme}). Overriding related CLI args."
                )
                # override cli args with values from the loaded file for consistency
                args_dict["shape"] = loaded_shape_type
                args_dict["color"] = loaded_color_scheme
                num_loaded_n = (
                    n_loaded  # store the number of shapes loaded from the file
                )

                # if user didn't specify --num-shapes, use the count from the loaded file
                if args_dict.get("num_shapes") is None:
                    args_dict["num_shapes"] = n_loaded
                    print(f" Using loaded number of shapes: {n_loaded}")
                else:
                    # if user did specify --num-shapes, it might differ from loaded count, which is handled later
                    print(
                        f" User specified --num-shapes ({args_dict['num_shapes']}), potentially overriding loaded count ({n_loaded})."
                    )

                use_loaded_shapes = True
                loaded_shapes_array = loaded_shapes_array_raw.astype(
                    np.float32
                )  # ensure float32
                loaded_param_names = (
                    loaded_param_names_meta  # use param names config from loaded file
                )

            # construct the main parameters dictionary
            params = {
                "input_image_path": args_dict["input"],
                "output_directory": args_dict["output"],
                "shape_type": args_dict["shape"],
                "color_scheme": args_dict["color"],
                "border_type": args_dict["border"],
                "optimization_method": args_dict["method"],
                "evaluation_metrics": args_dict["metric"],
                "metric_weights": args_dict["weights"],
                "num_shapes": args_dict[
                    "num_shapes"
                ],  # this might be from user or loaded file
                "gif_options": args_dict["gif"],
                "method_params": {},  # for method specific hyperparams like temp, swarm_size
                "param_init_type": args_dict.get("param_init", "midpoint"),
                "coord_mode": args_dict.get("coord_mode", "random"),
                "coord_fix_details": args_dict.get(
                    "coord_fix_details", {}
                ),  # from parser logic
                "fixed_params_non_coord": args_dict.get("fix_params_non_coord", []),
                "fixed_values_non_coord": args_dict.get("fixed_values_non_coord", []),
                "save_results_file": args_dict.get("save_results", False),
                "save_shape_data": args_dict.get(
                    "save_shapes_flag", False
                ),  # corrected name from parser
                "save_optimized_only": args_dict.get("save_optimized_only", False),
                "use_loaded_shapes": use_loaded_shapes,
                "loaded_shapes_array": loaded_shapes_array,
                "loaded_param_names_config": loaded_param_names,
                "render_loaded_as_canvas": args_dict.get("load_render_canvas", False),
                "truncation_mode": args_dict.get("load_truncate_mode", "first"),
                "render_untruncated_shapes": args_dict.get(
                    "load_render_untruncated", False
                ),
            }

            # --save-optimized-only is only meaningful if shapes are loaded
            if params["save_optimized_only"]:
                if not params["use_loaded_shapes"]:
                    print(
                        "Warning: --save-optimized-only ignored as shapes are not loaded."
                    )
                    params["save_optimized_only"] = False

            # map for method specific arguments from cli
            method_args_map = {
                "sa": ["init_temp", "cooling_rate", "iterations"],
                "pso": [
                    "swarm_size",
                    "cognitive_param",
                    "social_param",
                    "inertia_weight",
                    "iterations",
                ],
                "hc": ["iterations"],
            }
            # populate method_params based on the chosen optimization method
            if params["optimization_method"] in method_args_map:
                for param_name in method_args_map[params["optimization_method"]]:
                    if param_name in args_dict and args_dict[param_name] is not None:
                        params["method_params"][param_name] = args_dict[param_name]

            # ensure 'iterations' is present if the method requires it (it's common)
            if params["optimization_method"] in ["sa", "pso", "hc"]:
                if "iterations" not in params["method_params"]:
                    # this case should ideally be handled by parser defaults, but double check
                    if (
                        "iterations" in args_dict
                        and args_dict["iterations"] is not None
                    ):
                        params["method_params"]["iterations"] = args_dict["iterations"]
                    else:
                        # this indicates an internal issue if parser defaults didn't cover it
                        self.error_message = f"Internal Error: Missing 'iterations' parameter for method '{params['optimization_method']}'."
                        print(f"Error: {self.error_message}")
                        return None

            # validate consistency for fixed non-coordinate parameters
            if len(params["fixed_params_non_coord"]) != len(
                params["fixed_values_non_coord"]
            ):
                self.error_message = (
                    "Error: Mismatch between fixed parameter names and values count."
                )
                print(f"Error: {self.error_message}")
                return None
            if not isinstance(params["coord_fix_details"], dict):
                # coord_fix_details should always be a dict from the parser
                self.error_message = (
                    "Internal Error: Coordinate fixing details should be a dictionary."
                )
                print(f"Error: {self.error_message}")
                return None

            # logging collected parameters for user feedback
            if params["use_loaded_shapes"]:
                print(
                    f" Using {num_loaded_n} loaded shapes from {args_dict['load_shapes']}"
                )
                print(
                    f" Target Num Shapes (K): {params['num_shapes']}"
                )  # K is the target number of shapes for current run
                if params["render_loaded_as_canvas"]:
                    print("  Mode: Render Loaded as Canvas")
                else:
                    k_gui = params["num_shapes"]  # target K for optimization
                    # N is num_loaded_n, the number of shapes actually loaded
                    if k_gui < num_loaded_n:
                        print(f"  Mode: Optimize Subset (K={k_gui} < N={num_loaded_n})")
                        print(f"   Truncation: {params['truncation_mode']}")
                        print(
                            f"   Render Untruncated: {params['render_untruncated_shapes']}"
                        )
                        # this specific save_optimized_only applies to how the *final combined* shape array is constructed for saving
                        # if K < N and this is true, only the K optimized shapes are saved
                        # if K < N and this is false, the N shapes (with K updated) are saved
                        print(
                            f"   Save Optimized Only (when K<N): {params['save_optimized_only']}"
                        )
                    elif k_gui > num_loaded_n:
                        print(
                            f"  Mode: Optimize Superset (K={k_gui} > N={num_loaded_n})"
                        )
                    else:  # k_gui == num_loaded_n
                        print(f"  Mode: Optimize Loaded (K=N={num_loaded_n})")
                # this is the general flag passed from cli, affecting how shapes_to_save is determined at the end
                print(
                    f"   Save Optimized Only (General Flag): {params['save_optimized_only']}"
                )
            else:
                print(
                    f" Collected CLI Params: Method={params['optimization_method']}, Shape={params['shape_type']}, N={params['num_shapes']}"
                )
                print(f"  Param Init: {params['param_init_type']}")

            print(f"  Coord Init: {params['coord_mode']}")
            print(
                f"  Coord Fixing: {params['coord_fix_details'] if params['coord_fix_details'] else 'Dynamic'}"
            )
            print(
                f"  Fixed Non-Coord: {dict(zip(params['fixed_params_non_coord'], params['fixed_values_non_coord'])) if params['fixed_params_non_coord'] else 'None'}"
            )

            return params

        except KeyError as e:
            self.error_message = f"Missing expected argument key: {e}"
            print(f"Error: {self.error_message}")
            traceback.print_exc()
            return None
        except Exception as e:
            self.error_message = (
                f"Unexpected error during CLI parameter collection: {e}"
            )
            print(f"Error: {self.error_message}")
            traceback.print_exc()
            return None

    def prepare_data(self, params: Dict[str, Any]):
        """
        Prepares all necessary data for the approximation run.

        This includes loading and processing the input image, applying borders,
        initializing or loading shapes, validating metrics, and preparing
        parameter bounds and fixed parameter structures for the optimizers.

        :param params: A dictionary of parameters collected from CLI or GUI.
        :type params: Dict[str, Any]
        :return: A tuple containing all prepared data items, or None if an error occurs.
                 The tuple elements are: (None, original_shape, initial_shapes, initial_canvas,
                 output_directory, params, config_param_names, param_mins_arr, param_maxs_arr,
                 metric_names, metric_weights, fixed_params_specific_dict,
                 coord_indices_to_freeze_list, is_target_grayscale, target_image_uint8,
                 shape_params_include_rgb, shape_params_include_alpha, coord_mode, full_param_index_map).
                 The first element is None as bordered_image is processed into target_image_uint8.
        :rtype: Optional[Tuple[Optional[np.ndarray], Tuple[int, int], Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any], List[str], Optional[np.ndarray], Optional[np.ndarray], List[str], List[float], Dict[str, float], List[int], bool, Optional[np.ndarray], bool, bool, str, Dict[str, int]]]
        """
        self.error_message = ""
        if not hasattr(self, "config_loader"):
            # config_loader should be initialized in run_cli or __init__ (if gui path considered)
            self.error_message = "Internal error: ConfigLoader not initialized."
            print(self.error_message)
            return None
        print("Preparing data...")
        try:
            input_image = image_utils.load_image(params["input_image_path"])
        except Exception as e:
            self.error_message = (
                f"Failed to load input image '{params['input_image_path']}': {e}"
            )
            print(self.error_message)
            return None
        try:
            border_type = params["border_type"]
            border_args = {}  # arguments for apply_border function
            if border_type == "b_circle":
                try:
                    # load border configuration to get default/initial padding
                    border_config = self.config_loader.load_border_config(border_type)
                    padding_param = next(
                        (
                            p
                            for p in border_config.get("parameters", [])
                            if p["name"] == "padding"
                        ),
                        None,
                    )
                    border_args["padding"] = (
                        int(padding_param["initial"])
                        if padding_param and "initial" in padding_param
                        else 10  # default padding if not found in config
                    )
                except FileNotFoundError:
                    print(
                        f"Warning: Border config file not found for '{border_type}'. Using default padding 10."
                    )
                    border_args["padding"] = 10
                except Exception as cfg_e:  # catch other config loading issues
                    print(
                        f"Warning: Failed loading border config for '{border_type}': {cfg_e}. Using default padding 10."
                    )
                    border_args["padding"] = 10
            bordered_image, original_shape = borders.apply_border(
                input_image, border_type, **border_args
            )
            print(f"Applied border '{border_type}'. New shape: {bordered_image.shape}")
        except Exception as e:
            self.error_message = (
                f"Failed to apply border '{params['border_type']}': {e}"
            )
            print(self.error_message)
            traceback.print_exc()
            return None
        output_directory = params["output_directory"]
        try:
            os.makedirs(output_directory, exist_ok=True)
        except OSError as e:
            self.error_message = (
                f"Failed to create output directory '{output_directory}': {e}"
            )
            print(self.error_message)
            return None

        # determine if the target image for optimization should be grayscale
        is_target_grayscale = params.get("color_scheme") == "grayscale"
        target_image_uint8: Optional[np.ndarray] = (
            None  # final target image for optimization
        )
        print(
            f" Preparing target image (is_target_grayscale={is_target_grayscale}) for color scheme: {params['color_scheme']}"
        )
        try:
            target_image: Optional[np.ndarray] = None
            if is_target_grayscale:
                if bordered_image.ndim == 3:  # if original was rgb/rgba
                    target_image = image_utils.convert_to_grayscale(bordered_image)
                elif bordered_image.ndim == 2:  # if original was already grayscale
                    target_image = bordered_image.copy()
                else:
                    raise ValueError(
                        f"Unsupported bordered image dimensions for grayscale target: {bordered_image.ndim}"
                    )
            else:  # target is rgb (for "rgb" or "both" color schemes)
                if (
                    bordered_image.ndim == 3 and bordered_image.shape[2] == 3
                ):  # already rgb
                    target_image = bordered_image.copy()
                elif (
                    bordered_image.ndim == 3 and bordered_image.shape[2] == 4
                ):  # rgba, drop alpha
                    target_image = bordered_image[:, :, :3].copy()
                elif bordered_image.ndim == 2:  # grayscale, convert to rgb by stacking
                    target_image = np.stack((bordered_image,) * 3, axis=-1)
                else:
                    raise ValueError(
                        f"Unsupported bordered image dimensions for RGB target: {bordered_image.ndim}"
                    )
            # ensure target_image is uint8 [0, 255]
            if target_image.dtype != np.uint8:
                target_image_clipped = target_image
                # handle float images (from some internal processing)
                if np.issubdtype(target_image.dtype, np.floating):
                    temp_target = target_image * 255.0
                    temp_target[temp_target < 0] = 0
                    temp_target[temp_target > 255] = 255
                    target_image_clipped = temp_target
                # handle other integer types (uint16)
                elif (
                    np.issubdtype(target_image.dtype, np.integer)
                    and target_image.dtype != np.uint8
                ):
                    temp_target = target_image.copy()
                    temp_target[temp_target < 0] = 0
                    temp_target[temp_target > 255] = 255
                    target_image_clipped = temp_target
                target_image_uint8 = target_image_clipped.astype(np.uint8)
            else:
                target_image_uint8 = target_image.copy()
        except Exception as e:
            self.error_message = f"Failed to prepare target image for color scheme '{params['color_scheme']}': {e}"
            print(self.error_message)
            traceback.print_exc()
            return None
        if target_image_uint8 is None:  # sanity check
            self.error_message = "Target image preparation failed unexpectedly."
            print(self.error_message)
            return None

        # initialize shapes, canvas, and parameter configurations
        initial_shapes: Optional[np.ndarray] = None
        initial_canvas: Optional[np.ndarray] = None  # canvas for prerendered shapes
        config_param_names: List[str] = (
            []
        )  # names of parameters from shape config (radius)
        param_mins_arr: Optional[np.ndarray] = None  # min bounds for config_param_names
        param_maxs_arr: Optional[np.ndarray] = None  # max bounds for config_param_names
        selected_indices_for_save: Optional[np.ndarray] = (
            None  # used when K < N loaded, for merging later
        )
        # determine if shape parameters should include rgb components and alpha
        shape_params_include_rgb = params["color_scheme"] in ("rgb", "both")
        shape_params_include_alpha = (
            True  # currently, alpha is always assumed in param structure
        )
        param_init_type = params.get("param_init_type", "midpoint").lower()

        if params["use_loaded_shapes"]:
            print("Using loaded shapes...")
            loaded_shapes_array = params["loaded_shapes_array"]  # numpy array of Nxp
            config_param_names = params[
                "loaded_param_names_config"
            ]  # param names from loaded json
            num_config_params = len(config_param_names)
            k_target = params[
                "num_shapes"
            ]  # target number of shapes for current run (K)
            n_loaded = loaded_shapes_array.shape[0]  # number of shapes loaded (N)
            try:
                # load shape config to get min/max bounds for loaded parameters
                # this is important if loaded shapes are further optimized or new shapes added
                shape_config = self.config_loader.load_shape_config(
                    params["shape_type"]
                )
                shape_parameters_config = shape_config.get("parameters", [])
                param_mins_dict, param_maxs_dict = {}, {}
                for p_conf in shape_parameters_config:
                    name = p_conf["name"]
                    min_val = p_conf.get("min", 0.0)
                    max_val = p_conf.get(
                        "max", 1.0 if p_conf.get("type") == "float32" else 255
                    )
                    if min_val > max_val:  # ensure min <= max
                        min_val, max_val = max_val, min_val
                    param_mins_dict[name] = min_val
                    param_maxs_dict[name] = max_val
                # create arrays for bounds based on the order in config_param_names (from loaded file)
                param_mins_arr = np.array(
                    [param_mins_dict.get(name, -np.inf) for name in config_param_names],
                    dtype=np.float32,
                )
                param_maxs_arr = np.array(
                    [param_maxs_dict.get(name, np.inf) for name in config_param_names],
                    dtype=np.float32,
                )
            except Exception as e:
                self.error_message = f"Failed to load shape config '{params['shape_type']}' for bounds during shape loading: {e}"
                print(f"Error: {self.error_message}")
                return None

            if params["render_loaded_as_canvas"]:
                print(f" Mode: Render all {n_loaded} loaded shapes as canvas.")
                canvas_shape = target_image_uint8.shape
                # determine if initial_canvas should be grayscale or rgb
                if (
                    is_target_grayscale
                ):  # this is_target_grayscale matches the optimization target
                    initial_canvas = np.full(
                        (canvas_shape[0], canvas_shape[1]), 255, dtype=np.uint8
                    )
                    render_func = shapes._render_shapes_grayscale
                else:
                    initial_canvas = np.full(
                        (canvas_shape[0], canvas_shape[1], 3), 255, dtype=np.uint8
                    )
                    render_func = shapes._render_shapes_rgb
                try:
                    offsets = shapes._get_param_offsets(
                        params["shape_type"],
                        num_config_params,
                        shape_params_include_rgb,
                        shape_params_include_alpha,
                    )
                    numba_param_names = NumbaList()
                    if config_param_names:  # convert python list to numba list
                        [numba_param_names.append(n) for n in config_param_names]
                    # render the loaded shapes onto the initial_canvas
                    render_func(
                        initial_canvas,
                        loaded_shapes_array,
                        params["shape_type"],
                        numba_param_names,
                        offsets,
                        shape_params_include_rgb,
                        shape_params_include_alpha,
                    )
                    print(" Successfully rendered loaded shapes to initial canvas.")
                except Exception as e:
                    self.error_message = f"Error rendering loaded shapes to canvas: {e}"
                    print(f"Error: {self.error_message}")
                    traceback.print_exc()
                    return None
                # then, initialize k_target *new* shapes to be optimized on top of this canvas
                print(f" Initializing {k_target} new shapes for optimization.")
                try:
                    initial_shapes = shapes.initialize_shapes(
                        k_target,  # number of new shapes to init
                        params["shape_type"],
                        target_image_uint8.shape[:2],
                        config_param_names,  # use config_param_names from loaded file for consistency
                        param_mins_arr,  # use bounds derived from loaded shape_type config
                        param_maxs_arr,
                        params[
                            "coord_mode"
                        ],  # use user specified coord_mode for new shapes
                        shape_params_include_rgb,
                        shape_params_include_alpha,
                        param_init_type,  # use user specified param_init for new shapes
                        initial_coords=None,  # new shapes don't have predefined coords
                    )
                except Exception as e:
                    self.error_message = (
                        f"Error initializing new shapes for canvas mode: {e}"
                    )
                    print(f"Error: {self.error_message}")
                    traceback.print_exc()
                    return None
            else:  # not rendering loaded as canvas; loaded shapes will be optimized or part of them
                initial_canvas = None  # no prerendered canvas from loaded shapes
                selected_indices_for_save = None  # for merging back if K < N
                if k_target < n_loaded:  # optimize a subset of loaded shapes
                    trunc_mode = params["truncation_mode"]
                    print(
                        f" Mode: Optimizing Subset K={k_target} < N={n_loaded}, Truncation='{trunc_mode}'"
                    )
                    indices = np.arange(n_loaded)
                    if trunc_mode == "random":
                        selected_indices_for_save = np.random.choice(
                            indices, k_target, replace=False
                        )
                    elif trunc_mode == "last":
                        selected_indices_for_save = indices[-k_target:]
                    else:  # "first" or default
                        selected_indices_for_save = indices[:k_target]
                    initial_shapes = loaded_shapes_array[selected_indices_for_save]

                    if params["render_untruncated_shapes"]:
                        print("  Rendering untruncated shapes to canvas...")
                        untruncated_indices = np.setdiff1d(
                            indices, selected_indices_for_save
                        )
                        if len(untruncated_indices) > 0:
                            shapes_to_render_on_canvas = loaded_shapes_array[
                                untruncated_indices
                            ]
                            canvas_shape = target_image_uint8.shape
                            if is_target_grayscale:
                                initial_canvas = np.full(
                                    (canvas_shape[0], canvas_shape[1]),
                                    255,
                                    dtype=np.uint8,
                                )
                                render_func = shapes._render_shapes_grayscale
                            else:
                                initial_canvas = np.full(
                                    (canvas_shape[0], canvas_shape[1], 3),
                                    255,
                                    dtype=np.uint8,
                                )
                                render_func = shapes._render_shapes_rgb
                            try:
                                offsets = shapes._get_param_offsets(
                                    params["shape_type"],
                                    num_config_params,
                                    shape_params_include_rgb,
                                    shape_params_include_alpha,
                                )
                                numba_param_names = NumbaList()
                                if config_param_names:
                                    [
                                        numba_param_names.append(n)
                                        for n in config_param_names
                                    ]
                                render_func(
                                    initial_canvas,
                                    shapes_to_render_on_canvas,  # render the untruncated ones
                                    params["shape_type"],
                                    numba_param_names,
                                    offsets,
                                    shape_params_include_rgb,
                                    shape_params_include_alpha,
                                )
                            except Exception as e:
                                self.error_message = (
                                    f"Error rendering untruncated shapes: {e}"
                                )
                                print(f"Error: {self.error_message}")
                                initial_canvas = None  # fallback if rendering fails
                elif k_target > n_loaded:  # optimize loaded shapes + some new ones
                    print(f" Mode: Optimizing Superset K={k_target} > N={n_loaded}")
                    num_new_shapes = k_target - n_loaded
                    try:
                        new_shapes = shapes.initialize_shapes(
                            num_new_shapes,
                            params["shape_type"],
                            target_image_uint8.shape[:2],
                            config_param_names,  # use loaded config names for new shapes too
                            param_mins_arr,  # and bounds
                            param_maxs_arr,
                            params["coord_mode"],  # user specified init for new ones
                            shape_params_include_rgb,
                            shape_params_include_alpha,
                            param_init_type,
                            initial_coords=None,
                        )
                        initial_shapes = np.vstack((loaded_shapes_array, new_shapes))
                    except Exception as e:
                        self.error_message = (
                            f"Error initializing shapes for superset mode: {e}"
                        )
                        print(f"Error: {self.error_message}")
                        traceback.print_exc()
                        return None
                else:  # k_target == n_loaded, optimize all loaded shapes
                    print(f" Mode: Optimizing exactly {n_loaded} loaded shapes.")
                    initial_shapes = loaded_shapes_array
                params["selected_indices_for_save"] = selected_indices_for_save
        else:  # not using loaded shapes, initialize all new shapes
            initial_canvas = None  # no prerendered canvas
            params["selected_indices_for_save"] = None  # not applicable
            print("Initializing new shapes...")
            try:
                shape_config = self.config_loader.load_shape_config(
                    params["shape_type"]
                )
            except Exception as e:
                self.error_message = (
                    f"Failed to load shape config for '{params['shape_type']}': {e}"
                )
                print(self.error_message)
                return None
            shape_parameters_config = shape_config.get("parameters", [])
            # these are the names of parameters specific to the shape type from its json config
            config_param_names = [p["name"] for p in shape_parameters_config]
            param_mins_dict = {}
            param_maxs_dict = {}
            print(f"Using parameter init mode: {param_init_type}")
            for p_conf in shape_parameters_config:
                name = p_conf["name"]
                p_type = p_conf.get("type", "float32")
                min_val = p_conf.get("min", 0.0)
                # default max depends on type (1.0 for float, 255 for int-like color)
                max_val = p_conf.get("max", 1.0 if p_type == "float32" else 255)
                if min_val > max_val:  # ensure min <= max
                    min_val, max_val = max_val, min_val
                param_mins_dict[name] = min_val
                param_maxs_dict[name] = max_val
            # create arrays for bounds based on the order in config_param_names
            param_mins_arr = np.array(
                [param_mins_dict.get(name, -np.inf) for name in config_param_names],
                dtype=np.float32,
            )
            param_maxs_arr = np.array(
                [param_maxs_dict.get(name, np.inf) for name in config_param_names],
                dtype=np.float32,
            )

            coord_init_type = params.get("coord_mode", "random").lower()
            num_shapes_to_init = params["num_shapes"]
            img_h, img_w = target_image_uint8.shape[:2]
            initial_coords_sampled: Optional[np.ndarray] = None  # for pdf-based init

            # handle PDF-based coordinate initialization if selected
            if coord_init_type in ("intensity_pdf", "ssim_pdf"):
                print(f"Calculating PDF for Coordinate Init Mode: {coord_init_type}...")
                pdf = None
                try:
                    if coord_init_type == "intensity_pdf":
                        grayscale_target = target_image_uint8
                        if (
                            target_image_uint8.ndim == 3
                        ):  # convert to grayscale if needed
                            grayscale_target = image_utils.convert_to_grayscale(
                                target_image_uint8
                            )
                        # darker areas get higher probability (255 - intensity)
                        density_map = 255.0 - grayscale_target.astype(np.float32)
                    elif coord_init_type == "ssim_pdf":
                        # ssim_pdf uses 1 - ssim(white_bg, target) as density
                        # areas with low ssim to white (i.e. dark or complex areas) get higher probability
                        white_bg = np.full_like(target_image_uint8, 255, dtype=np.uint8)
                        ssim_map_avg = np.zeros((img_h, img_w), dtype=np.float32)
                        if target_image_uint8.ndim == 2:  # grayscale target
                            ssim_map_avg = metrics._ssim_channel(
                                white_bg, target_image_uint8
                            )
                        elif target_image_uint8.ndim == 3:  # rgb target
                            num_channels = target_image_uint8.shape[2]
                            if num_channels == 3:
                                for c in range(num_channels):
                                    ssim_map_avg += metrics._ssim_channel(
                                        white_bg[:, :, c], target_image_uint8[:, :, c]
                                    )
                                ssim_map_avg /= np.float32(num_channels)
                            else:  # unsupported channel count for ssim, fallback
                                print(
                                    f"Warning: SSIM PDF fallback for {num_channels} channels."
                                )
                                ssim_map_avg = np.ones((img_h, img_w), dtype=np.float32)
                        else:  # unsupported dims for ssim, fallback
                            print(
                                f"Warning: SSIM PDF fallback for image dims {target_image_uint8.ndim}."
                            )
                            ssim_map_avg = np.ones((img_h, img_w), dtype=np.float32)
                        density_map = (
                            1.0 - ssim_map_avg
                        )  # 1-ssim gives higher values for dissimilar areas
                        density_map = np.maximum(
                            0.0, density_map
                        )  # ensure non-negative

                    map_sum = np.sum(density_map)
                    if map_sum > 1e-9:  # avoid division by zero if map is all zeros
                        pdf = density_map / map_sum
                    else:
                        print(
                            "Warning: PDF calculation resulted in near-zero sum. Falling back."
                        )
                        pdf = None
                except Exception as pdf_e:
                    print(f"Error calculating PDF: {pdf_e}. Falling back.")
                    pdf = None

                if pdf is not None:
                    try:
                        pdf_flat = pdf.ravel()
                        # sample indices from the flattened pdf
                        chosen_indices_1d = np.random.choice(
                            pdf_flat.size, size=num_shapes_to_init, p=pdf_flat
                        )
                        # convert 1d indices back to 2d coordinates (y,x)
                        coords_yx = np.unravel_index(chosen_indices_1d, pdf.shape)
                        initial_coords_sampled = np.stack(coords_yx, axis=-1).astype(
                            np.float32
                        )
                        initial_coords_sampled = initial_coords_sampled[
                            :, ::-1
                        ]  # convert (y,x) to (x,y)
                        print(
                            f"Sampled {len(initial_coords_sampled)} coordinates using {coord_init_type}."
                        )
                    except Exception as sample_e:
                        print(
                            f"Error sampling coordinates from PDF: {sample_e}. Falling back."
                        )
                        initial_coords_sampled = None
                # if pdf failed or sampling failed, fallback to random
                if initial_coords_sampled is None and coord_init_type in (
                    "intensity_pdf",
                    "ssim_pdf",
                ):
                    print(
                        f"Falling back to random coordinate initialization for {coord_init_type}."
                    )
                    coord_init_type = (
                        "random"  # update coord_init_type for initialize_shapes call
                    )

            try:
                initial_shapes = shapes.initialize_shapes(
                    num_shapes_to_init,
                    params["shape_type"],
                    target_image_uint8.shape[:2],
                    config_param_names,  # from shape's json config
                    param_mins_arr,  # bounds from shape's json config
                    param_maxs_arr,
                    coord_init_type,  # potentially updated if pdf init failed
                    shape_params_include_rgb,
                    shape_params_include_alpha,
                    param_init_type,
                    initial_coords=initial_coords_sampled,  # pass sampled coords if available
                )
            except ValueError as e:  # specific error from initialize_shapes
                self.error_message = f"Failed to initialize shapes: {e}"
                print(self.error_message)
                traceback.print_exc()
                return None
            except Exception as e:  # other unexpected errors
                self.error_message = (
                    f"Unexpected error during shape initialization: {e}"
                )
                print(self.error_message)
                traceback.print_exc()
                return None

        if initial_shapes is None:  # sanity check after all init paths
            self.error_message = "Shape initialization failed."
            print(f"Error: {self.error_message}")
            return None
        if param_mins_arr is None or param_maxs_arr is None:  # sanity check
            self.error_message = "Parameter bounds were not initialized."
            print(f"Error: {self.error_message}")
            return None

        # validate metrics and weights
        metric_names = params.get("evaluation_metrics", ["ssim"])
        metric_weights = params.get("metric_weights", [1.0])
        if len(metric_names) != len(metric_weights):
            self.error_message = "Internal Error: Metrics and weights count mismatch."
            print(self.error_message)
            return None
        if not np.isclose(sum(metric_weights), 1.0):
            weight_sum = sum(metric_weights)
            if weight_sum > 1e-6:  # avoid division by zero if sum is tiny
                print(f"Normalizing metric weights (sum was {weight_sum:.3f}).")
                metric_weights = [w / weight_sum for w in metric_weights]
            else:  # if sum is very small, use equal weights as fallback
                print("Warning: Metric weights sum near zero, using equal weights.")
                metric_weights = [1.0 / len(metric_names)] * len(metric_names)
        # ensure all selected metrics are actually available
        valid_metrics = [
            name for name in metric_names if name in metrics.AVAILABLE_METRICS
        ]
        if len(valid_metrics) != len(metric_names):
            invalid_metrics = [
                name for name in metric_names if name not in metrics.AVAILABLE_METRICS
            ]
            self.error_message = f"Invalid metric name(s) found: {invalid_metrics}"
            print(self.error_message)
            return None
        metric_names = valid_metrics  # use only the validated list
        print(
            f"Validated metrics: {metric_names} with weights: {[f'{w:.3f}' for w in metric_weights]}"
        )

        # prepare parameter index map and bounds map for fixed parameter handling
        temp_typed_list = NumbaList()  # for numba compatibility
        if (
            config_param_names
        ):  # config_param_names is now set either from loaded or new
            for name in config_param_names:
                temp_typed_list.append(name)
        try:
            full_param_index_map = optimizers._get_param_index_map(
                params["shape_type"],
                temp_typed_list,  # use the numba typed list
                shape_params_include_rgb,
                shape_params_include_alpha,
            )
        except Exception as e:
            self.error_message = f"Failed to get parameter index map: {e}"
            print(self.error_message)
            traceback.print_exc()
            return None

        param_bounds_map = (
            {}
        )  # stores (min, max) for each parameter name in full_param_index_map
        num_config_params_final = len(config_param_names)
        temp_coord_indices = shapes.get_coord_indices(  # get raw coord indices based on config param count
            params["shape_type"], num_config_params_final
        )
        # populate param_bounds_map using the full_param_index_map
        for name, index in full_param_index_map.items():
            try:
                min_b, max_b = optimizers.get_param_bounds(
                    int(index),  # index from full_param_index_map
                    num_config_params_final,  # count of only config-specific params
                    temp_coord_indices,  # raw coord indices
                    param_mins_arr,  # bounds for config-specific params
                    param_maxs_arr,
                    target_image_uint8.shape[:2],
                    shape_params_include_rgb,
                    shape_params_include_alpha,
                    params["shape_type"],
                )
                param_bounds_map[name] = (min_b, max_b)
            except IndexError:  # fallback if bounds calculation fails for some reason
                param_bounds_map[name] = (-np.inf, np.inf)
                print(f"Warning: Failed to get bounds for parameter '{name}'.")

        # process fixed parameters (non-coordinate and coordinate)
        fixed_params_specific_dict = (
            {}
        )  # final dict of {name: value} for specifically fixed params
        fixed_params_non_coord = params.get("fixed_params_non_coord", [])
        fixed_values_non_coord = params.get("fixed_values_non_coord", [])
        for i, name in enumerate(fixed_params_non_coord):
            if name not in full_param_index_map:
                self.error_message = f"Parameter '{name}' to be fixed is not valid for the current configuration."
                print(self.error_message)
                return None
            value = fixed_values_non_coord[i]
            min_b, max_b = param_bounds_map.get(name, (-np.inf, np.inf))
            if not (min_b <= value <= max_b):  # validate fixed value against its bounds
                self.error_message = f"Fixed value {value} for '{name}' is out of bounds [{min_b:.2f}, {max_b:.2f}]."
                print(self.error_message)
                return None
            fixed_params_specific_dict[name] = value

        coord_fix_details = params.get("coord_fix_details", {})
        for name, details in coord_fix_details.items():
            if (
                name not in full_param_index_map
            ):  # skip if coord name isn't in the map (x2 for circle)
                continue
            if (
                details.get("mode") == "specific"
            ):  # if coord is fixed to a specific value
                value = details.get("value")
                if value is None:  # should not happen if parser/gui logic is correct
                    self.error_message = f"Coordinate '{name}' selected for specific fixing is missing a value."
                    print(self.error_message)
                    return None
                min_b, max_b = param_bounds_map.get(name, (-np.inf, np.inf))
                if not (
                    min_b <= value <= max_b
                ):  # validate fixed value against its bounds
                    self.error_message = f"Fixed coordinate value {value} for '{name}' is out of bounds [{min_b:.2f}, {max_b:.2f}]."
                    print(self.error_message)
                    return None
                if (
                    name in fixed_params_specific_dict
                ):  # warn if already fixed via non-coord list
                    print(
                        f"Warning: Coordinate '{name}' was already fixed via non-coordinate list; specific coordinate fixing overrides."
                    )
                fixed_params_specific_dict[name] = (
                    value  # add/override in specific dict
                )

        # identify coordinate indices to be frozen to their initial values
        coord_indices_to_freeze_list = []  # list of integer indices
        axes_frozen_to_initial = []  # list of names for logging
        for name, details in coord_fix_details.items():
            if (
                details.get("mode") == "initial"  # if coord is fixed to initial value
                and name in full_param_index_map  # ensure it's a valid param
                and name
                not in fixed_params_specific_dict  # ensure not already fixed to specific value
            ):
                coord_indices_to_freeze_list.append(int(full_param_index_map[name]))
                axes_frozen_to_initial.append(name)

        print(
            f"Fixed Specific Params: {fixed_params_specific_dict if fixed_params_specific_dict else 'None'}"
        )
        if coord_indices_to_freeze_list:
            print(
                f"Coords Frozen to Initial: {axes_frozen_to_initial} (Indices: {coord_indices_to_freeze_list})"
            )
        else:
            print("Coordinates Frozen to Initial: None")

        print("Data preparation successful.")
        return (
            None,  # bordered_image is now target_image_uint8, so pass None for its original spot
            original_shape,
            initial_shapes,
            initial_canvas,
            output_directory,
            params,  # pass the potentially modified params dict back
            config_param_names,  # final list of config param names
            param_mins_arr,  # final bounds for config params
            param_maxs_arr,
            metric_names,  # validated list of metric names
            metric_weights,  # normalized list of weights
            fixed_params_specific_dict,  # {name: value} for params fixed to specific values
            coord_indices_to_freeze_list,  # list of indices for coords frozen to initial
            is_target_grayscale,  # boolean indicating if optimization target is grayscale
            target_image_uint8,  # the actual target image for optimization
            shape_params_include_rgb,  # boolean
            shape_params_include_alpha,  # boolean
            params["coord_mode"],  # final coord_mode used (could be fallback from pdf)
            full_param_index_map,  # map of all param names to their flat indices
        )

    def run_approximation(
        self, params: Dict[str, Any], progress_emitter: Optional[Callable] = None
    ) -> Optional[Tuple[Optional[np.ndarray], List[float], Optional[np.ndarray]]]:
        """
        Runs the core image approximation algorithm.

        This method orchestrates the optimization process, including data preparation,
        invoking the selected optimization algorithm, handling results, and saving outputs.

        :param params: Dictionary of parameters for the run.
        :type params: Dict[str, Any]
        :param progress_emitter: Optional callable to emit progress updates (used by GUI).
        :type progress_emitter: Optional[Callable]
        :return: A tuple containing the final best image, a history of best metric scores,
                 and the final best shapes array, or None if a major error occurs.
        :rtype: Optional[Tuple[Optional[np.ndarray], List[float], Optional[np.ndarray]]]
        """
        self.error_message = ""
        metrics_history: List[float] = []  # history of the best combined metric score
        self.pso_pbest_scores_history = None  # for pso-specific plotting data
        self.final_metrics_orig = (
            {}
        )  # stores final metrics calculated on original scale
        start_run_time = datetime.datetime.now()
        self.stop_flag[0] = 0  # ensure stop flag is reset at the beginning of a run

        prepared_data = self.prepare_data(params)
        if prepared_data is None:
            # error_message should be set by prepare_data
            print(f"Data prep failed: {self.error_message}")
            return None

        # unpack prepared data
        (
            _,  # placeholder for original bordered_image
            original_shape,
            initial_shapes,
            initial_canvas,  # may be None or a prerendered canvas
            output_directory,
            passed_params,  # params dict, potentially updated by prepare_data
            config_param_names,  # list of names for shape-specific config params
            param_mins_arr,  # min bounds for config_param_names
            param_maxs_arr,  # max bounds for config_param_names
            metric_names,  # list of metric names to use
            metric_weights,  # list of corresponding weights
            fixed_params_specific_dict,  # dict of {name:value} for params fixed to a value
            coord_indices_to_freeze_list,  # list of indices for coords frozen to initial values
            _,  # is_target_grayscale, not directly used here, derived from target_image.ndim
            target_image,  # the uint8 target image for optimization (grayscale or rgb)
            shape_params_include_rgb,  # bool
            shape_params_include_alpha,  # bool
            coord_init_type,  # coord init mode used (could be fallback)
            full_param_index_map,  # map of all param names to indices
        ) = prepared_data

        # determine if optimization should run on grayscale image based on color_scheme
        # this is different from is_target_grayscale from prepare_data if color_scheme is 'both'
        # for 'both', target_image is RGB, but optimization might effectively be on grayscale logic if shapes don't have color
        run_optimization_as_grayscale = passed_params["color_scheme"] == "grayscale"
        shape_type = passed_params["shape_type"]
        method_params = passed_params["method_params"]  # temp, swarm_size
        opt_method = passed_params["optimization_method"]
        iterations = int(method_params.get("iterations", 1000))  # ensure it's an int
        gif_images = []  # list to store images for gif creation
        optimizer_generator = None  # will hold the generator from the chosen optimizer
        best_final_image: Optional[np.ndarray] = None
        best_final_shapes: Optional[np.ndarray] = None
        best_final_metric: np.float64 = np.float64(
            -np.inf
        )  # track best combined metric
        pso_history_from_gen: Optional[List[np.ndarray]] = None  # for pso data

        # convert python dicts/lists to numba-typed versions for numba function args
        fixed_params_specific_numba_dict = NumbaDict.empty(
            key_type=numba_types.unicode_type, value_type=numba_types.float32
        )
        if fixed_params_specific_dict:  # if there are any specific fixed params
            [
                fixed_params_specific_numba_dict.update({k: np.float32(v)})
                for k, v in fixed_params_specific_dict.items()
            ]
        coord_indices_to_freeze_arr = np.array(
            coord_indices_to_freeze_list, dtype=np.int64
        )
        config_param_names_typed = NumbaList.empty_list(numba_types.unicode_type)
        if config_param_names:
            [config_param_names_typed.append(n) for n in config_param_names]
        metric_names_typed = NumbaList.empty_list(numba_types.unicode_type)
        if metric_names:
            [metric_names_typed.append(n) for n in metric_names]
        metric_weights_arr = np.array(metric_weights, dtype=np.float32)

        try:
            # callback_interval determines how often the optimizer yields intermediate results
            callback_interval = max(1, iterations // 100) if iterations > 0 else 1
            print("\nStarting Optimization")
            print(f" Method: {opt_method.upper()}, Iterations: {iterations}")
            print(
                f" Running Optimization As Grayscale: {run_optimization_as_grayscale}"
            )
            if initial_canvas is not None:
                print(
                    f" Using prerendered initial canvas (shape: {initial_canvas.shape})"
                )
            print(
                f" Fixed Specific Params: {fixed_params_specific_dict if fixed_params_specific_dict else 'None'}"
            )
            print(
                f" Indices Frozen to Initial: {coord_indices_to_freeze_list if coord_indices_to_freeze_list else 'None'}"
            )
            print(
                f" Metrics: {metric_names}, Weights: {[f'{w:.3f}' for w in metric_weights]}"
            )
            print(f" Callback Interval: {callback_interval}")
            print("-----------------------------")

            # common arguments for all optimizer functions
            common_args = {
                "initial_shapes": initial_shapes,
                "shape_type": shape_type,
                "image_shape": target_image.shape[:2],  # (height, width)
                "target_image": target_image,  # the uint8 image to match
                "metric_names": metric_names_typed,
                "metric_weights": metric_weights_arr,
                "param_names": config_param_names_typed,  # names from shape's json config
                "param_mins": param_mins_arr,  # bounds for config_param_names
                "param_maxs": param_maxs_arr,
                "fixed_params_specific_dict": fixed_params_specific_numba_dict,  # numba dict for specific fixed values
                "coord_indices_to_freeze_arr": coord_indices_to_freeze_arr,  # np array of indices
                "iterations": iterations,
                "callback_interval": callback_interval,
                "shape_params_include_rgb": shape_params_include_rgb,
                "shape_params_include_alpha": shape_params_include_alpha,
                "initial_canvas": initial_canvas,  # pass the prerendered canvas if it exists
                "stop_flag": self.stop_flag,  # pass the stop flag array
            }

            # select and call the appropriate optimizer
            if opt_method == "sa":
                sa_params = {
                    "init_temp": float(method_params["init_temp"]),
                    "cooling_rate": float(method_params["cooling_rate"]),
                }
                optimizer_func = optimizers.simulated_annealing
                optimizer_generator = optimizer_func(**common_args, **sa_params)
            elif opt_method == "pso":
                pso_params = {
                    "swarm_size": int(method_params["swarm_size"]),
                    "cognitive_param": float(method_params["cognitive_param"]),
                    "social_param": float(method_params["social_param"]),
                    "inertia_weight": float(method_params["inertia_weight"]),
                }
                optimizer_func = optimizers.pso
                optimizer_generator = optimizer_func(**common_args, **pso_params)
            elif opt_method == "hc":
                optimizer_func = optimizers.hill_climbing
                optimizer_generator = optimizer_func(
                    **common_args
                )  # hc might only need common_args
            else:
                self.error_message = (
                    f"Invalid optimization method specified: {opt_method}"
                )
                print(f"\nError: {self.error_message}")
                return None

            if optimizer_generator:
                start_loop_time = datetime.datetime.now()
                last_img_yielded: Optional[np.ndarray] = (
                    None  # store last valid image for fallback
                )
                last_metric_yielded: np.float64 = np.float64(-np.inf)
                last_shapes_yielded: Optional[np.ndarray] = None
                yield_count = 0

                if self.stop_flag[0] == 1:  # check stop flag before starting loop
                    print("\nStop requested before optimization loop started.")
                    return None

                # main optimization loop, iterating through yields from the optimizer
                for i, result_yield in enumerate(optimizer_generator):
                    # the optimizer's numba loop should internally check self.stop_flag
                    if (
                        result_yield is None
                    ):  # should not happen with current optimizers
                        print("Warning: Optimizer yielded None unexpectedly.")
                        continue

                    # unpack yield: image, metric, shapes, and optional pso history
                    img, metric, current_best_shapes, *pso_hist_update = result_yield
                    if pso_hist_update:  # pso yields an extra item
                        pso_history_from_gen = pso_hist_update[0]
                    else:
                        pso_history_from_gen = None  # for sa, hc

                    # handle metric value, ensuring it's valid
                    if metric is None or not np.isfinite(metric):
                        metric_float64 = (
                            last_metric_yielded  # use last valid if current is bad
                        )
                    else:
                        metric_float64 = np.float64(metric)
                        last_metric_yielded = metric_float64  # update last valid
                        if metric_float64 > best_final_metric:  # update overall best
                            best_final_metric = metric_float64
                        if img is not None:  # update best image if current is better
                            best_final_image = img.copy()
                        if current_best_shapes is not None:  # update best shapes
                            best_final_shapes = current_best_shapes.copy()
                        metrics_history.append(metric_float64)  # append to history

                    img_u8 = None  # for gif, ensure uint8
                    if img is not None:
                        img_u8 = img
                        if img.dtype != np.uint8:  # convert to uint8 if not already
                            if np.issubdtype(img.dtype, np.floating):
                                temp_img = img * 255.0
                                temp_img[temp_img < 0] = 0
                                temp_img[temp_img > 255] = 255
                                img_u8 = temp_img.astype(np.uint8)
                            elif np.issubdtype(img.dtype, np.integer):  # int32
                                temp_img = img.copy()
                                temp_img[temp_img < 0] = 0
                                temp_img[temp_img > 255] = 255
                                img_u8 = temp_img.astype(np.uint8)
                        gif_images.append(img_u8)  # add to gif frame list
                        last_img_yielded = img_u8  # store for fallback
                    if current_best_shapes is not None:
                        last_shapes_yielded = current_best_shapes  # store for fallback

                    yield_count += 1
                    current_time = datetime.datetime.now()
                    elapsed_time = current_time - start_loop_time
                    # progress calculation: actual steps are yield_count * callback_interval
                    progress_steps = yield_count * callback_interval
                    est_total_iterations = iterations  # total iterations for this run
                    progress_fraction = (
                        min(1.0, float(progress_steps) / float(est_total_iterations))
                        if est_total_iterations > 0
                        else 0.0
                    )
                    remaining_time = datetime.timedelta(
                        days=99
                    )  # init with large value
                    if (
                        progress_fraction > 1e-9
                    ):  # avoid division by zero for time estimation
                        total_estimated_time = elapsed_time / progress_fraction
                        remaining_time = max(
                            datetime.timedelta(0), total_estimated_time - elapsed_time
                        )
                    elapsed_sec = elapsed_time.total_seconds()
                    remaining_sec = remaining_time.total_seconds()
                    # format remaining time string
                    rem_str = "..."
                    if remaining_sec < np.inf:  # check if calculable
                        rem_secs = remaining_sec
                        if rem_secs > 7200:  # > 2 hours
                            rem_str = f"{rem_secs/3600:.1f}h"
                        elif rem_secs > 120:  # > 2 minutes
                            rem_str = f"{rem_secs/60:.1f}m"
                        elif rem_secs >= 0:
                            rem_str = f"{rem_secs:.0f}s"
                    print(
                        f" Step ~{progress_steps}/{est_total_iterations} ({progress_fraction*100:.1f}%), Best Metric: {best_final_metric:.5f}, Rem: {rem_str}, Elapsed: {str(elapsed_time).split('.')[0]}",
                        end="\r",  # overwrite previous line in console
                    )
                    if progress_emitter:  # if gui is used, emit signal
                        progress_emitter.emit(
                            int(progress_steps),
                            int(est_total_iterations),
                            float(progress_fraction * 100.0),
                            float(elapsed_sec),
                            float(remaining_sec),
                        )

                if self.stop_flag[0] == 1:  # if loop exited due to stop flag
                    print("\nOptimization stopped by user.")
                else:
                    print("\nOptimization loop finished naturally.")

                # ensure final bests are from last valid yield if loop finishes/stops
                if best_final_image is None:
                    best_final_image = last_img_yielded
                if best_final_shapes is None:
                    best_final_shapes = last_shapes_yielded
                if pso_history_from_gen:  # if pso, store its specific history
                    self.pso_pbest_scores_history = [
                        np.array(arr)
                        for arr in pso_history_from_gen  # convert list of numba arrays to list of numpy arrays
                    ]

        except (KeyError, ValueError, TypeError, IndexError) as e:  # expected errors
            self.error_message = f"Optimization loop failed: {e}"
            print(f"\nError: {self.error_message}")
            traceback.print_exc()
            return None, metrics_history, None  # return partial results if any
        except Exception as e:  # unexpected errors
            self.error_message = f"Unexpected optimization error: {e}"
            print(f"\nError: {self.error_message}")
            traceback.print_exc()
            return None, metrics_history, None
        finally:
            # always ensure stop flag is reset for subsequent runs (in gui)
            self.stop_flag[0] = 0
            print("")  # ensure newline after progress printing

        # process results even if stopped or minor error occurred, if we have some best image/shapes
        if best_final_image is None or best_final_shapes is None:
            was_stopped = (
                self.stop_flag[0] == 1
            )  # check original state (already reset above)
            if not self.error_message:  # if no specific error message set yet
                stop_msg = " after being stopped" if was_stopped else ""
                self.error_message = (
                    f"Optimization did not produce a final image or shapes{stop_msg}."
                )
            print(f"Warning: {self.error_message}")
            return None, metrics_history, best_final_shapes  # return what we have

        was_stopped_before_reset = (
            self.stop_flag[0]
            == 1  # check before finally might have reset it (it's already reset)
        )
        if (
            was_stopped_before_reset
        ):  # this check is now redundant due to reset in finally
            print("Processing and saving best results found before stop...")

        end_run_time = datetime.datetime.now()
        elapsed_time = end_run_time - start_run_time
        print(f"Total Run Time: {elapsed_time}")
        print(f"Final Best Combined Metric: {best_final_metric:.5f}")

        # calculate final metrics on original scale (non-normalized)
        self.final_metrics_orig = {}
        print("Calculating final original metrics...")
        # ensure target image for original metrics is uint8
        img1_u8_target = target_image
        if img1_u8_target.dtype != np.uint8:
            if np.issubdtype(img1_u8_target.dtype, np.floating):
                temp_img = img1_u8_target * 255.0
                temp_img[temp_img < 0] = 0
                temp_img[temp_img > 255] = 255
                img1_u8_target = temp_img.astype(np.uint8)
            else:  # assume other integer types
                temp_img = img1_u8_target.copy()
                temp_img[temp_img < 0] = 0
                temp_img[temp_img > 255] = 255
                img1_u8_target = temp_img.astype(np.uint8)
        # ensure final best image is uint8
        img2_u8_final = best_final_image
        if img2_u8_final.dtype != np.uint8:
            if np.issubdtype(img2_u8_final.dtype, np.floating):
                temp_img = img2_u8_final * 255.0
                temp_img[temp_img < 0] = 0
                temp_img[temp_img > 255] = 255
                img2_u8_final = temp_img.astype(np.uint8)
            else:
                temp_img = img2_u8_final.copy()
                temp_img[temp_img < 0] = 0
                temp_img[temp_img > 255] = 255
                img2_u8_final = temp_img.astype(np.uint8)

        if img1_u8_target.shape != img2_u8_final.shape:
            # this would be a significant issue, likely from border handling or canvas logic
            self.error_message += f" Error: Final image shape {img2_u8_final.shape} mismatch target {img1_u8_target.shape}."
            print(f"Error: Shape mismatch between target and final image.")
        else:
            for name, func in metrics.ORIGINAL_METRIC_FUNCTIONS.items():
                try:
                    metric_value = func(img1_u8_target, img2_u8_final)
                    self.final_metrics_orig[name] = metric_value
                    value_to_print = metric_value
                    # format for printing, handling inf/nan
                    format_str = (
                        f"{value_to_print:.4f}"
                        if isinstance(value_to_print, (int, float))
                        and np.isfinite(value_to_print)
                        else str(value_to_print)
                    )
                    print(f"  {name.upper()}: {format_str}")
                except Exception as e:
                    self.final_metrics_orig[name] = "Error"  # store error string
                    print(f"  Error calculating original metric {name.upper()}: {e}")

        print("Saving final image(s)...")
        original_name = os.path.splitext(
            os.path.basename(passed_params["input_image_path"])
        )[0]
        # create a detailed filename string
        method_params_parts = []
        sorted_method_params = sorted(passed_params["method_params"].items())
        for k, v in sorted_method_params:  # short names for method params in filename
            k_short = {
                "init_temp": "T0",
                "cooling_rate": "CR",
                "iterations": "i",
                "swarm_size": "swarm",
                "cognitive_param": "cp",
                "social_param": "sp",
                "inertia_weight": "iw",
            }.get(
                k, k[:3]
            )  # default to first 3 chars if not in map
            # format float values nicely for filename
            v_str = (
                f"{v:.4f}".rstrip("0").rstrip(
                    "."
                )  # remove trailing zeros and decimal if integer-like
                if isinstance(v, float)
                and abs(v) > 1e-4
                and abs(v) < 1e4  # for typical range
                else (
                    f"{v:.2e}" if isinstance(v, float) else str(v)
                )  # scientific for extremes, or str
            )
            method_params_parts.append(f"{k_short}{v_str}")
        method_params_str = (
            "_".join(method_params_parts) if method_params_parts else "defaults"
        )

        coord_fix_str_fn_parts = []  # parts for coordinate fixing in filename
        coord_details_for_fn = passed_params.get("coord_fix_details", {})
        fixed_axes_initial = []  # X, Y
        fixed_axes_specific = []  # XS, YS (S for specific value)
        if coord_details_for_fn:
            sorted_axes = sorted(coord_details_for_fn.keys())  # ensure consistent order
            for axis in sorted_axes:
                details = coord_details_for_fn[axis]
                if details["mode"] == "initial":
                    fixed_axes_initial.append(axis.upper())
                elif details["mode"] == "specific":
                    fixed_axes_specific.append(f"{axis.upper()}S")
        if fixed_axes_initial:
            coord_fix_str_fn_parts.append(f"FixI{''.join(fixed_axes_initial)}")
        if fixed_axes_specific:
            coord_fix_str_fn_parts.append(f"FixS{''.join(fixed_axes_specific)}")
        coord_fix_str_fn = (  # final string part, FixIX_FixSYS or Dyn if none
            "_".join(coord_fix_str_fn_parts) if coord_fix_str_fn_parts else "Dyn"
        )

        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        metric_names_str = "_".join(
            passed_params.get("evaluation_metrics", ["unknown"])
        )
        param_init_str_fn = passed_params.get("param_init_type", "mid")[:3]  # mid, ran

        base_filename_img = f"{original_name}_{passed_params['shape_type']}{passed_params['num_shapes']}_{passed_params['optimization_method']}_{param_init_str_fn}_{coord_init_type}_{coord_fix_str_fn}_{method_params_str}_{metric_names_str}_{current_time_str}"

        save_successful_primary = False
        save_successful_gray_derived = False
        final_output_filename_primary = ""  # full path for results json
        final_output_filename_gray_derived = ""
        best_final_image_save = img2_u8_final  # use the uint8 version for saving
        grayscale_image_derived = None  # for "both" color scheme

        try:
            # determine suffix based on whether optimization ran as grayscale or rgb
            file_suffix = "grayscale" if run_optimization_as_grayscale else "rgb"
            final_output_filename_primary_base = f"{base_filename_img}_{file_suffix}"
            image_utils.save_image(
                best_final_image_save,
                output_directory,
                final_output_filename_primary_base,  # saves as .png by default
            )
            final_output_filename_primary = (
                f"{final_output_filename_primary_base}.png"  # store full name
            )
            print(f"Saved primary {file_suffix} image: {final_output_filename_primary}")
            save_successful_primary = True
        except Exception as e:
            save_error = f"Error saving primary image: {e}"
            print(save_error)
            self.error_message += f" ({save_error})"  # append to any existing error
            traceback.print_exc()

        # if color scheme is "both" and primary (rgb) image was saved, also save a derived grayscale version
        if (
            passed_params["color_scheme"] == "both"
            and save_successful_primary  # only if rgb saved
            and not run_optimization_as_grayscale  # ensure primary was indeed rgb
        ):
            try:
                print("Converting final RGB image to grayscale for 'both' scheme...")
                grayscale_image_derived = image_utils.convert_to_grayscale(
                    best_final_image_save  # convert the saved rgb image
                )
                gs_filename_base = f"{base_filename_img}_grayscale_derived"
                image_utils.save_image(
                    grayscale_image_derived, output_directory, gs_filename_base
                )
                final_output_filename_gray_derived = f"{gs_filename_base}.png"
                print(
                    f"Saved derived grayscale image: {final_output_filename_gray_derived}"
                )
                save_successful_gray_derived = True
            except Exception as e:
                save_error_both = f"Error saving derived grayscale image: {e}"
                print(save_error_both)
                self.error_message += f" ({save_error_both})"
                traceback.print_exc()

        gif_mode = passed_params.get("gif_options", "none")
        gif_file_paths = {}  # store {"type": "filename.gif"} for results json
        if gif_mode != "none" and gif_images:  # if gif requested and frames collected
            print("Creating GIF(s)...")
            try:
                gif_base = f"{base_filename_img}_anim"  # base for gif filenames
                if gif_mode in ("single", "both"):
                    saved_path = gif_creator.create_gif(
                        gif_images,
                        output_directory,
                        f"{gif_base}_single",  # ..._anim_single.gif
                        loop=1,  # single loop
                    )
                    if saved_path:
                        gif_file_paths["single_loop"] = os.path.basename(saved_path)
                if gif_mode in ("infinite", "both"):
                    saved_path = gif_creator.create_gif(
                        gif_images,
                        output_directory,
                        f"{gif_base}_infinite",
                        loop=0,  # infinite loop
                    )
                    if saved_path:
                        gif_file_paths["infinite_loop"] = os.path.basename(saved_path)
            except Exception as e:
                gif_error = f"GIF creation failed: {e}"
                print(f"Warning: {gif_error}")
                self.error_message += f" ({gif_error})"
                traceback.print_exc()

        # save detailed results to a json file if requested
        if passed_params.get("save_results_file", False):
            base_filename_results = (
                f"results_{original_name}_{current_time_str}"  # base for json/npy
            )
            results_filepath_base = os.path.join(
                output_directory, base_filename_results
            )
            results_dict = {}  # dictionary to be saved as json
            results_dict["run_timestamp"] = start_run_time.isoformat()
            results_dict["completion_timestamp"] = end_run_time.isoformat()
            results_dict["input_image"] = passed_params["input_image_path"]
            try:  # gather stats about original input image
                in_h, in_w = original_shape  # original dims before border
                in_c_str = "N/A"
                try:  # try to reload to get channel count if not easily available
                    input_image_for_stats = image_utils.load_image(
                        passed_params["input_image_path"]
                    )
                    if input_image_for_stats is not None:
                        in_c = (
                            input_image_for_stats.shape[2]
                            if input_image_for_stats.ndim == 3
                            else 1  # assume 1 channel for 2d
                        )
                        in_c_str = str(in_c)
                except Exception:
                    print("Warning: Could not reload input image for stats.")
                in_size_kb = os.path.getsize(passed_params["input_image_path"]) / 1024
                results_dict["input_original_dims"] = {
                    "w": in_w,
                    "h": in_h,
                    "c": in_c_str,
                }
                results_dict["input_file_size_kb"] = float(f"{in_size_kb:.2f}")
            except Exception as img_e:
                results_dict["input_stats_error"] = str(img_e)

            results_dict["settings"] = {
                "shape_type": passed_params["shape_type"],
                "num_shapes": passed_params["num_shapes"],  # target K for this run
                "color_scheme": passed_params["color_scheme"],
                "border_type": passed_params["border_type"],
                "param_init_mode": passed_params["param_init_type"],
                "coord_init_mode": coord_init_type,  # actual coord init used
                "coord_fix_details": passed_params["coord_fix_details"],
                "fixed_non_coord_params": dict(  # store as dict for readability
                    zip(
                        params[
                            "fixed_params_non_coord"
                        ],  # use params from CLI collection
                        params["fixed_values_non_coord"],
                    )
                ),
                "optimization_method": passed_params["optimization_method"].upper(),
                "method_parameters": passed_params["method_params"],
                "evaluation_metrics": metric_names,  # validated and used metrics
                "metric_weights": [
                    float(f"{w:.4f}") for w in metric_weights
                ],  # used weights
                "loaded_shapes_info": None,  # placeholder for info if shapes were loaded
            }
            if passed_params["use_loaded_shapes"]:
                num_loaded_actual = (  # N, actual number of shapes in the loaded array
                    len(passed_params["loaded_shapes_array"])
                    if passed_params["loaded_shapes_array"] is not None
                    else 0
                )
                k_gui = passed_params["num_shapes"]  # K, target for current run
                load_info = {
                    "used_loaded_shapes": True,
                    "num_loaded": num_loaded_actual,
                    "param_names_config": passed_params["loaded_param_names_config"],
                    "render_loaded_as_canvas": passed_params["render_loaded_as_canvas"],
                    "truncation_mode": (  # only relevant if not canvas and K < N
                        passed_params["truncation_mode"]
                        if not passed_params["render_loaded_as_canvas"]
                        and k_gui < num_loaded_actual
                        else None
                    ),
                    "rendered_untruncated": (  # only relevant if not canvas and K < N
                        passed_params["render_untruncated_shapes"]
                        if not passed_params["render_loaded_as_canvas"]
                        and k_gui < num_loaded_actual
                        else None
                    ),
                    "saved_optimized_only": passed_params[
                        "save_optimized_only"
                    ],  # the general flag
                }
                results_dict["settings"]["loaded_shapes_info"] = load_info

            results_dict["results"] = {
                "total_elapsed_time_sec": elapsed_time.total_seconds(),
                "final_best_combined_metric": float(f"{best_final_metric:.6f}"),
                "final_original_metrics": {},  # placeholder
                "status": (
                    "Completed" if self.stop_flag[0] == 0 else "Stopped"
                ),  # final status
            }
            if self.final_metrics_orig:  # populate original metrics
                for name, val in sorted(self.final_metrics_orig.items()):
                    format_val = val
                    if isinstance(val, (int, float)) and not np.isfinite(val):
                        format_val = str(val)  # store inf/nan as string
                    elif isinstance(
                        val, np.generic
                    ):  # convert numpy types to python types
                        format_val = val.item()
                    results_dict["results"]["final_original_metrics"][
                        name.upper()
                    ] = format_val
            else:
                results_dict["results"][
                    "final_original_metrics"
                ] = "No original metrics calculated or error occurred"

            results_dict["output"] = {
                "image_files": [],  # list of dicts for each saved image file
                "gif_mode": passed_params["gif_options"],
                "gif_files": gif_file_paths,  # dict of {"type": "filename.gif"}
            }
            if save_successful_primary:  # if primary image was saved
                primary_filename = os.path.basename(final_output_filename_primary)
                primary_entry = {
                    "type": "primary",
                    "filename": primary_filename,
                    "dims": None,
                    "size_kb": None,
                }
                try:  # gather stats for saved primary output image
                    primary_filepath_full = (
                        os.path.normpath(  # ensure path is correct for os
                            os.path.join(output_directory, primary_filename)
                        )
                    )
                    out_h, out_w = best_final_image_save.shape[:2]
                    out_c = (
                        best_final_image_save.shape[2]
                        if best_final_image_save.ndim == 3
                        else 1
                    )
                    out_size_kb = os.path.getsize(primary_filepath_full) / 1024
                    primary_entry["dims"] = {"w": out_w, "h": out_h, "c": out_c}
                    primary_entry["size_kb"] = float(f"{out_size_kb:.2f}")
                except Exception as out_e:
                    primary_entry["stats_error"] = str(out_e)
                results_dict["output"]["image_files"].append(primary_entry)
            else:  # if primary image save failed
                results_dict["output"]["image_files"].append(
                    {
                        "type": "primary",
                        "filename": "Not Saved Successfully",
                        "dims": None,
                        "size_kb": None,
                    }
                )

            # if "both" scheme and derived grayscale was saved
            if (
                passed_params["color_scheme"] == "both"
                and save_successful_gray_derived
                and grayscale_image_derived is not None
            ):
                gray_filename = os.path.basename(final_output_filename_gray_derived)
                gray_entry = {
                    "type": "grayscale_derived",
                    "filename": gray_filename,
                    "dims": None,
                    "size_kb": None,
                }
                try:  # gather stats for saved derived grayscale image
                    gray_filepath_full = os.path.normpath(
                        os.path.join(output_directory, gray_filename)
                    )
                    out_h, out_w = grayscale_image_derived.shape[:2]
                    out_c = 1  # grayscale is always 1 channel
                    out_size_kb = os.path.getsize(gray_filepath_full) / 1024
                    gray_entry["dims"] = {"w": out_w, "h": out_h, "c": out_c}
                    gray_entry["size_kb"] = float(f"{out_size_kb:.2f}")
                except Exception as out_e:
                    gray_entry["stats_error"] = str(out_e)
                results_dict["output"]["image_files"].append(gray_entry)
            elif (  # if "both" but derived grayscale save failed
                passed_params["color_scheme"] == "both"
                and not save_successful_gray_derived
            ):
                results_dict["output"]["image_files"].append(
                    {
                        "type": "grayscale_derived",
                        "filename": "Not Saved Successfully",
                        "dims": None,
                        "size_kb": None,
                    }
                )

            # add metadata required for reloading shapes from this results file
            results_dict["shape_type"] = passed_params["shape_type"]
            results_dict["color_scheme"] = passed_params["color_scheme"]
            results_dict["param_names_config"] = (
                config_param_names  # crucial for interpreting saved .npy
            )

            shapes_to_save = None  # the actual np.array of shapes to save to .npy
            num_shapes_for_json = (
                0  # the 'num_shapes' field in json, reflects count in .npy
            )
            save_shape_data_flag = passed_params.get("save_shape_data", False)

            if save_shape_data_flag and best_final_shapes is not None:
                save_optimized_only = passed_params[
                    "save_optimized_only"
                ]  # general flag
                k_optimized = best_final_shapes.shape[
                    0
                ]  # number of shapes in the optimized array

                if (  # if shapes were loaded initially
                    passed_params["use_loaded_shapes"]
                    and passed_params["loaded_shapes_array"] is not None
                ):
                    loaded_array_n = passed_params["loaded_shapes_array"]
                    n_loaded = loaded_array_n.shape[0]  # N
                    render_canvas = passed_params["render_loaded_as_canvas"]

                    if save_optimized_only:
                        # if this flag is true, always save just the K optimized shapes
                        shapes_to_save = best_final_shapes
                        num_shapes_for_json = k_optimized
                        print(
                            f"Saving only K={k_optimized} optimized shapes ('Save Optimized Only' checked)."
                        )
                    else:  # save_optimized_only is false, so try to merge/combine
                        if render_canvas:
                            # save N loaded + K new optimized shapes
                            shapes_to_save = np.vstack(
                                (loaded_array_n, best_final_shapes)
                            )
                            num_shapes_for_json = n_loaded + k_optimized
                            print(
                                f"Saving combined N={n_loaded} loaded + K={k_optimized} optimized shapes (Render Canvas mode)."
                            )
                        elif (
                            k_optimized < n_loaded
                        ):  # truncation case (K_optimized < N_loaded)
                            selected_indices = passed_params.get(
                                "selected_indices_for_save"  # indices of loaded shapes that were optimized
                            )
                            if (  # ensure indices are valid for merging
                                selected_indices is not None
                                and len(selected_indices) == k_optimized
                            ):
                                full_shapes = loaded_array_n.copy()
                                full_shapes[selected_indices] = (
                                    best_final_shapes  # update the selected ones
                                )
                                shapes_to_save = full_shapes  # save the full N shapes
                                num_shapes_for_json = n_loaded
                                print(
                                    f"Saving merged N={n_loaded} shapes (K={k_optimized} updated) (Truncation mode)."
                                )
                            else:  # fallback if indices are bad, just save optimized K
                                shapes_to_save = best_final_shapes
                                num_shapes_for_json = k_optimized
                                print(
                                    "Warning: Indices missing for merging. Saving only K optimized shapes."
                                )
                        else:  # K_optimized >= N_loaded (superset or K=N case)
                            # in these cases, best_final_shapes already contains combined/all relevant shapes
                            shapes_to_save = best_final_shapes
                            num_shapes_for_json = k_optimized
                            print(
                                f"Saving K={k_optimized} optimized shapes (K>=N or K=N mode)."
                            )
                else:  # not use_loaded_shapes, so just save the K optimized shapes
                    shapes_to_save = best_final_shapes
                    num_shapes_for_json = best_final_shapes.shape[0]
                    print(
                        f"Saving K={num_shapes_for_json} optimized shapes (No load mode)."
                    )
            # this 'num_shapes' in json refers to the number of shapes in the accompanying .npy file
            results_dict["num_shapes"] = num_shapes_for_json

            try:
                save_results_and_shapes(
                    results_filepath_base,
                    results_dict,
                    shapes_to_save,  # the determined array of shapes
                    save_shape_data_flag,  # only save .npy if this is true and shapes_to_save is not None
                )
            except Exception as e:
                txt_error = f"Failed to save results/shapes: {e}"
                print(f"Error: {txt_error}")
                self.error_message += f" ({txt_error})"
                traceback.print_exc()

        return (
            best_final_image_save,  # the uint8 image
            metrics_history,
            best_final_shapes,  # the float32 array of shapes
        )
