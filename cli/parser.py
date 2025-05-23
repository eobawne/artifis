import argparse
import numpy as np
from utils.config_loader import ConfigLoader
from core.metrics import AVAILABLE_METRICS
import os
import sys


class ShapeDataRequiresResults(argparse.Action):
    """
    Custom argparse action to ensure --save-shapes requires --save-results
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Checks if --save-results is specified when --save-shapes is used

        :param parser: The ArgumentParser object
        :type parser: argparse.ArgumentParser
        :param namespace: The argparse.Namespace object to store attributes
        :type namespace: argparse.Namespace
        :param values: The associated command line arguments (not used for this flag)
        :param option_string: The option string that was used ('--save-shapes')
        :type option_string: str
        :raises argparse.ArgumentError: If --save-results is not specified
        """
        save_results_specified = getattr(namespace, "save_results", False)
        if not save_results_specified:
            parser.error(f"{option_string} requires --save-results to be specified")
        # store value, for flags that don't take an argument, this is typically None or a const
        setattr(namespace, self.dest, values)


class OptimizedOnlyRequiresSaveShapes(argparse.Action):
    """
    Custom argparse action to ensure --save-optimized-only requires --save-shapes
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Checks if --save-shapes is specified when --save-optimized-only is used

        :param parser: The ArgumentParser object
        :type parser: argparse.ArgumentParser
        :param namespace: The argparse.Namespace object to store attributes
        :type namespace: argparse.Namespace
        :param values: The associated command line arguments (not used for this flag)
        :param option_string: The option string that was used ('--save-optimized-only')
        :type option_string: str
        :raises argparse.ArgumentError: If --save-shapes is not specified
        """
        # check if --save-shapes was specified by looking at its internal destination name
        save_shapes_specified = (
            getattr(namespace, "save_shapes_flag_internal", None) is not None
        )
        if not save_shapes_specified:
            parser.error(f"{option_string} requires --save-shapes to be specified")
        # store True if flag is present, as it's a boolean flag action
        setattr(namespace, self.dest, True)


def parse_arguments(args):
    """
    Parses command line arguments for the Artifis application

    :param args: A list of command line arguments (typically sys.argv[1:])
    :type args: list[str]
    :return: An argparse.Namespace object containing parsed arguments
    :rtype: argparse.Namespace
    """
    config_loader = ConfigLoader()
    available_shapes = config_loader.get_available_shapes()
    available_borders = config_loader.get_available_borders()
    available_metrics = AVAILABLE_METRICS
    parser = argparse.ArgumentParser(
        description="Image approximation with basic shapes"
    )

    # input/output args
    parser.add_argument("-i", "--input", required=True, help="Path to input image")
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output directory"
    )

    # core params
    parser.add_argument(
        "-s", "--shape", choices=available_shapes, required=True, help="Shape type"
    )
    parser.add_argument(
        "-n",
        "--num_shapes",
        type=int,
        help="Number of shapes (K) If loading shapes and this is omitted, uses loaded count (N)",
    )
    parser.add_argument(
        "-c",
        "--color",
        choices=["grayscale", "rgb", "both"],
        required=True,
        help="Color scheme (determines target image and parameter structure)",
    )
    parser.add_argument(
        "-b",
        "--border",
        choices=available_borders,
        default="b_original",
        help="Border type",
    )
    # parameter initialization mode argument
    parser.add_argument(
        "--param-init",
        choices=["midpoint", "min", "max", "random"],
        default="midpoint",
        help="Initialization mode for configurable shape parameters (radius, length)",
    )

    # coordinate settings group
    coord_group = parser.add_argument_group("Coordinate Settings")
    coord_group.add_argument(
        "--coord-init",
        choices=["random", "grid", "zero", "intensity_pdf", "ssim_pdf"],
        default="random",
        help="Coordinate initialization mode",
    )
    # flags to indicate fixing an axis (either to initial or specific value)
    # only allow fixing x, y for all shapes, as x2/y2 are derived for line
    coord_group.add_argument(
        "--fix-coord-x",
        action="store_true",
        help="Fix X coordinate axis (x or x1) (to initial or specific value if provided)",
    )
    coord_group.add_argument(
        "--fix-coord-y",
        action="store_true",
        help="Fix Y coordinate axis (y or y1) (to initial or specific value if provided)",
    )
    # args for setting specific fixed values (override freezing to initial)
    coord_group.add_argument(
        "--fixed-coord-x-val",
        type=float,
        help="Set ALL shapes' X coordinate (x or x1) to this specific value (requires --fix-coord-x)",
    )
    coord_group.add_argument(
        "--fixed-coord-y-val",
        type=float,
        help="Set ALL shapes' Y coordinate (y or y1) to this specific value (requires --fix-coord-y)",
    )

    # non-coordinate fixing group
    fix_param_group = parser.add_argument_group("Non-Coordinate Parameter Fixing")
    fix_param_group.add_argument(
        "--fix-params-non-coord",
        dest="fix_params_non_coord",
        nargs="*",
        default=[],
        help="Names of NON-COORDINATE parameters to fix to specific values (radius, length, angle, alpha, r, g, b, gray, stroke_)",
    )
    fix_param_group.add_argument(
        "--fix-values-non-coord",
        dest="fix_values_non_coord",
        type=float,
        nargs="*",
        default=[],
        help="Specific values for --fix-params-non-coord (must match order and count)",
    )

    # optimization params group
    opt_group = parser.add_argument_group("Optimization Settings")
    opt_group.add_argument(
        "-m",
        "--method",
        choices=["sa", "pso", "hc"],  # only list implemented methods
        required=True,
        help="Optimization method",
    )
    opt_group.add_argument(
        "-e",
        "--metric",
        choices=available_metrics,
        nargs="+",
        required=True,
        help="Metric(s) for evaluation (1-3) Higher score (closer to 1) is better for normalized metrics",
    )
    opt_group.add_argument(
        "-w",
        "--weights",
        type=float,
        nargs="+",
        help="Weights for metrics (must sum to 10, will be normalized otherwise)",
    )

    # output options group
    output_group = parser.add_argument_group("Output Settings")
    output_group.add_argument(
        "-g",
        "--gif",
        choices=["none", "single", "infinite", "both"],
        default="none",
        help="GIF generation option",
    )
    output_group.add_argument(
        "--save-results",
        action="store_true",
        help="Save detailed results to a json file",
    )
    # add save shapes flag with dependency check
    output_group.add_argument(
        "--save-shapes",
        action=ShapeDataRequiresResults,  # use custom action
        nargs=0,  # flag doesn't take value, presence matters
        help="Save final shape parameters to a npy file (requires --save-results)",
        dest="save_shapes_flag_internal",  # use different dest to avoid conflict later
    )
    # rename flag and update help text and action
    output_group.add_argument(
        "--save-optimized-only",
        action=OptimizedOnlyRequiresSaveShapes,  # use new custom action
        nargs=0,
        help="When saving shape data after loading, save only K optimized shapes, not combined/merged set (requires --save-shapes and --load-shapes)",
        dest="save_optimized_only",  # directly store flag
    )

    # loading shapes group
    load_group = parser.add_argument_group("Loading Settings")
    load_group.add_argument(
        "--load-shapes",
        metavar="PATH_TO_JSON",
        help="Path to a results json file to load initial shapes and settings from",
    )
    # cli options for handling loaded shapes (simplified for cli)
    load_group.add_argument(
        "--load-render-canvas",
        action="store_true",
        help="Render all loaded shapes as initial canvas before optimizing new shapes (requires --load-shapes)",
    )
    load_group.add_argument(
        "--load-truncate-mode",
        choices=["random", "first", "last"],
        default="first",  # different default for cli simplicity
        help="How to truncate loaded shapes if --num-shapes is less than loaded count (requires --load-shapes, ignored if --load-render-canvas)",
    )
    load_group.add_argument(
        "--load-render-untruncated",
        action="store_true",
        help="Render untruncated shapes onto canvas when truncating (requires --load-shapes and K < N, ignored if --load-render-canvas)",
    )

    # method subparsers / parameters group
    method_params_group = parser.add_argument_group("Method-Specific Parameters")
    # common
    method_params_group.add_argument(
        "--iterations", type=int, help="SA/PSO: Number of iterations"
    )
    # sa
    method_params_group.add_argument(
        "--init_temp", type=float, help="SA: Initial temperature"
    )
    method_params_group.add_argument(
        "--cooling_rate", type=float, help="SA: Cooling rate"
    )
    # pso
    method_params_group.add_argument("--swarm_size", type=int, help="PSO: Swarm size")
    method_params_group.add_argument(
        "--cognitive_param", type=float, help="PSO: Cognitive parameter (c1)"
    )
    method_params_group.add_argument(
        "--social_param", type=float, help="PSO: Social parameter (c2)"
    )
    method_params_group.add_argument(
        "--inertia_weight", type=float, help="PSO: Inertia weight (w)"
    )

    parsed_args = parser.parse_args(args)

    # post-parsing validation and processing

    # loading validation
    if parsed_args.load_shapes and not os.path.exists(parsed_args.load_shapes):
        parser.error(f"Load shapes JSON file not found: {parsed_args.load_shapes}")
    if parsed_args.load_render_canvas and not parsed_args.load_shapes:
        parser.error("--load-render-canvas requires --load-shapes")
    # note: further validation of loaded file content happens in app.py

    # set save_shapes_flag based on presence (action stores None if flag present)
    save_shapes_flag = (
        hasattr(parsed_args, "save_shapes_flag_internal")
        and parsed_args.save_shapes_flag_internal is None
    )
    parsed_args.save_shapes_flag = save_shapes_flag

    # saving validation (redundant due to custom action, but safe)
    if save_shapes_flag and not parsed_args.save_results:
        parser.error("--save-shapes requires --save-results")

    # set save_optimized_only flag (action stores True or it defaults to None/False)
    parsed_args.save_optimized_only = getattr(parsed_args, "save_optimized_only", False)
    if parsed_args.save_optimized_only and not save_shapes_flag:
        parser.error("--save-optimized-only requires --save-shapes")
    # also requires load shapes
    if parsed_args.save_optimized_only and not parsed_args.load_shapes:
        parser.error(
            "--save-optimized-only requires --load-shapes (as it only affects saving loaded/merged data)"
        )

    # metrics/weights validation
    if isinstance(parsed_args.metric, str):
        parsed_args.metric = [parsed_args.metric]
    num_metrics = len(parsed_args.metric)
    if not (1 <= num_metrics <= 3):
        parser.error("Please provide 1 to 3 evaluation metrics using --metric")
    if parsed_args.weights is None:
        if num_metrics > 1:
            print(
                f"Warning: No weights provided for {num_metrics} metrics Using equal weights"
            )
        parsed_args.weights = [1.0 / num_metrics] * num_metrics
    elif isinstance(parsed_args.weights, float):
        parsed_args.weights = [parsed_args.weights]
    if len(parsed_args.weights) != num_metrics:
        parser.error(
            f"Number of weights ({len(parsed_args.weights)}) must match number of metrics ({num_metrics})"
        )
    weight_sum = sum(parsed_args.weights)
    if not np.isclose(weight_sum, 1.0):
        if weight_sum > 1e-6:
            print(
                f"Warning: Metric weights sum to {weight_sum:.4f} Normalizing weights"
            )
            parsed_args.weights = [w / weight_sum for w in parsed_args.weights]
        else:
            print(
                f"Warning: Metric weights sum ({weight_sum:.4f}) close to zero Using equal weights"
            )
            parsed_args.weights = [1.0 / num_metrics] * num_metrics
    elif any(w < 0 for w in parsed_args.weights):
        parser.error("Metric weights cannot be negative")

    # fixed params (non-coord) count match
    if len(parsed_args.fix_params_non_coord) != len(parsed_args.fix_values_non_coord):
        parser.error(
            f"Mismatch between --fix-params-non-coord ({len(parsed_args.fix_params_non_coord)}) and --fix-values-non-coord ({len(parsed_args.fix_values_non_coord)})"
        )

    # coordinate fixing value validation
    if parsed_args.fixed_coord_x_val is not None and not parsed_args.fix_coord_x:
        parser.error("--fixed-coord-x-val requires --fix-coord-x")
    if parsed_args.fixed_coord_y_val is not None and not parsed_args.fix_coord_y:
        parser.error("--fixed-coord-y-val requires --fix-coord-y")

    # build coord_fix_details dict for app.py
    parsed_args.coord_fix_details = {}
    coord_axes_flags_values = {
        "x": (parsed_args.fix_coord_x, parsed_args.fixed_coord_x_val),
        "y": (parsed_args.fix_coord_y, parsed_args.fixed_coord_y_val),
    }
    for axis, (fix_flag, specific_val) in coord_axes_flags_values.items():
        if fix_flag:
            if specific_val is not None:
                parsed_args.coord_fix_details[axis] = {
                    "mode": "specific",
                    "value": specific_val,
                }
            else:
                parsed_args.coord_fix_details[axis] = {"mode": "initial", "value": None}

    # method defaults / iterations
    # check if user specified num_shapes explicitly by looking at raw sys.argv
    user_provided_num_shapes = False
    for arg_idx, arg_val in enumerate(sys.argv):
        if arg_val == "--num-shapes" or arg_val == "-n":
            if arg_idx + 1 < len(sys.argv):
                user_provided_num_shapes = True
                break
    # set num_shapes default only if loading shapes and user didn't provide it
    if parsed_args.load_shapes and not user_provided_num_shapes:
        # num_shapes will be set during loading in collect_parameters_cli if needed
        # so, allow it to be None here if not provided by user
        if parsed_args.num_shapes is None:
            pass
    elif parsed_args.num_shapes is None:  # required if not loading and not provided
        parser.error("the following arguments are required: -n/--num_shapes")

    # method param defaults
    if parsed_args.iterations is None:
        if parsed_args.method in ["sa", "hc"]:
            parsed_args.iterations = 10000
        elif parsed_args.method == "pso":
            parsed_args.iterations = 1000
        # only print default message if method is one that uses iterations
        if parsed_args.method in ["sa", "pso", "hc"]:
            print(
                f"Using default iterations: {parsed_args.iterations} for {parsed_args.method.upper()}"
            )
        else:  # should not happen if choices are restricted
            parser.error(
                f"Unknown method '{parsed_args.method}' for setting default iterations"
            )

    if parsed_args.method == "sa":
        if parsed_args.init_temp is None:
            parsed_args.init_temp = 1.0
        if parsed_args.cooling_rate is None:
            parsed_args.cooling_rate = 0.95
    elif parsed_args.method == "pso":
        if parsed_args.swarm_size is None:
            parsed_args.swarm_size = 50
        if parsed_args.cognitive_param is None:
            parsed_args.cognitive_param = 1.5
        if parsed_args.social_param is None:
            parsed_args.social_param = 1.5
        if parsed_args.inertia_weight is None:
            parsed_args.inertia_weight = 0.7

    # remove temporary internal flags used for validation logic
    del parsed_args.fix_coord_x
    del parsed_args.fix_coord_y
    del parsed_args.fixed_coord_x_val
    del parsed_args.fixed_coord_y_val
    if hasattr(parsed_args, "save_shapes_flag_internal"):
        delattr(parsed_args, "save_shapes_flag_internal")

    return parsed_args
