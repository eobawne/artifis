import json
import os
from typing import Dict, List, Any


class ConfigLoader:
    """
    Loads and validates shape and border configurations from JSON files

    It expects configurations to be stored in specified directories
    and performs basic validation on the structure of these JSON files,
    particularly for shape configurations
    """

    def __init__(
        self, shapes_dir: str = "configs/shapes", borders_dir: str = "configs/borders"
    ):
        """
        Initializes the ConfigLoader

        :param shapes_dir: Path to the directory containing shape configurations
                           (relative to the project root or an absolute path)
        :type shapes_dir: str
        :param borders_dir: Path to the directory containing border configurations
                            (relative to the project root or an absolute path)
        :type borders_dir: str
        """
        self.shapes_dir = shapes_dir
        self.borders_dir = borders_dir

    def load_shape_config(self, shape_type: str) -> Dict[str, Any]:
        """
        Loads the configuration for a specific shape type

        The configuration is expected to be a JSON file named `{shape_type}.json`
        within the `self.shapes_dir` directory

        :param shape_type: The type of shape ("circle", "rectangle")
                           This corresponds to the filename without the .json extension
        :type shape_type: str
        :raises FileNotFoundError: If the configuration file is not found
        :raises ValueError: If the JSON is invalid or missing required fields
        :return: A dictionary containing the shape configuration
        :rtype: Dict[str, Any]
        """
        filepath = os.path.join(self.shapes_dir, f"{shape_type}.json")
        return self._load_config(filepath, "shape")

    def load_border_config(self, border_type: str) -> Dict[str, Any]:
        """
        Loads the configuration for a specific border type

        The configuration is expected to be a JSON file named `{border_type}.json`
        within the `self.borders_dir` directory

        :param border_type: The type of border ("b_circle", "b_original")
                            This corresponds to the filename without the .json extension
        :type border_type: str
        :raises FileNotFoundError: If the configuration file is not found
        :raises ValueError: If the JSON is invalid
        :return: A dictionary containing the border configuration
        :rtype: Dict[str, Any]
        """
        filepath = os.path.join(self.borders_dir, f"{border_type}.json")
        # border configs currently don't have specific validation beyond being valid json
        return self._load_config(filepath, "border")

    def _load_config(self, filepath: str, config_type: str) -> Dict[str, Any]:
        """
        Loads and validates a generic configuration file (shape or border)

        This is a helper method used by `load_shape_config` and `load_border_config`

        :param filepath: The path to the JSON configuration file
        :type filepath: str
        :param config_type:  A string indicating the type of configuration ("shape" or "border"),
                             used for error messages
        :type config_type: str
        :raises FileNotFoundError: If the file doesn't exist
        :raises ValueError:  If the JSON is invalid or, for shapes, missing required fields
        :return: The loaded configuration as a dictionary
        :rtype: Dict[str, Any]
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Could not find {config_type} configuration file: {filepath}"
            )

        with open(filepath, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                # raise a more specific error if json parsing fails
                raise ValueError(
                    f"Invalid JSON in {config_type} configuration file: {filepath}"
                )

        # perform specific validation for shape configurations
        if config_type == "shape":
            self._validate_shape_config(config, filepath)
        return config

    def _validate_shape_config(self, config: Dict[str, Any], filepath: str):
        """
        Validates the structure and content of a loaded shape configuration dictionary

        Checks for essential keys like "shape" and "parameters", and validates
        the structure of each parameter entry (name, type, min/max if present)

        :param config: The loaded shape configuration dictionary
        :type config: Dict[str, Any]
        :param filepath: The path to the configuration file (used for error messages)
        :type filepath: str
        :raises ValueError: If the configuration is invalid (missing fields, incorrect types, etc)
        """
        if "shape" not in config:
            raise ValueError(
                f"Missing 'shape' field in shape configuration: {filepath}"
            )
        if "parameters" not in config:
            raise ValueError(
                f"Missing 'parameters' field in shape configuration: {filepath}"
            )
        if not isinstance(config["parameters"], list):
            raise ValueError(
                f"'parameters' field must be a list in shape configuration: {filepath}"
            )

        # validate each parameter defined in the "parameters" list
        for param in config["parameters"]:
            if not all(key in param for key in ["name", "type"]):
                raise ValueError(
                    f"Each parameter must have 'name', 'type' fields: {filepath}"
                )
            # check for supported parameter types
            if param["type"] not in [
                "float32",
                "int32",
                "str",
            ]:  # str type is listed but not extensively used elsewhere currently
                raise ValueError(
                    f"Invalid parameter type '{param['type']}' Must be 'float32', 'int32' or 'str': {filepath}"
                )
            # validate min/max if present
            if "min" in param and not isinstance(param["min"], (int, float)):
                raise ValueError(f"Parameter 'min' must be a number: {filepath}")
            if "max" in param and not isinstance(param["max"], (int, float)):
                raise ValueError(f"Parameter 'max' must be a number: {filepath}")
            if "min" in param and "max" in param and param["min"] > param["max"]:
                # this check ensures min is not greater than max
                # the actual enforcement of min <= max happens in app.py when processing these
                raise ValueError(
                    f"Parameter 'min' cannot be greater than 'max': {filepath}"
                )

    def get_available_shapes(self) -> List[str]:
        """
        Returns a list of available shape types by scanning the shapes configuration directory

        Shape types are derived from the filenames (excluding .json extension)

        :return: A list of strings, where each string is an available shape type
        :rtype: List[str]
        """
        # list comprehension to find all .json files and extract their names
        return [
            f.replace(".json", "")
            for f in os.listdir(self.shapes_dir)
            if f.endswith(".json")
        ]

    def get_available_borders(self) -> List[str]:
        """
        Returns a list of available border types by scanning the borders configuration directory

        Border types are derived from the filenames (excluding .json extension)

        :return: A list of strings, where each string is an available border type
        :rtype: List[str]
        """
        return [
            f.replace(".json", "")
            for f in os.listdir(self.borders_dir)
            if f.endswith(".json")
        ]
