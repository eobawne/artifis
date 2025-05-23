import numpy as np
import json
import os
from typing import Tuple, Dict, Any, List, Optional


def save_results_and_shapes(
    results_filepath_base: str,
    results_data: Dict[str, Any],
    shapes_array: Optional[np.ndarray] = None,
    save_shape_data_flag: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Saves results data to a JSON file and optionally saves shape data to an NPY file

    The NPY filename is derived from the JSON filename and stored within the JSON data
    if shapes are saved

    :param results_filepath_base: The base path and filename without extension for the results JSON
    :type results_filepath_base: str
    :param results_data: Dictionary containing results and metadata to be saved in JSON
    :type results_data: Dict[str, Any]
    :param shapes_array: The NumPy array of shape parameters (N x P), if saving
    :type shapes_array: Optional[np.ndarray]
    :param save_shape_data_flag: Boolean indicating if `shapes_array` should be saved to an NPY file
    :type save_shape_data_flag: bool
    :return: A tuple containing (path_to_saved_json, path_to_saved_npy)
             Paths can be None if saving respective file failed or was not requested
    :rtype: Tuple[Optional[str], Optional[str]]
    """
    json_filepath = f"{results_filepath_base}.json"
    npy_filepath = None  # full path to npy file if created
    saved_npy_path_ret = None  # return value for npy path

    results_data_to_save = (
        results_data.copy()
    )  # work on a copy to avoid modifying original dict

    # if requested and shape data is available, prepare to save npy file
    if save_shape_data_flag and shapes_array is not None:
        # construct npy filename based on the json base filename
        npy_filename = f"{os.path.basename(results_filepath_base)}_shapes.npy"
        # store npy filename in the json data for future loading reference
        results_data_to_save["shape_data_file"] = npy_filename
        npy_filepath = os.path.join(
            os.path.dirname(results_filepath_base), npy_filename
        )
    else:
        # ensure field exists but is null if shapes are not saved or not provided
        results_data_to_save["shape_data_file"] = None

    # save json results file
    try:
        with open(json_filepath, "w") as f:
            json.dump(results_data_to_save, f, indent=4)  # pretty print with indent
        print(f"Saved results to: {json_filepath}")
    except Exception as e:
        print(f"Error saving results JSON file {json_filepath}: {e}")
        return None, None  # indicate failure for both if json fails

    # save npy shapes file if path was determined and array exists
    if npy_filepath and shapes_array is not None:
        try:
            np.save(npy_filepath, shapes_array)
            print(f"Saved shape data to: {npy_filepath}")
            saved_npy_path_ret = npy_filepath  # store path for return
        except Exception as e:
            print(f"Error saving shape data NPY file {npy_filepath}: {e}")
            # json was saved, but npy failed; return json path and None for npy
            return json_filepath, None

    return json_filepath, saved_npy_path_ret


def load_results_and_shapes(
    json_filepath: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Loads results data from a JSON file and optionally loads associated shape data from an NPY file

    The NPY file to load is determined by the 'shape_data_file' field within the JSON data

    :param json_filepath: Path to the results JSON file
    :type json_filepath: str
    :return: A tuple containing (loaded_results_data_dict, loaded_shapes_numpy_array)
             Either can be None if loading failed or data was not present
    :rtype: Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]
    """
    if not os.path.exists(json_filepath):
        print(f"Error: Results JSON file not found: {json_filepath}")
        return None, None

    results_data = None
    shapes_array = None

    # load json results
    try:
        with open(json_filepath, "r") as f:
            results_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in results file: {json_filepath}")
        return None, None  # cannot proceed if json is invalid
    except Exception as e:
        print(f"Error reading results JSON file {json_filepath}: {e}")
        return None, None

    # attempt to load shapes npy file if specified in json
    npy_filename_in_json = results_data.get("shape_data_file")
    if npy_filename_in_json and isinstance(npy_filename_in_json, str):
        # construct full path to npy file, assuming it's in same dir as json
        npy_filepath_to_load = os.path.join(
            os.path.dirname(json_filepath), npy_filename_in_json
        )
        if os.path.exists(npy_filepath_to_load):
            try:
                shapes_array = np.load(npy_filepath_to_load)
                print(f"Loaded shape data from: {npy_filepath_to_load}")
            except Exception as e:
                print(f"Error loading shape data NPY file {npy_filepath_to_load}: {e}")
                # return loaded json data (if successful), but None for shapes due to npy load error
                return results_data, None
        else:
            # npy file specified in json but not found on disk
            print(
                f"Warning: Shape data file specified in JSON not found: {npy_filepath_to_load}"
            )
            # json loaded, but npy missing; return json data, None for shapes
            return results_data, None
    # else: shape_data_file was null, not specified, or not a string in json
    # in this case, shapes_array remains None, which is correct

    return results_data, shapes_array
