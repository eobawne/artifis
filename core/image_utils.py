import imageio.v3 as iio
import numpy as np
import os


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the specified path using imageio

    Converts the loaded image to a NumPy array Ensures the output array
    has dtype uint8 and values in the range [0, 255] Handles common
    image modes (grayscale, RGB, RGBA) by converting RGBA to RGB

    :param image_path: Path to the image file
    :type image_path: str
    :raises FileNotFoundError: If the image file does not exist
    :raises ValueError: If the image file is not a supported format or an error occurs during loading
    :return: A NumPy array representing the image (HxW for grayscale, HxWxC for color)
    :rtype: np.ndarray
    """
    try:
        img = iio.imread(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        # catch other potential imageio errors during loading
        raise ValueError(f"Error loading image: {e}") from e

    # ensure output image is uint8 [0, 255]
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            # common case for float images (often in range [0, 1])
            # clip to ensure values are within [0,1] before scaling
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            # for other integer types (like uint16), clip to 0-255 range and convert
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:  # grayscale image (H, W)
        return img
    elif img.ndim == 3 and img.shape[2] == 3:  # rgb image (H, W, 3)
        return img
    elif img.ndim == 3 and img.shape[2] == 4:  # rgba image (H, W, 4)
        # convert rgba to rgb by simply dropping the alpha channel
        # an alternative could be alpha blending onto a white background,
        # but dropping alpha is simpler and often sufficient if transparency isn't critical for the core logic
        img_rgb = img[:, :, :3]
        return img_rgb
    else:
        # if image dimensions or channels are not supported
        raise ValueError(
            f"Unsupported image format (dims/channels): {image_path}, shape={img.shape}, dtype={img.dtype}"
        )


def save_image(
    image: np.ndarray, output_path: str, filename: str, file_format: str = "png"
):
    """
    Saves a NumPy array as an image file using imageio

    Ensures the image is in uint8 format [0, 255] before saving
    Creates the output directory if it doesn't exist

    :param image: The NumPy array representing the image to save
    :type image: np.ndarray
    :param output_path: The directory where the image should be saved
    :type output_path: str
    :param filename: Base filename for saving (without extension)
    :type filename: str
    :param file_format: Image file format (default is "png")
    :type file_format: str
    :raises ValueError: If the image data is invalid (though primarily handled by imageio)
    :raises OSError: If there is an error writing the file
    """
    filepath = os.path.join(output_path, f"{filename}.{file_format}")
    os.makedirs(output_path, exist_ok=True)  # ensure directory exists

    # ensure image is in correct format (uint8, range 0-255) for saving
    processed_image = image  # start with a reference, modify if needed
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            # assume float image is in range [0, 1]
            processed_image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            # for other numeric types, clip and convert
            processed_image = np.clip(image, 0, 255).astype(np.uint8)
    else:  # if already uint8, ensure it's within range just in case
        if image.min() < 0 or image.max() > 255:
            # this might indicate an issue upstream if uint8 data is out of range
            processed_image = np.clip(image, 0, 255).astype(np.uint8)

    try:
        iio.imwrite(filepath, processed_image)
    except Exception as e:
        # catch potential imageio errors during saving
        raise OSError(f"Error saving image to {filepath}: {e}") from e


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB or RGBA image to grayscale using the luminosity method

    If the input image is already 2D (grayscale), it ensures it's uint8
    and returns it Otherwise, it converts 3D (RGB/RGBA) images

    The luminosity method coefficients are (0299*R + 0587*G + 0114*B)

    :param image: The input image (NumPy array)
    :type image: np.ndarray
    :raises ValueError: If image dimensions/channels are invalid for conversion
    :return: A grayscale version of the image (NumPy array, dtype=uint8)
    :rtype: np.ndarray
    """
    if image.ndim == 2:
        # if already grayscale, ensure dtype is uint8
        if image.dtype == np.uint8:
            return image
        else:
            # clip and convert if not uint8 (float grayscale)
            return np.clip(image, 0, 255).astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] == 3:  # RGB image
        # ensure input is uint8 before applying luminosity calculation for consistency
        img_to_convert = image
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                img_to_convert = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:  # other integer types
                img_to_convert = np.clip(image, 0, 255).astype(np.uint8)

        # luminosity method: Y = 0299R + 0587G + 0114B
        # use float coefficients for calculation, then convert back to uint8
        grayscale_image_float = (
            0.299 * img_to_convert[:, :, 0].astype(np.float32)
            + 0.587 * img_to_convert[:, :, 1].astype(np.float32)
            + 0.114 * img_to_convert[:, :, 2].astype(np.float32)
        )
        # clip result before casting to uint8 to ensure values are in [0, 255]
        return np.clip(grayscale_image_float, 0, 255).astype(np.uint8)
    elif image.ndim == 3 and image.shape[2] == 4:  # handle RGBA
        # convert to RGB first by dropping alpha (consistent with load_image)
        img_rgb = image[:, :, :3]
        # then convert the resulting RGB to grayscale recursively
        return convert_to_grayscale(img_rgb)
    else:
        # if image dimensions/channels are not suitable for grayscale conversion
        raise ValueError(
            f"Invalid image dimensions/channels for grayscale conversion: {image.shape}"
        )
