import imageio
import os
import numpy as np


def create_gif(images, output_path, filename, loop=0):
    """
    Creates a GIF from a list of images

    The images are expected to be NumPy arrays If they are not uint8,
    they will be processed (float [0,1] scaled to [0,255], other types clipped)
    to uint8 before being added to the GIF

    :param images: List of images (NumPy arrays)
    :type images: list[np.ndarray]
    :param output_path: Directory where the GIF will be saved
    :type output_path: str
    :param filename: Base filename for the GIF (without .gif extension)
    :type filename: str
    :param loop: Number of times the GIF should loop 0 means infinite loop
    :type loop: int
    """
    gif_path = os.path.join(output_path, f"{filename}.gif")
    # ensure output directory exists, create it if not
    os.makedirs(output_path, exist_ok=True)

    # imageio writer for gifs typically expects uint8 images
    # this loop processes images to ensure they are in the correct format
    processed_images = []
    for img in images:
        if img.dtype == np.uint8:
            processed_images.append(img)
        elif np.issubdtype(img.dtype, np.floating):
            # clip to handle potential minor out-of-range values, scale, then convert
            img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            processed_images.append(img_u8)
        else:
            # for other numeric types (int16, int32),
            # clip to the uint8 range [0, 255] and convert
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            processed_images.append(img_u8)

    if not processed_images:
        # if no valid images could be processed, don't attempt to create a gif
        print(f"Warning: No valid images provided to create GIF: {gif_path}")
        return

    try:
        # use imageio's get_writer context manager to handle file opening/closing
        # mode 'I' is for multiple images (animated gif)
        # duration is in milliseconds per frame (100ms = 10fps)
        with imageio.get_writer(gif_path, mode="I", duration=100, loop=loop) as writer:
            for image in processed_images:
                writer.append_data(image)
        print(f"GIF created: {gif_path}")
    except Exception as e:
        # catch any errors during gif writing process
        print(f"Error creating GIF {gif_path}: {e}")
