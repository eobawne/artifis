import numpy as np
from numba import njit, prange, int32

# constants for normalization of metrics to [0, 1] range where 1 is better
# these values are chosen empirically to map typical metric ranges to a sensitive part of the exponential curve
_K_MSE = np.float32(0.0005)  # for mse, lower is better, so exp(-k*mse)
_K_PSNR = np.float32(0.1)  # for psnr, higher is better, so 1 - exp(-k*psnr)
_K_GMSD = np.float32(15.0)  # for gmsd, lower is better, so exp(-k*gmsd)


@njit(fastmath=True)
def _mse_original(image1: np.ndarray, image2: np.ndarray) -> np.float32:
    """
    Calculates the original Mean Squared Error (MSE) between two images

    Lower MSE values indicate better similarity

    :param image1: The first image (NumPy array)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array), must have same shape and type as image1
    :type image2: np.ndarray
    :return: The mean squared error value
    :rtype: np.float32
    """
    # ensure float32 for calculation to avoid overflow/precision issues with uint8
    img1_f = image1.astype(np.float32)
    img2_f = image2.astype(np.float32)
    err = np.sum((img1_f - img2_f) ** 2)
    err /= np.float32(image1.size)  # divide by total number of pixels
    return np.float32(err)


@njit(fastmath=True)
def _psnr_original(image1: np.ndarray, image2: np.ndarray) -> np.float32:
    """
    Calculates the original Peak Signal-to-Noise Ratio (PSNR) between two images

    Higher PSNR values indicate better similarity Assumes pixel values are in [0, 255]

    :param image1: The first image (NumPy array)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array)
    :type image2: np.ndarray
    :return: The PSNR value, or np.inf if MSE is zero, or 0.0 if PSNR is negative
    :rtype: np.float32
    """
    mse_val = _mse_original(image1.astype(np.float32), image2.astype(np.float32))
    epsilon = np.float32(1e-10)  # small value to avoid division by zero
    if mse_val <= epsilon:  # if images are identical (or very close)
        return np.inf  # PSNR is infinite
    max_pixel_val = np.float32(255.0)
    psnr_val = np.float32(20.0) * np.log10(max_pixel_val) - np.float32(10.0) * np.log10(
        mse_val
    )
    # psnr can technically be negative if mse is very large, clip to 0 for practical purposes
    return psnr_val if psnr_val >= 0.0 else np.float32(0.0)


@njit(fastmath=True, parallel=True)  # parallel=True enables parallel execution of loops
def _ssim_channel(
    img1: np.ndarray,  # single channel image (HxW)
    img2: np.ndarray,  # single channel image (HxW)
    win_size: int = 7,  # window size for local statistics
    K1: float = 0.01,  # ssim constant
    K2: float = 0.03,  # ssim constant
    L: float = 255.0,  # dynamic range of pixel values (255 for uint8)
) -> np.ndarray:  # returns HxW map of ssim values
    """
    Computes SSIM for a single channel of an image using a sliding window approach

    This is a helper for the main SSIM function

    :param img1: First single-channel image (NumPy array, HxW)
    :type img1: np.ndarray
    :param img2: Second single-channel image (NumPy array, HxW)
    :type img2: np.ndarray
    :param win_size: Size of the N_x_N sliding window
    :type win_size: int
    :param K1: SSIM algorithm constant
    :type K1: float
    :param K2: SSIM algorithm constant
    :type K2: float
    :param L: Dynamic range of pixel values (default 255 for uint8)
    :type L: float
    :return: A map of SSIM values for each pixel (NumPy array, HxW)
    :rtype: np.ndarray
    """
    C1 = np.float32((K1 * L) ** 2)  # stability constant for luminance
    C2 = np.float32((K2 * L) ** 2)  # stability constant for contrast
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)

    # preallocate arrays for local means, variances, and covariance
    mu1 = np.zeros_like(img1_f, dtype=np.float32)
    mu2 = np.zeros_like(img1_f, dtype=np.float32)
    sigma1_sq = np.zeros_like(img1_f, dtype=np.float32)
    sigma2_sq = np.zeros_like(img1_f, dtype=np.float32)
    sigma12 = np.zeros_like(img1_f, dtype=np.float32)

    pad = int32(win_size // 2)  # padding for window operations
    H, W = img1_f.shape

    # calculate local statistics using sliding window
    # prange enables parallel execution of this outer loop by numba
    for y in prange(pad, H - pad):
        for x in range(pad, W - pad):
            window1 = img1_f[y - pad : y + pad + 1, x - pad : x + pad + 1]
            window2 = img2_f[y - pad : y + pad + 1, x - pad : x + pad + 1]
            mu1_local = window1.mean()
            mu2_local = window2.mean()
            mu1[y, x] = mu1_local
            mu2[y, x] = mu2_local
            sigma1_sq[y, x] = window1.var()
            sigma2_sq[y, x] = window2.var()
            # covariance calculation
            sigma12[y, x] = np.mean((window1 - mu1_local) * (window2 - mu2_local))

    # ssim formula components
    num = (2.0 * mu1 * mu2 + C1) * (2.0 * sigma12 + C2)  # numerator
    den = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)  # denominator
    ssim_map = np.zeros_like(den, dtype=np.float32)
    epsilon = np.float32(1e-8)  # for numerical stability

    # calculate ssim value for each pixel
    for i in range(H):
        for j in range(W):
            den_val = den[i, j]
            num_val = num[i, j]
            if den_val > epsilon:  # avoid division by zero
                ssim_val = num_val / den_val
                # ssim values are theoretically [-1, 1], clip for robustness
                ssim_map[i, j] = max(np.float32(-1.0), min(np.float32(1.0), ssim_val))
            else:
                # handle case where denominator is zero (or very small)
                # if numerator is also zero, regions are identical (ssim=1)
                # otherwise, implies significant difference (ssim=0 or other value if num is also small)
                if abs(num_val) < epsilon:  # both num and den are zero
                    ssim_map[i, j] = np.float32(1.0)
                else:  # den is zero, num is not
                    ssim_map[i, j] = np.float32(0.0)  # or some other appropriate value

    return ssim_map


@njit(fastmath=True)
def _sobel_magnitude(image_channel: np.ndarray) -> np.ndarray:  # HxW single channel
    """
    Computes Sobel gradient magnitude for a single image channel

    Used as a step in GMSD calculation

    :param image_channel: Single-channel image (NumPy array, HxW)
    :type image_channel: np.ndarray
    :return: Gradient magnitude map (NumPy array, HxW)
    :rtype: np.ndarray
    """
    img = image_channel.astype(np.float32)
    H, W = img.shape
    Gx = np.zeros_like(img)  # gradient in x
    Gy = np.zeros_like(img)  # gradient in y
    # sobel kernels
    kernel_x = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32
    )
    kernel_y = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32
    )
    # convolve image with sobel kernels
    for y in range(1, H - 1):  # iterate avoiding borders
        for x in range(1, W - 1):
            sub_img = img[y - 1 : y + 2, x - 1 : x + 2]  # 3x3 window
            Gx[y, x] = np.sum(sub_img * kernel_x)
            Gy[y, x] = np.sum(sub_img * kernel_y)
    magnitude = np.sqrt(Gx * Gx + Gy * Gy)
    return magnitude


@njit(fastmath=True)
def _gms_map(
    img1_ch: np.ndarray, img2_ch: np.ndarray
) -> np.ndarray:  # HxW single channels
    """
    Computes Gradient Magnitude Similarity (GMS) map between two single-channel images

    Helper for GMSD calculation

    :param img1_ch: First single-channel image (NumPy array, HxW)
    :type img1_ch: np.ndarray
    :param img2_ch: Second single-channel image (NumPy array, HxW)
    :type img2_ch: np.ndarray
    :return: GMS map (NumPy array, HxW), values in [0,1]
    :rtype: np.ndarray
    """
    gm1 = _sobel_magnitude(img1_ch)  # gradient magnitude of first image
    gm2 = _sobel_magnitude(img2_ch)  # gradient magnitude of second image
    c = np.float32(170.0)  # constant from GMSD paper, for stability
    # gms formula
    gms_map_calc = (2.0 * gm1 * gm2 + c) / (gm1 * gm1 + gm2 * gm2 + c)
    H, W = gms_map_calc.shape
    # clip values to [0,1] for robustness, although theoretically they should be
    for i in range(H):
        for j in range(W):
            gms_map_calc[i, j] = max(
                np.float32(0.0), min(np.float32(1.0), gms_map_calc[i, j])
            )
    return gms_map_calc


@njit(fastmath=True)
def _gmsd_original_channel(img1_ch: np.ndarray, img2_ch: np.ndarray) -> np.float32:
    """
    Calculates Gradient Magnitude Similarity Deviation (GMSD) for a single channel

    Lower GMSD values indicate better similarity (less deviation in gradient similarity)

    :param img1_ch: First single-channel image (NumPy array, HxW)
    :type img1_ch: np.ndarray
    :param img2_ch: Second single-channel image (NumPy array, HxW)
    :type img2_ch: np.ndarray
    :return: GMSD value for the channel
    :rtype: np.float32
    """
    gms = _gms_map(img1_ch, img2_ch)  # get gradient magnitude similarity map
    mean_gms = np.mean(gms)
    # variance of gms map (standard deviation is sqrt of variance)
    var_gms = np.mean((gms - mean_gms) ** 2)
    var_gms = max(np.float32(0.0), var_gms)  # ensure non-negative variance
    std_dev_gms = np.sqrt(var_gms)
    return std_dev_gms


# public normalized metric functions ([0, 1], 1=best)
@njit(fastmath=True)
def mse(image1: np.ndarray, image2: np.ndarray) -> np.float32:
    """
    Calculates normalized Mean Squared Error (MSE)

    Normalized to [0, 1] where 1 is best (perfect match)
    Uses exponential decay: exp(-K_MSE * original_mse)

    :param image1: The first image (NumPy array)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array)
    :type image2: np.ndarray
    :return: Normalized MSE score
    :rtype: np.float32
    """
    mse_val = _mse_original(image1, image2)
    mse_val = max(np.float32(0.0), mse_val)  # ensure non-negative before exp
    return np.float32(np.exp(-_K_MSE * mse_val))


@njit(fastmath=True)
def psnr(image1: np.ndarray, image2: np.ndarray) -> np.float32:
    """
    Calculates normalized Peak Signal-to-Noise Ratio (PSNR)

    Normalized to [0, 1] where 1 is best (perfect match or infinite PSNR)
    Uses: 1 - exp(-K_PSNR * original_psnr)

    :param image1: The first image (NumPy array)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array)
    :type image2: np.ndarray
    :return: Normalized PSNR score
    :rtype: np.float32
    """
    psnr_val = _psnr_original(image1, image2)
    if np.isinf(psnr_val):  # if original psnr is infinite (mse=0)
        return np.float32(1.0)  # perfect score
    psnr_val = max(np.float32(0.0), psnr_val)  # ensure non-negative
    # for psnr, higher is better, so we want 1 - decay
    return np.float32(np.float32(1.0) - np.exp(-_K_PSNR * psnr_val))


# this non-jitted version handles type checking and channel dispatch for ssim
# it calls the jitted _ssim_channel
def _ssim_original(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculates original Structural Similarity Index (SSIM) between two images

    SSIM values range from -1 to 1, where 1 indicates perfect similarity
    This function handles both grayscale and 3-channel (RGB) images

    :param image1: The first image (NumPy array, HxW or HxWxC)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array, HxW or HxWxC)
    :type image2: np.ndarray
    :raises ValueError: If input images have mismatched shapes/dims or unsupported channel counts
    :raises TypeError: If input image dtypes are not convertible to uint8
    :return: The mean SSIM value
    :rtype: float
    """
    if image1.shape != image2.shape or image1.ndim not in [2, 3]:
        raise ValueError("SSIM: Input images must have the same shape and be 2D or 3D")
    try:
        # ensure images are uint8 for _ssim_channel, which expects typical image range
        img1_u8 = (
            image1
            if image1.dtype == np.uint8
            else np.clip(image1, 0, 255).astype(np.uint8)
        )
        img2_u8 = (
            image2
            if image2.dtype == np.uint8
            else np.clip(image2, 0, 255).astype(np.uint8)
        )
    except Exception:  # catch general conversion errors
        raise TypeError("SSIM expects uint8 images or types convertible to uint8")

    if img1_u8.ndim == 2:  # grayscale image
        ssim_map = _ssim_channel(img1_u8, img2_u8)
        mean_ssim_val = np.mean(ssim_map).astype(np.float32)
    elif img1_u8.ndim == 3:  # color image
        mean_ssim_channels_val = np.float32(0.0)
        n_channels = img1_u8.shape[2]
        if n_channels != 3:  # ssim typically expects 3 channels for color (rgb)
            raise ValueError("SSIM: 3D image must have 3 channels (RGB)")
        # calculate ssim for each channel and average
        for i in range(n_channels):
            ssim_map_channel = _ssim_channel(img1_u8[:, :, i], img2_u8[:, :, i])
            mean_ssim_channels_val += np.mean(ssim_map_channel).astype(np.float32)
        mean_ssim_val = mean_ssim_channels_val / np.float32(n_channels)
    else:  # should be caught by initial check, but as a safeguard
        raise ValueError(f"SSIM: Unsupported image dimensions: {img1_u8.ndim}")
    # ensure result is within theoretical bounds of ssim
    return float(max(np.float32(-1.0), min(np.float32(1.0), mean_ssim_val)))


@njit(fastmath=True, parallel=True)  # parallel for channel loop if applicable
def ssim(image1: np.ndarray, image2: np.ndarray) -> np.float32:
    """
    Calculates normalized Structural Similarity Index (SSIM)

    Original SSIM is [-1, 1] Normalized to [0, 1] where 1 is best
    Normalization: (original_ssim + 1) / 2

    :param image1: The first image (NumPy array)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array)
    :type image2: np.ndarray
    :return: Normalized SSIM score
    :rtype: np.float32
    """
    if image1.shape != image2.shape or image1.ndim not in [2, 3]:
        return np.float32(0.0)  # return 0 for invalid input
    mean_ssim_val: np.float32 = 0.0
    if image1.ndim == 2:  # grayscale
        # ensure float32 for _ssim_channel if it expects floats, though it casts internally
        img1_f32 = image1.astype(np.float32)
        img2_f32 = image2.astype(np.float32)
        ssim_map = _ssim_channel(img1_f32, img2_f32)
        mean_ssim_val = np.mean(ssim_map)
    elif image1.ndim == 3:  # color
        mean_ssim_channels_val = np.float32(0.0)
        n_channels = image1.shape[2]
        if n_channels != 3:  # only support 3-channel for direct ssim calculation here
            return np.float32(0.0)

        # prange could be used here if _ssim_channel itself wasn't parallel,
        # but _ssim_channel is already parallel, so simple loop is fine
        for i in range(n_channels):
            img1_ch_f32 = image1[:, :, i].astype(np.float32)
            img2_ch_f32 = image2[:, :, i].astype(np.float32)
            ssim_map_channel = _ssim_channel(img1_ch_f32, img2_ch_f32)
            mean_ssim_channels_val += np.mean(ssim_map_channel)
        mean_ssim_val = mean_ssim_channels_val / np.float32(n_channels)
    # normalize ssim from [-1, 1] to [0, 1]
    normalized_val = (mean_ssim_val + np.float32(1.0)) / np.float32(2.0)
    return max(np.float32(0.0), min(np.float32(1.0), normalized_val))  # clip to [0,1]


# non-jitted version for gmsd, similar to ssim, handles type checks and channel dispatch
def _gmsd_original(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculates original Gradient Magnitude Similarity Deviation (GMSD)

    Lower GMSD values indicate better similarity Handles grayscale and 3-channel (RGB) images

    :param image1: The first image (NumPy array, HxW or HxWxC)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array, HxW or HxWxC)
    :type image2: np.ndarray
    :raises ValueError: If input images have mismatched shapes/dims or unsupported channel counts
    :raises TypeError: If input image dtypes are not convertible to uint8
    :return: The mean GMSD value
    :rtype: float
    """
    if image1.shape != image2.shape or image1.ndim not in [2, 3]:
        raise ValueError("GMSD: Input images must have the same shape and be 2D or 3D")
    try:
        # ensure uint8 for consistency with how _gmsd_original_channel might be used
        img1_u8 = (
            image1
            if image1.dtype == np.uint8
            else np.clip(image1, 0, 255).astype(np.uint8)
        )
        img2_u8 = (
            image2
            if image2.dtype == np.uint8
            else np.clip(image2, 0, 255).astype(np.uint8)
        )
    except Exception:
        raise TypeError("GMSD expects uint8 images or types convertible to uint8")

    gmsd_val_calc: np.float32 = 0.0
    if img1_u8.ndim == 2:  # grayscale
        gmsd_val_calc = _gmsd_original_channel(img1_u8, img2_u8)
    elif img1_u8.ndim == 3:  # color
        gmsd_sum_val = np.float32(0.0)
        n_channels = img1_u8.shape[2]
        if n_channels != 3:  # expect 3 channels for rgb
            raise ValueError("GMSD: 3D image must have 3 channels")
        # calculate gmsd for each channel and average
        for i in range(n_channels):
            gmsd_sum_val += _gmsd_original_channel(img1_u8[:, :, i], img2_u8[:, :, i])
        gmsd_val_calc = gmsd_sum_val / np.float32(n_channels)
    else:  # safeguard
        raise ValueError(f"GMSD: Unsupported image dimensions: {img1_u8.ndim}")
    # gmsd is non-negative
    return float(max(np.float32(0.0), gmsd_val_calc))


@njit(fastmath=True)  # parallel not beneficial here as _gmsd_original_channel is serial
def gmsd(image1: np.ndarray, image2: np.ndarray) -> np.float32:
    """
    Calculates normalized Gradient Magnitude Similarity Deviation (GMSD)

    Normalized to [0, 1] where 1 is best (perfect match, original_gmsd = 0)
    Uses exponential decay: exp(-K_GMSD * original_gmsd)

    :param image1: The first image (NumPy array)
    :type image1: np.ndarray
    :param image2: The second image (NumPy array)
    :type image2: np.ndarray
    :return: Normalized GMSD score
    :rtype: np.float32
    """
    if image1.shape != image2.shape or image1.ndim not in [2, 3]:
        return np.float32(0.0)  # invalid input
    gmsd_val_orig_calc: np.float32 = 0.0
    if image1.ndim == 2:  # grayscale
        gmsd_val_orig_calc = _gmsd_original_channel(image1, image2)
    elif image1.ndim == 3:  # color
        gmsd_sum_val = np.float32(0.0)
        n_channels = image1.shape[2]
        if n_channels != 3:  # only support 3-channel
            return np.float32(0.0)

        for i in range(n_channels):  # average over channels
            gmsd_sum_val += _gmsd_original_channel(image1[:, :, i], image2[:, :, i])
        gmsd_val_orig_calc = gmsd_sum_val / np.float32(n_channels)
    else:  # unsupported dims
        return np.float32(0.0)

    gmsd_val_orig_calc = max(np.float32(0.0), gmsd_val_orig_calc)  # ensure non-negative
    # normalize using exponential decay
    normalized_val = np.float32(np.exp(-_K_GMSD * gmsd_val_orig_calc))
    return max(np.float32(0.0), min(np.float32(1.0), normalized_val))  # clip to [0,1]


# dictionary mapping metric names to their (normalized) function implementations
METRIC_FUNCTIONS = {
    "mse": mse,
    "psnr": psnr,
    "ssim": ssim,
    "gmsd": gmsd,
}
AVAILABLE_METRICS = list(METRIC_FUNCTIONS.keys())  # list of available metric names
# dictionary for original (non-normalized) metric functions, used for final reporting
ORIGINAL_METRIC_FUNCTIONS = {
    "mse": _mse_original,
    "psnr": _psnr_original,
    "ssim": _ssim_original,
    "gmsd": _gmsd_original,
}

# this print statement executes when the module is imported
print(f"Available metrics: {AVAILABLE_METRICS}")
