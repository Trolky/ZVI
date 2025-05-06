import numpy as np
import cv2
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter


def manual_threshold(image, threshold_value):
    """
    Apply manual thresholding to an image using NumPy.

    Args:
        image: Grayscale input image
        threshold_value: Threshold value (0-255)

    Returns:
        Binary image
    """
    # Create an output array of zeros (same shape as input)
    binary = np.zeros_like(image)

    # Set pixels above threshold to 255 (white)
    binary[image > threshold_value] = 255

    return binary


def otsu_threshold(image):
    """
    Apply Otsu thresholding to an image using NumPy for histogram calculation.

    Args:
        image: Grayscale input image (NumPy array, expected to be 8-bit, 0-255)

    Returns:
        Binary image and the computed threshold value
    """
    # Ensure image is a NumPy array and is 2D (grayscale)
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    if image.dtype != np.uint8:
        # Or convert: image = image.astype(np.uint8) if conversion is acceptable
        print("Warning: Otsu's method is typically applied to 8-bit images. Ensure pixel values are in [0, 255].")

    # Calculate histogram using NumPy
    hist = np.bincount(image.ravel(), minlength=256)

    # Convert hist to float for division
    hist = hist.astype(float)

    # Normalize histogram to get probabilities
    pixel_count = image.shape[0] * image.shape[1]
    if pixel_count == 0:  # Handle empty image case
        return np.zeros_like(image), 0

    hist = hist / pixel_count

    # Compute cumulative sums
    cumsum = np.cumsum(hist)

    # Compute cumulative means
    indices = np.arange(256)
    cum_means = np.cumsum(hist * indices)

    # Compute global mean
    global_mean = cum_means[-1]

    # Initialize variables for maximum variance and optimal threshold
    max_variance = 0.0
    optimal_threshold = 0

    # Compute between-class variance for each possible threshold
    for t in range(1, 256):
        # Weight for background class
        w_bg = cumsum[t - 1]
        if w_bg == 0:
            continue

        # Weight for foreground class
        w_fg = 1.0 - w_bg
        if w_fg == 0:
            break

        # Mean for background class
        mean_bg = cum_means[t - 1] / w_bg

        # Mean for foreground class
        mean_fg = (global_mean - cum_means[t - 1]) / w_fg

        # Compute between-class variance
        variance = w_bg * w_fg * ((mean_bg - mean_fg) ** 2)

        # Update optimal threshold if current variance is higher
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t

    # Apply threshold
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[image >= optimal_threshold] = 255

    return binary, optimal_threshold


def su_local_max_min_threshold(image, window_size=15, k=0.5, R=128):
    """
    Apply SU_local_max_min thresholding to an image.
    This method uses local maximum and minimum values within a window to determine the threshold.

    Args:
        image: Grayscale input image
        window_size: Size of the local window
        k: Weight parameter, usually between 0.2 and 0.8
        R: Normalization factor, typically 128 for 8-bit images

    Returns:
        Binary image
    """
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1

    # Calculate local maximum and minimum using filters
    local_max = maximum_filter(image, size=window_size)
    local_min = minimum_filter(image, size=window_size)

    # Calculate local contrast
    local_contrast = local_max - local_min

    # Calculate threshold
    threshold = k * local_max + (1 - k) * local_min

    # Apply normalization if contrast is low
    # If contrast is below R, adjust threshold
    low_contrast_mask = local_contrast < R
    if np.any(low_contrast_mask):
        # For low contrast regions, use a weighted average
        mean_value = uniform_filter(image.astype(float), size=window_size)
        threshold[low_contrast_mask] = mean_value[low_contrast_mask]

    # Apply threshold
    binary = np.zeros_like(image)
    binary[image > threshold] = 255

    return binary


def recursive_otsu_threshold(image, max_depth=3):
    """
    Apply Recursive Otsu thresholding to an image.

    Args:
        image: Grayscale input image
        max_depth: Maximum recursion depth

    Returns:
        Binary image
    """

    def recursive_otsu(img, depth=0):
        if depth >= max_depth:
            return img

        # Apply Otsu thresholding
        thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Create binary image
        binary = np.zeros_like(img)
        binary[img > thresh] = 255

        # Find foreground and background pixels
        foreground = img.copy()
        foreground[img <= thresh] = 0

        background = img.copy()
        background[img > thresh] = 0

        # Recursively apply Otsu to foreground and background
        if depth < max_depth - 1:
            # Only process non-empty regions
            if np.count_nonzero(foreground) > 0:
                foreground_thresh = recursive_otsu(foreground, depth + 1)
                binary[foreground_thresh > 0] = 255

            if np.count_nonzero(background) > 0:
                background_thresh = recursive_otsu(background, depth + 1)
                binary[background_thresh > 0] = 255

        return binary

    return recursive_otsu(image)