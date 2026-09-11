# osie_ground_truth.py — Utilities for loading OSIE fixation data
# and converting fixation points into Gaussian heatmaps.

import numpy as np        # Numerical operations (arrays, math)
import scipy.io as sio    # Loading MATLAB .mat files


# Fraction of image width used as the Gaussian sigma.
# Same value used in collect.py for consistency.
SIGMA_FRAC = 0.05


def load_fixations(mat_path):
    """Load fixation data from the OSIE MATLAB file.

    Returns a dict mapping image filenames to lists of (x, y) fixation points
    aggregated across all subjects.
    """
    data = sio.loadmat(mat_path)
    entries = data["fixations"]  # Structured array from the .mat file

    result = {}
    for i in range(entries.shape[0]):
        # Each entry contains per-image fixation data
        record = entries[i, 0][0, 0]
        image_name = str(record["img"][0])  # Extract the image filename

        points = []
        subjects = record["subjects"]  # One sub-array per human subject
        for s in range(subjects.shape[0]):
            subject_data = subjects[s, 0][0, 0]
            # Extract the x and y fixation coordinate arrays for this subject
            xs = np.asarray(subject_data["fix_x"]).ravel()
            ys = np.asarray(subject_data["fix_y"]).ravel()
            # Merge all subjects' fixations into a single list
            points.extend(zip(xs.tolist(), ys.tolist()))

        result[image_name] = points

    return result


def points_to_heatmap(points, image_shape, sigma_frac=SIGMA_FRAC):
    """Convert a list of (x, y) fixation points into a normalized heatmap.

    Each point contributes a 2-D Gaussian blob.  The resulting heatmap
    is normalized so that the maximum value is 1.0.
    """
    height, width = image_shape
    # Start with an all-zeros heatmap at the given resolution
    heatmap = np.zeros((height, width), dtype=np.float32)
    if not points:
        return heatmap

    # Sigma is proportional to the image width
    sigma = sigma_frac * width
    # Pre-compute coordinate grids for efficient vectorized Gaussian evaluation
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)

    for x, y in points:
        # Compute the squared Euclidean distance from every pixel to (x, y)
        squared_distance = (x_grid - x) ** 2 + (y_grid - y) ** 2
        # Accumulate the Gaussian blob for this fixation point
        heatmap += np.exp(-squared_distance / (2.0 * sigma ** 2))

    # Normalize so the peak value equals 1.0
    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap
