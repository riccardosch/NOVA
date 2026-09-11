import numpy as np


def center_prior_saliency(image_shape, sigma_frac=0.25):
    height, width = image_shape

    cx = width / 2.0
    cy = height / 2.0
    sigma_x = sigma_frac * width
    sigma_y = sigma_frac * height

    # Griglia di coordinate: xx[i,j] = j (colonna), yy[i,j] = i (riga).
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    exponent = -((xx - cx) ** 2 / (2.0 * sigma_x ** 2)
                 + (yy - cy) ** 2 / (2.0 * sigma_y ** 2))
    gauss = np.exp(exponent)

    return _normalize(gauss)


def _normalize(saliency_map):
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)
