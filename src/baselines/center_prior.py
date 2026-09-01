"""
Center prior: baseline che ignora il contenuto dell'immagine.

Genera una Gaussiana 2D fissa al centro. Serve come termine minimo di
paragone: le persone guardano il centro di una scena piu' della periferia
(center bias), quindi un modello che non batte questa mappa non sta
estraendo informazione dall'immagine.
"""

import numpy as np


def center_prior_saliency(image_shape, sigma_frac=0.25):
    """Gaussiana 2D centrata sull'immagine.

    Parameters
    ----------
    image_shape : tuple(int, int)
        (altezza, larghezza). Si passa la forma, non l'immagine: il
        contenuto e' irrilevante per questo modello.
    sigma_frac : float
        Deviazione standard come frazione delle dimensioni. Proporzionale e
        non fissa in pixel, altrimenti la macchia avrebbe forma diversa su
        immagini non quadrate o a risoluzioni diverse.

    Returns
    -------
    np.ndarray, float32, shape (altezza, larghezza), valori in [0, 1].
    """
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
    """Riscala una mappa in [0, 1]."""
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)
