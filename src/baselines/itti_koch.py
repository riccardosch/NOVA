"""
Itti-Koch (IEEE TPAMI, 1998): saliency da contrasto center-surround.

Un punto e' saliente se spicca rispetto a cio' che lo circonda. Il modello
misura questo contrasto su tre canali che il sistema visivo umano elabora in
parallelo nelle prime fasi della visione: intensita', colore (opponenza
rosso/verde e blu/giallo, come nella retina) e orientazione (filtri di
Gabor, che modellano le cellule della corteccia visiva primaria).

Semplificazioni rispetto al paper originale, dichiarate:
  1. Coppie di scale center-surround fisse invece di un set piu' ampio.
  2. Operatore di normalizzazione a singolo passaggio invece che iterativo.
Riducono il costo mantenendo il comportamento qualitativo del modello.
"""

import numpy as np
import cv2


def itti_koch_saliency(image, pyramid_levels=7):
    """Saliency map di Itti-Koch per un'immagine BGR.

    Parameters
    ----------
    image : np.ndarray, uint8, shape (h, w, 3)
        Immagine BGR (convenzione OpenCV).
    pyramid_levels : int
        Livelli della piramide Gaussiana: piu' livelli = si catturano
        contrasti a scale piu' grandi.

    Returns
    -------
    np.ndarray, float32, shape (h, w), valori in [0, 1].
    """
    img = image.astype(np.float32)
    b, g, r = cv2.split(img)

    # --- 1. Canali di feature ---------------------------------------------
    intensity = (r + g + b) / 3.0

    # Opponenza cromatica. La divisione per l'intensita' normalizza rispetto
    # all'illuminazione: lo stesso rosso in ombra o in luce ha la stessa
    # tinta. eps evita divisioni per zero nelle zone scure.
    eps = 1e-3
    rg = (r - g) / (intensity + eps)
    by = (b - np.minimum(r, g)) / (intensity + eps)

    # --- 2. Piramidi multi-scala ------------------------------------------
    intensity_pyr = _gaussian_pyramid(intensity, pyramid_levels)
    rg_pyr = _gaussian_pyramid(rg, pyramid_levels)
    by_pyr = _gaussian_pyramid(by, pyramid_levels)

    orientation_pyramids = []
    for theta_deg in (0, 45, 90, 135):
        kernel = cv2.getGaborKernel(
            ksize=(15, 15), sigma=4.0, theta=np.radians(theta_deg),
            lambd=10.0, gamma=0.5, psi=0,
        )
        orientation_pyramids.append(
            [cv2.filter2D(level, cv2.CV_32F, kernel) for level in intensity_pyr]
        )

    # --- 3. Center-surround + normalizzazione -----------------------------
    intensity_maps = [_normalize_map(m) for m in _center_surround(intensity_pyr)]

    color_maps = ([_normalize_map(m) for m in _center_surround(rg_pyr)]
                  + [_normalize_map(m) for m in _center_surround(by_pyr)])

    orientation_maps = []
    for pyr in orientation_pyramids:
        orientation_maps += [_normalize_map(m) for m in _center_surround(pyr)]

    # --- 4. Conspicuity map, una per canale -------------------------------
    target_shape = intensity_maps[0].shape

    def combine(maps):
        resized = [_resize_to(m, target_shape) for m in maps]
        return _normalize_map(np.sum(resized, axis=0))

    conspicuity_intensity = combine(intensity_maps)
    conspicuity_color = combine(color_maps)
    conspicuity_orientation = combine(orientation_maps)

    # --- 5. Media dei tre canali ------------------------------------------
    saliency = (conspicuity_intensity
                + conspicuity_color
                + conspicuity_orientation) / 3.0

    saliency = cv2.resize(saliency, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    minimum, maximum = saliency.min(), saliency.max()
    if maximum - minimum > 0:
        saliency = (saliency - minimum) / (maximum - minimum)
    else:
        saliency = np.zeros_like(saliency)
    return saliency.astype(np.float32)


def _gaussian_pyramid(channel, levels):
    """Lista di versioni via via dimezzate dello stesso canale.

    Serve a guardare la scena "a piu' distanze": un dettaglio fine e un
    oggetto grande producono contrasto a scale diverse. cv2.pyrDown sfoca
    prima di sottocampionare, evitando aliasing.
    """
    pyramid = [channel.astype(np.float32)]
    for _ in range(levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]).astype(np.float32))
    return pyramid


def _resize_to(feature_map, shape):
    """Riporta una mappa a (altezza, larghezza) indicate.

    Esiste per isolare in un punto solo l'inversione shape -> (w, h) che
    cv2.resize richiede.
    """
    return cv2.resize(feature_map, (shape[1], shape[0]),
                      interpolation=cv2.INTER_LINEAR)


def _center_surround(pyramid, center_scales=(2, 3), deltas=(3, 4)):
    """Differenze |scala fine - scala grossolana|.

    La scala fine ("center") vede il dettaglio, quella grossolana
    ("surround") il contesto attorno: la differenza in valore assoluto
    misura quanto il dettaglio si stacca dallo sfondo.
    """
    maps = []
    for c in center_scales:
        for delta in deltas:
            s = c + delta
            if s >= len(pyramid):
                continue
            center = pyramid[c]
            surround = _resize_to(pyramid[s], center.shape[:2])
            maps.append(np.abs(center - surround))
    return maps


def _normalize_map(feature_map, M=10.0):
    """Operatore N(.): premia le mappe con pochi picchi netti.

    Due mappe possono avere lo stesso massimo ma significato opposto: una
    con un picco isolato indica un elemento distintivo, una con venti picchi
    simili indica rumore. Si porta la mappa in [0, M] e la si scala per
    (M - media dei massimi locali)^2: con tanti picchi alti il fattore tende
    a zero e la mappa viene soppressa.
    """
    feature_map = feature_map - feature_map.min()
    if feature_map.max() > 0:
        feature_map = feature_map / feature_map.max() * M

    # La dilatazione sostituisce ogni pixel col massimo della sua vicinanza:
    # i pixel che restano uguali a se stessi sono i massimi locali.
    local_max = cv2.dilate(feature_map, np.ones((7, 7), np.uint8))
    is_local_max = (feature_map == local_max) & (feature_map < M - 1e-3)
    peaks = feature_map[is_local_max]
    mean_peak = peaks.mean() if peaks.size > 0 else 0.0

    return feature_map * (M - mean_peak) ** 2
