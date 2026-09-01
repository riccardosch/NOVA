"""
Visualizzazione delle saliency map.

Separato dall'app perche' serve anche per generare le figure della
documentazione e per l'ispezione durante lo sviluppo.
"""

import numpy as np
import cv2


def saliency_to_heatmap(saliency_map):
    """Saliency [0,1] -> immagine RGB colorata (colormap JET)."""
    # applyColorMap lavora su uint8. Il clip difende da modelli che
    # restituissero valori fuori range: senza, il cast farebbe wrap-around.
    saliency_uint8 = (np.clip(saliency_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_saliency(image_rgb, saliency_map, alpha=0.5):
    """Sovrappone la heatmap all'immagine originale.

    addWeighted calcola image*(1-alpha) + heatmap*alpha.
    """
    heatmap_rgb = saliency_to_heatmap(saliency_map)

    if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb,
                                 (image_rgb.shape[1], image_rgb.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

    return cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)


def mark_peak(image_rgb, saliency_map, radius=14, thickness=3):
    """Cerchio sul punto di massima saliency.

    Restituisce (immagine annotata, (x, y) del picco). Due cerchi
    concentrici, nero sotto e bianco sopra, per restare visibile su
    qualunque sfondo.
    """
    annotated = image_rgb.copy()   # cv2.circle disegna in-place
    py, px = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)

    cv2.circle(annotated, (int(px), int(py)), radius + 2, (0, 0, 0), thickness + 2)
    cv2.circle(annotated, (int(px), int(py)), radius, (255, 255, 255), thickness)

    return annotated, (int(px), int(py))
