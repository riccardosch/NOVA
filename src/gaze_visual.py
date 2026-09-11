
import numpy as np
import cv2


def saliency_to_heatmap(saliency_map):
    # applyColorMap lavora su uint8. Il clip difende da modelli che
    # restituissero valori fuori range: senza, il cast farebbe wrap-around.
    saliency_uint8 = (np.clip(saliency_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(saliency_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_saliency(image_rgb, saliency_map, alpha=0.5):
    heatmap_rgb = saliency_to_heatmap(saliency_map)

    if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(heatmap_rgb,
                                 (image_rgb.shape[1], image_rgb.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

    return cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)


def mark_peak(image_rgb, saliency_map, radius=14, thickness=3):
    annotated = image_rgb.copy()   # cv2.circle disegna in-place
    py, px = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)

    cv2.circle(annotated, (int(px), int(py)), radius + 2, (0, 0, 0), thickness + 2)
    cv2.circle(annotated, (int(px), int(py)), radius, (255, 255, 255), thickness)

    return annotated, (int(px), int(py))
