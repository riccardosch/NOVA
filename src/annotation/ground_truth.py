"""
Conversione dei click in ground truth.

Servono due rappresentazioni, perche' metriche diverse le richiedono:
  - mappa di fissazione binaria -> NSS, AUC-Judd (ragionano su punti)
  - heatmap continua            -> CC, SIM, KL (confrontano distribuzioni)
E' la stessa doppia forma usata dai benchmark MIT/Tuebingen.
"""

import numpy as np


# sigma della Gaussiana come frazione della larghezza. Il 5% approssima
# l'ampiezza della fovea (~1 grado di angolo visivo): l'area effettivamente
# messa a fuoco da una singola fissazione.
DEFAULT_SIGMA_FRAC = 0.05


def clicks_to_fixation_map(clicks, image_size):
    """Mappa binaria: 1 sui pixel cliccati, 0 altrove."""
    height, width = image_size
    fixation_map = np.zeros((height, width), dtype=np.float32)

    for click in clicks:
        # clip: un click sul bordo estremo puo' dare una coordinata pari
        # alla larghezza, che come indice sarebbe fuori range.
        x = int(np.clip(click["x"], 0, width - 1))
        y = int(np.clip(click["y"], 0, height - 1))
        fixation_map[y, x] = 1.0

    return fixation_map


def clicks_to_heatmap(clicks, image_size, sigma_frac=DEFAULT_SIGMA_FRAC):
    """Heatmap continua: ogni click diventa una Gaussiana, tutte sommate."""
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    if not clicks:
        return heatmap

    sigma = sigma_frac * width
    # Griglia costruita una volta sola e riusata per ogni click.
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)

    for click in clicks:
        squared_distance = (x_grid - click["x"]) ** 2 + (y_grid - click["y"]) ** 2
        heatmap += np.exp(-squared_distance / (2.0 * sigma ** 2))

    max_value = heatmap.max()
    if max_value > 0:
        heatmap /= max_value
    return heatmap.astype(np.float32)


def aggregate_annotators_heatmap(annotators_clicks, image_size,
                                 sigma_frac=DEFAULT_SIGMA_FRAC):
    """Ground truth finale: i click di tutti in un'unica nuvola di punti.

    Le zone di consenso emergono da sole, perche' vi si sovrappongono piu'
    Gaussiane. Aggregare prima e normalizzare dopo (invece di mediare
    heatmap individuali) evita di amplificare chi ha fatto meno click.
    """
    all_clicks = []
    for clicks in annotators_clicks.values():
        all_clicks.extend(clicks)
    return clicks_to_heatmap(all_clicks, image_size, sigma_frac=sigma_frac)


def all_clicks_of(annotators_clicks):
    """Lista piatta dei click di tutti gli annotatori."""
    all_clicks = []
    for clicks in annotators_clicks.values():
        all_clicks.extend(clicks)
    return all_clicks
