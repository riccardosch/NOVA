"""
MSI-Net — Multi-Scale Information Network (Kroner et al., 2020)

Wrapper per il modello di saliency pre-addestrato, caricato da Kaggle Hub
tramite tensorflow_hub.

Installazione:
    pip install tensorflow tensorflow_hub

Riferimento originale:
    A. Kroner, M. Senden, K. Driessens, R. Goebel,
    "Contextual Encoder-Decoder Network for Visual Saliency Prediction",
    Neural Networks, 2020.
"""

import numpy as np
import cv2

# --------------------------------------------------------------------------- #
#  Risoluzione fissa di addestramento di MSI-Net (altezza × larghezza)
# --------------------------------------------------------------------------- #
_INPUT_H, _INPUT_W = 240, 320

# --------------------------------------------------------------------------- #
#  Cache del modello a livello di modulo: caricato una sola volta
# --------------------------------------------------------------------------- #
_model = None


def is_available() -> bool:
    """Ritorna True se tensorflow e tensorflow_hub sono installati."""
    try:
        import tensorflow          # noqa: F401
        import tensorflow_hub      # noqa: F401
        return True
    except ImportError:
        return False


def _check_deps():
    """Solleva ImportError se le dipendenze mancano."""
    if not is_available():
        raise ImportError(
            "MSI-Net richiede: pip install tensorflow tensorflow_hub"
        )


def _load_model():
    """Carica il modello da Kaggle Hub via tensorflow_hub (una sola volta)."""
    global _model
    if _model is None:
        _check_deps()
        import tensorflow_hub as hub
        _model = hub.load(
            "https://www.kaggle.com/models/alexanderkroner/msi-net/tensorFlow2/salicon/1"
        ).signatures["serving_default"]
    return _model


def _normalize(saliency_map):
    """Riscala una mappa in [0, 1] (min-max)."""
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)


def msinet_saliency(image_bgr):
    """
    Calcola la saliency map con MSI-Net.

    Parameters
    ----------
    image_bgr : np.ndarray, uint8, shape (h, w, 3)

    Returns
    -------
    np.ndarray, float32, shape (h, w), valori in [0, 1]
    """
    _check_deps()
    model = _load_model()

    # Dimensioni originali
    h, w = image_bgr.shape[:2]

    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Ridimensiona alla risoluzione fissa di addestramento (240×320)
    image_resized = cv2.resize(image_rgb, (_INPUT_W, _INPUT_H))

    # Normalizza in [0, 1] e aggiungi dimensione di batch: (1, 240, 320, 3)
    tensor = image_resized.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, ...]

    # Inferenza — il modello è una signature TF, richiede tf.constant
    import tensorflow as tf
    tensor = tf.constant(tensor)
    outputs = model(tensor)

    # La signature restituisce un dict di tensori; estraiamo l'unico valore
    if isinstance(outputs, dict):
        output = list(outputs.values())[0].numpy()
    else:
        output = outputs.numpy()

    # Rimuovi dimensioni di batch/canale
    saliency = np.squeeze(output)

    # Ridimensiona alle dimensioni originali dell'immagine
    saliency = cv2.resize(saliency, (w, h))

    # Normalizzazione min-max in [0, 1]
    saliency = _normalize(saliency)

    return saliency
