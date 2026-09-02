"""
Registro centrale dei modelli.

I modelli hanno firme diverse: il center prior vuole la FORMA
dell'immagine, gli altri l'immagine stessa. Senza un livello di
uniformazione ogni consumatore (app, script di valutazione) dovrebbe
ripetere lo stesso if/else.

Aggiungere un modello significa modificare QUESTO file e nient'altro.
"""

from src.baselines.center_prior import center_prior_saliency
from src.baselines.spectral_residual import spectral_residual_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.models.msi_net import msinet_saliency
from src.models.deepgaze import deepgaze_saliency


# nome -> (funzione, vuole_immagine_intera)
_REGISTRY = {
    "Center Prior": (center_prior_saliency, False),
    "Spectral Residual": (spectral_residual_saliency, True),
    "Itti-Koch": (itti_koch_saliency, True),
    "MSI-Net": (msinet_saliency, True),
    "DeepGaze IIE": (deepgaze_saliency, True),
}

AVAILABLE_MODELS = list(_REGISTRY.keys())


def compute_saliency(model_name, image_bgr):
    """Calcola la saliency map col modello richiesto.

    Parameters
    ----------
    model_name : str
        Uno dei nomi in AVAILABLE_MODELS.
    image_bgr : np.ndarray, uint8, shape (h, w, 3)
        Immagine BGR (convenzione OpenCV).

    Returns
    -------
    np.ndarray, float32, shape (h, w), valori in [0, 1].
    """
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Modello sconosciuto: '{model_name}'. "
            f"Disponibili: {', '.join(AVAILABLE_MODELS)}"
        )

    model_fn, needs_image = _REGISTRY[model_name]
    if needs_image:
        return model_fn(image_bgr)
    return model_fn(image_bgr.shape[:2])
