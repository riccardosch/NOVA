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
    if model_name not in _REGISTRY:
        raise KeyError(
            f"Modello sconosciuto: '{model_name}'. "
            f"Disponibili: {', '.join(AVAILABLE_MODELS)}"
        )

    model_fn, needs_image = _REGISTRY[model_name]
    if needs_image:
        return model_fn(image_bgr)
    return model_fn(image_bgr.shape[:2])
