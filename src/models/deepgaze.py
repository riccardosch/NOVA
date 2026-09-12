"""
DeepGaze IIE — Linardos et al., 2021

Wrapper per il modello di saliency pre-addestrato basato su PyTorch.
Restituisce una LOG-DENSITY: va convertita esplicitamente in mappa [0,1].

Installazione:
    pip install torch einops
    pip install git+https://github.com/matthias-k/DeepGaze.git
    pip install git+https://github.com/openai/CLIP.git
"""

import numpy as np
import cv2

# ------------------------------------------------------------------ #
#  Cache del modello a livello di modulo: caricato una sola volta
# ------------------------------------------------------------------ #
_model = None
_current_device = None


def is_available() -> bool:
    """Ritorna True se torch e deepgaze_pytorch sono importabili."""
    try:
        import torch              # noqa: F401
        import deepgaze_pytorch   # noqa: F401
        return True
    except ImportError:
        return False


def _check_deps():
    """Solleva ImportError se le dipendenze mancano."""
    if not is_available():
        raise ImportError(
            "DeepGaze IIE richiede:\n"
            "  pip install torch einops\n"
            "  pip install git+https://github.com/matthias-k/DeepGaze.git\n"
            "  pip install git+https://github.com/openai/CLIP.git"
        )


def _load_model(device):
    """Carica il modello DeepGaze IIE (una sola volta)."""
    global _model, _current_device
    if _model is None or _current_device != device:
        _check_deps()
        import torch
        import deepgaze_pytorch

        _current_device = device
        _model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
        _model.eval()
    return _model


# ------------------------------------------------------------------ #
#  Center bias uniforme
# ------------------------------------------------------------------ #

def build_uniform_centerbias(height, width):
    """Center bias piatto: log(1/N) per ogni pixel, N = height*width.
    exp() di questo array somma esattamente a 1."""
    n = height * width
    return np.full((height, width), np.log(1.0 / n), dtype=np.float32)


# ------------------------------------------------------------------ #
#  Normalizzazione min-max
# ------------------------------------------------------------------ #

def _normalize(saliency_map):
    """Riscala una mappa in [0, 1] (min-max)."""
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)


# ------------------------------------------------------------------ #
#  Funzione principale
# ------------------------------------------------------------------ #

def deepgaze_saliency(image_bgr, device="cpu"):
    """
    Calcola la saliency map con DeepGaze IIE.

    Parameters
    ----------
    image_bgr : np.ndarray, uint8, shape (h, w, 3)
    device : str, "cpu" o "cuda"

    Returns
    -------
    np.ndarray, float32, shape (h, w), valori in [0, 1]
    """
    _check_deps()
    import torch

    model = _load_model(device)

    h, w = image_bgr.shape[:2]

    # BGR -> RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Tensore NCHW float32
    image_tensor = torch.tensor(
        image_rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0,
        device=device,
    )

    # Center bias uniforme, shape (1, h, w)
    centerbias_np = build_uniform_centerbias(h, w)
    centerbias_tensor = torch.tensor(
        centerbias_np[np.newaxis, ...],
        device=device,
    )

    # Inferenza — il modello restituisce una log-density
    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)

    # Log-density -> density (sottrai il max per stabilità numerica)
    log_density = log_density.squeeze()
    density = torch.exp(log_density - log_density.max())

    # Numpy, normalizzazione min-max in [0, 1]
    saliency = density.cpu().numpy()
    saliency = _normalize(saliency)

    return saliency



# ------------------------------------------------------------------ #
#  Variante fine-tuned — stessa architettura, pesi aggiornati
# ------------------------------------------------------------------ #
_model_finetuned = None
_current_device_finetuned = None


def _load_model_finetuned(device, weights_path="deepgaze_finetuned.pt"):
    """Carica il modello base e ci sovrascrive i pesi fine-tuned (una sola volta)."""
    global _model_finetuned, _current_device_finetuned
    if _model_finetuned is None or _current_device_finetuned != device:
        _check_deps()
        import torch
        import deepgaze_pytorch

        _current_device_finetuned = device
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        _model_finetuned = model
    return _model_finetuned


def deepgaze_finetuned_saliency(image_bgr, device="cpu"):
    """Identica a deepgaze_saliency, ma con i pesi fine-tuned su OSIE."""
    _check_deps()
    import torch

    model = _load_model_finetuned(device)

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(
        image_rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0,
        device=device,
    )

    centerbias_np = build_uniform_centerbias(h, w)
    centerbias_tensor = torch.tensor(centerbias_np[np.newaxis, ...], device=device)

    with torch.no_grad():
        log_density = model(image_tensor, centerbias_tensor)

    log_density = log_density.squeeze()
    density = torch.exp(log_density - log_density.max())

    saliency = density.cpu().numpy()
    saliency = _normalize(saliency)
    return saliency