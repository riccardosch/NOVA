"""
Spectral Residual (Hou & Zhang, CVPR 2007).

Le immagini naturali hanno mediamente uno spettro di ampiezza molto
regolare. L'ipotesi del metodo e' che cio' che devia da questo andamento
atteso sia l'informazione interessante: si stima l'andamento sfocando il log
dello spettro, lo si sottrae, e il residuo viene riportato nello spazio
immagine.
"""

import numpy as np
import cv2


def spectral_residual_saliency(image, sigma=3.0, target_size=(64, 64)):
    """Saliency map dal residuo spettrale.

    Parameters
    ----------
    image : np.ndarray
        Immagine BGR o gia' in scala di grigi.
    sigma : float
        Deviazione standard dello smoothing finale.
    target_size : tuple(int, int)
        Risoluzione di lavoro (larghezza, altezza). La saliency e' un
        fenomeno a bassa frequenza: lavorare a piena risoluzione costa molto
        senza aggiungere informazione.

    Returns
    -------
    np.ndarray, float32, stessa shape dell'input, valori in [0, 1].
    """
    original_h, original_w = image.shape[:2]

    # 1. Scala di grigi (nota: cosi' l'informazione di colore va persa).
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 2. Riduzione. INTER_AREA media i pixel: evita aliasing, che qui
    #    falserebbe l'analisi dello spettro.
    small = cv2.resize(gray, target_size,
                       interpolation=cv2.INTER_AREA).astype(np.float32)

    # 3. FFT: ampiezza (quanta energia per frequenza) e fase (dove stanno le
    #    strutture nell'immagine).
    spectrum = np.fft.fft2(small)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # 4. Residuo: log-ampiezza meno la sua media locale.
    log_amplitude = np.log(amplitude + 1e-8)
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude

    # 5. Ricostruzione con la fase ORIGINALE: e' cio' che ancora la mappa
    #    alla posizione delle strutture nell'immagine.
    reconstructed = np.fft.ifft2(np.exp(spectral_residual + 1j * phase))
    saliency = (np.abs(reconstructed) ** 2).astype(np.float32)

    # 6. Smoothing: la ricostruzione grezza e' puntinata, l'attenzione umana
    #    si concentra su regioni. |1 forza la dimensione del kernel a dispari.
    ksize = int(np.ceil(sigma * 6)) | 1
    saliency = cv2.GaussianBlur(saliency, (ksize, ksize), sigma)

    saliency = _normalize(saliency)
    return cv2.resize(saliency, (original_w, original_h),
                      interpolation=cv2.INTER_LINEAR)


def _normalize(saliency_map):
    minimum, maximum = saliency_map.min(), saliency_map.max()
    if maximum - minimum > 0:
        saliency_map = (saliency_map - minimum) / (maximum - minimum)
    else:
        saliency_map = np.zeros_like(saliency_map)
    return saliency_map.astype(np.float32)
