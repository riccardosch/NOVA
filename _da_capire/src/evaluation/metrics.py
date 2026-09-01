"""
Metriche di valutazione, implementate da zero.

Due famiglie:

  INTRINSECHE -- confrontano la saliency map con la ground truth
    CC   Pearson correlation        [-1, 1]   alto = meglio
    SIM  histogram intersection     [0, 1]    alto = meglio
    KL   Kullback-Leibler           [0, inf)  BASSO = meglio
    NSS  Normalized Scanpath Sal.   (-inf,inf) alto = meglio
    AUC  Area Under ROC (Judd)      [0, 1]    alto = meglio

  ESTRINSECHE -- confrontano due ordinamenti di oggetti
    Spearman, Kendall tau-b, Top-1 accuracy

Perche' non basta una metrica: ognuna e' cieca a qualcosa. CC ignora dove
sta il picco purche' l'andamento corrisponda; NSS guarda solo i punti
fissati; AUC e' invariante a trasformazioni monotone (conta l'ordinamento,
non i valori).
"""

import numpy as np


EPS = 1e-12


# ---------------------------------------------------------------------------
# Supporto
# ---------------------------------------------------------------------------

def _to_distribution(saliency_map):
    """Riscala la mappa perche' la somma valga 1 (per SIM e KL).

    float64 non e' pedanteria: KL somma logaritmi su decine di migliaia di
    pixel, e in float32 l'errore si accumula.
    """
    m = saliency_map.astype(np.float64)
    m = m - m.min()
    total = m.sum()
    if total < EPS:
        return np.full_like(m, 1.0 / m.size)
    return m / total


def _standardize(saliency_map):
    """Media 0, deviazione standard 1 (per CC e NSS)."""
    m = saliency_map.astype(np.float64)
    std = m.std()
    if std < EPS:
        return np.zeros_like(m)
    return (m - m.mean()) / std


def _check_shape(a, b):
    if a.shape != b.shape:
        raise ValueError(
            f"Dimensioni incompatibili: {a.shape} vs {b.shape}. "
            "Le mappe vanno riportate alla stessa risoluzione."
        )


# ---------------------------------------------------------------------------
# Metriche intrinseche
# ---------------------------------------------------------------------------

def cc(saliency_map, ground_truth_map):
    """Correlazione di Pearson fra due mappe continue.

    Standardizzando entrambe, la formula si riduce alla media del prodotto.
    Invariante a scala e offset: una mappa e la stessa x10 danno CC = 1.
    """
    _check_shape(saliency_map, ground_truth_map)
    return float((_standardize(saliency_map) * _standardize(ground_truth_map)).mean())


def sim(saliency_map, ground_truth_map):
    """Quanta massa di probabilita' condividono le due distribuzioni."""
    _check_shape(saliency_map, ground_truth_map)
    s = _to_distribution(saliency_map)
    g = _to_distribution(ground_truth_map)
    return float(np.minimum(s, g).sum())


def kl_divergence(saliency_map, ground_truth_map):
    """KL(ground truth || predizione). BASSO = MEGLIO.

    Asimmetrica di proposito: penalizza severamente il modello che assegna
    saliency quasi nulla dove la ground truth ne ha molta -- nell'uso reale,
    mancare un oggetto importante e' piu' grave che segnalarne uno di troppo.

    La somma corre solo dove g > 0 (altrove il termine vale 0 per
    definizione) e usa maximum(s, EPS), non s + EPS: sommare EPS altererebbe
    il rapporto ovunque, accumulando errore su ogni pixel.
    """
    _check_shape(saliency_map, ground_truth_map)
    s = _to_distribution(saliency_map)
    g = _to_distribution(ground_truth_map)

    support = g > 0
    if not support.any():
        return 0.0
    return float(np.sum(g[support] * np.log(g[support] / np.maximum(s[support], EPS))))


def nss(saliency_map, fixation_map):
    """Valore medio della mappa standardizzata sui punti fissati.

    0 = prestazione casuale. Restituisce NaN senza fissazioni: 0
    significherebbe "casuale" e falserebbe le medie sul dataset.
    """
    _check_shape(saliency_map, fixation_map)
    fixations = fixation_map > 0
    if not fixations.any():
        return float("nan")
    return float(_standardize(saliency_map)[fixations].mean())


def auc_judd(saliency_map, fixation_map):
    """Area sotto la curva ROC, variante di Judd.

    La saliency map viene letta come un classificatore: al variare della
    soglia, quante fissazioni cattura rispetto a quanta immagine seleziona?

    Nota: con UNA sola fissazione l'AUC non puo' scendere sotto 0.5. La
    curva ha tre punti -- (0,0), (fp,1), (1,1) -- e area 1 - fp/2, minima a
    0.5 quando fp = 1. Servono almeno due fissazioni perche' la metrica
    possa segnalare una predizione peggiore del caso.
    """
    _check_shape(saliency_map, fixation_map)
    fixations = fixation_map > 0
    if not fixations.any():
        return float("nan")

    s = saliency_map.astype(np.float64)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < EPS:
        return 0.5
    s = (s - s_min) / (s_max - s_min)

    fixation_values = np.sort(s[fixations])[::-1]
    all_values = np.sort(s.ravel())
    n_fixations = fixation_values.size
    n_pixels = all_values.size
    n_non_fixations = n_pixels - n_fixations
    if n_non_fixations <= 0:
        return 0.5

    tp_rates = [0.0]
    fp_rates = [0.0]

    # Le soglie sono i valori nei punti fissati: gli unici in cui il true
    # positive rate cambia. searchsorted usa la ricerca binaria, O(log n).
    for i, threshold in enumerate(fixation_values):
        above = n_pixels - np.searchsorted(all_values, threshold, side="left")
        tp_rates.append((i + 1) / n_fixations)
        # Le fissazioni sopra soglia sono VERI positivi: si sottraggono dal
        # numeratore e si escludono dal denominatore.
        fp_rates.append((above - (i + 1)) / n_non_fixations)

    tp_rates.append(1.0)
    fp_rates.append(1.0)

    tp = np.asarray(tp_rates)
    fp = np.asarray(fp_rates)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0))


# Nome -> (funzione, richiede_fissazioni, alto_e_meglio)
INTRINSIC_METRICS = {
    "CC": (cc, False, True),
    "SIM": (sim, False, True),
    "KL": (kl_divergence, False, False),
    "NSS": (nss, True, True),
    "AUC-Judd": (auc_judd, True, True),
}


def evaluate_all(saliency_map, ground_truth_map, fixation_map):
    """Calcola tutte le metriche intrinseche."""
    results = {}
    for name, (fn, needs_fixations, _) in INTRINSIC_METRICS.items():
        reference = fixation_map if needs_fixations else ground_truth_map
        results[name] = fn(saliency_map, reference)
    return results


# ---------------------------------------------------------------------------
# Metriche estrinseche (ranking)
# ---------------------------------------------------------------------------

def rank_data(values):
    """Ranghi con media in caso di pari merito: [10,20,20,40] -> [1,2.5,2.5,4].

    E' la convenzione che rende corretti Spearman e Kendall-b in presenza di
    valori identici: assegnare ranghi arbitrari inventerebbe un ordine.
    """
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    ranks = np.empty(n, dtype=np.float64)

    order = np.argsort(values, kind="mergesort")  # stabile
    sorted_values = values[order]

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    return ranks


def spearman(predicted, reference):
    """Correlazione di Pearson calcolata sui ranghi.

    Cattura qualsiasi relazione monotona: [1,2,3] e [10,200,3000] danno 1.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.size != reference.size:
        raise ValueError("Le due sequenze devono avere la stessa lunghezza.")
    if predicted.size < 2:
        return float("nan")

    rp = rank_data(predicted) 
    rt = rank_data(reference)
    rp = rp - rp.mean()
    rt = rt - rt.mean()

    denominator = np.sqrt((rp ** 2).sum() * (rt ** 2).sum())
    if denominator < EPS:
        return float("nan")
    return float((rp * rt).sum() / denominator)


def kendall_tau(predicted, reference):
    """Tau-b di Kendall: frazione di coppie su cui i due ordinamenti concordano.

    tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))
    con n0 = coppie totali, n1/n2 = coppie in pari merito nelle due sequenze.

    ATTENZIONE al denominatore: (n0 - n1) NON equivale a (C + D + n1). Le due
    espressioni coincidono quasi sempre, quindi l'errore resta invisibile
    tranne nei casi degeneri.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.size != reference.size:
        raise ValueError("Le due sequenze devono avere la stessa lunghezza.")

    n = predicted.size
    if n < 2:
        return float("nan")

    concordant = discordant = ties_predicted = ties_reference = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dp = predicted[i] - predicted[j]
            dt = reference[i] - reference[j]

            if dp == 0:
                ties_predicted += 1
            if dt == 0:
                ties_reference += 1

            if dp != 0 and dt != 0:
                if dp * dt > 0:
                    concordant += 1
                else:
                    discordant += 1

    n_pairs = n * (n - 1) // 2
    denominator = np.sqrt((n_pairs - ties_predicted) * (n_pairs - ties_reference))
    if denominator < EPS:
        return float("nan")
    return float((concordant - discordant) / denominator)


def top_k_accuracy(predicted, reference, k=1):
    """L'oggetto piu' importante secondo la ground truth e' nei primi k
    posti della predizione?

    Metrica binaria per immagine; la media sul dataset e' una percentuale.
    Con k=1 risponde alla domanda che conta davvero nell'applicazione:
    "il modello individua l'oggetto principale?".
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.size != reference.size:
        raise ValueError("Le due sequenze devono avere la stessa lunghezza.")
    if predicted.size == 0:
        return float("nan")

    true_best = int(np.argmax(reference))
    top_k_predicted = np.argsort(predicted)[::-1][:k]
    return float(true_best in top_k_predicted)


def evaluate_ranking(predicted, reference):
    """Calcola tutte le metriche di ranking."""
    return {
        "Spearman": spearman(predicted, reference),
        "Kendall tau": kendall_tau(predicted, reference),
        "Top-1": top_k_accuracy(predicted, reference, k=1),
    }
