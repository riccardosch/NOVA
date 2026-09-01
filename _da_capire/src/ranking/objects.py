"""
Ranking degli oggetti per importanza visiva.

E' il passaggio finale del progetto: mostra come le differenze fra i modelli
di saliency si ripercuotono su un compito applicativo concreto.

Due percorsi paralleli, confrontabili fra loro:
    saliency map + box  -> ranking PREDETTO
    click + box         -> ranking DI RIFERIMENTO
"""

import numpy as np


def crop_box(saliency_map, box):
    """Ritaglia la porzione di mappa dentro un box (x1, y1, x2, y2).

    I box di un detector sforano regolarmente i bordi quando un oggetto e'
    parzialmente fuori inquadratura: si riconducono dentro invece di
    sollevare un errore a meta' valutazione.
    """
    height, width = saliency_map.shape[:2]
    x1, y1, x2, y2 = box

    x1 = int(np.clip(x1, 0, width))
    x2 = int(np.clip(x2, 0, width))
    y1 = int(np.clip(y1, 0, height))
    y2 = int(np.clip(y2, 0, height))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=saliency_map.dtype)
    return saliency_map[y1:y2, x1:x2]


def score_box(saliency_map, box, strategy="mean"):
    """Riassume in un numero la saliency contenuta in un box.

    Le tre strategie rispondono a domande diverse:
      mean -- quanto e' saliente in media?      (indipendente dall'area)
      max  -- contiene il punto piu' saliente?  (indipendente ma fragile)
      sum  -- quanta attenzione totale?         (FORTE bias verso i box grandi)

    'sum' e' inclusa proprio per rendere misurabile quel bias: puo'
    ribaltare la classifica di un buon modello.
    """
    region = crop_box(saliency_map, box)
    if region.size == 0:
        return 0.0

    if strategy == "mean":
        return float(region.mean())
    if strategy == "max":
        return float(region.max())
    if strategy == "sum":
        return float(region.sum())
    raise KeyError(f"Strategia sconosciuta: '{strategy}'. Usa mean, max o sum.")


def rank_by_saliency(saliency_map, boxes, strategy="mean"):
    """Ordina i box per saliency contenuta.

    Il campo "index" conserva la posizione originale del box: la lista e'
    ordinata per punteggio, quindi senza di esso sarebbe impossibile
    allineare i due ranking per il confronto.
    """
    if not boxes:
        return []

    scores = np.array([score_box(saliency_map, b, strategy) for b in boxes])
    order = np.argsort(scores)[::-1]

    return [
        {"index": int(i), "box": tuple(boxes[i]),
         "score": float(scores[i]), "rank": position + 1}
        for position, i in enumerate(order)
    ]


def rank_by_clicks(clicks, boxes):
    """Ranking di riferimento dai click annotati.

    Ogni click contribuisce al box che lo contiene con peso 1/order: il
    primo click vale 1.0, il secondo 0.5, il terzo 0.33. La caduta e' ripida
    di proposito: la prima cosa notata e' qualitativamente diversa dalla
    quinta.

    Quando un click cade in piu' box sovrapposti (una persona e il suo
    volto) viene assegnato al PIU' PICCOLO: e' l'interpretazione piu'
    probabile dell'intenzione. E' un'euristica, va dichiarata fra i limiti.
    """
    if not boxes:
        return []

    scores = np.zeros(len(boxes), dtype=np.float64)

    for click in clicks:
        containing = [
            i for i, (x1, y1, x2, y2) in enumerate(boxes)
            if x1 <= click["x"] < x2 and y1 <= click["y"] < y2
        ]
        if not containing:
            continue  # click nello sfondo: nessun oggetto rilevato li'

        smallest = min(containing,
                       key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        scores[smallest] += 1.0 / click.get("order", 1)

    order = np.argsort(scores)[::-1]
    return [
        {"index": int(i), "box": tuple(boxes[i]),
         "score": float(scores[i]), "rank": position + 1}
        for position, i in enumerate(order)
    ]


def aligned_scores(predicted_ranking, reference_ranking, n_boxes):
    """Riporta i due ranking a vettori indicizzati per box originale.

    Le metriche richiedono che la posizione i si riferisca allo stesso
    oggetto in entrambe le sequenze. Passando i punteggi gia' ordinati si
    confronterebbe "il primo secondo il modello" con "il primo secondo la
    ground truth", ottenendo correlazione perfetta SEMPRE: un risultato
    troppo bello per essere vero e senza errori visibili.
    """
    predicted = np.zeros(n_boxes, dtype=np.float64)
    reference = np.zeros(n_boxes, dtype=np.float64)

    for entry in predicted_ranking:
        predicted[entry["index"]] = entry["score"]
    for entry in reference_ranking:
        reference[entry["index"]] = entry["score"]

    return predicted, reference
