import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

"""
rank_objects.py --- ranking oggetti per importanza visiva, guidato dalla saliency.

Uso:
    python rank_objects.py --images img/ --clicks clicks.json

Per ogni immagine: rileva oggetti con YOLOv8, assegna a ciascuna box un
punteggio di saliency (media dei pixel nella box) per ogni modello del
registro, e confronta il ranking risultante con l'ordine dei click
(ground truth) tramite Spearman e Kendall tau-b.
"""

import argparse
import json
import os

import numpy as np
import cv2
from scipy.stats import spearmanr, kendalltau

from src.registry import AVAILABLE_MODELS, compute_saliency


# ---------------------------------------------------------------------------
# Rilevamento oggetti (YOLOv8)
# ---------------------------------------------------------------------------

_yolo_model = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_objects(image_bgr):
    """Rileva oggetti con YOLOv8. Ritorna lista di dict: {box:(x1,y1,x2,y2), label, conf}."""
    model = _load_yolo()
    results = model(image_bgr, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        label = results.names[int(box.cls[0])]
        conf = float(box.conf[0])
        boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": label, "conf": conf})
    return boxes


# ---------------------------------------------------------------------------
# Ridimensionamento per i modelli pesanti (DeepGaze, MSI-Net)
# ---------------------------------------------------------------------------

MAX_SALIENCY_SIDE = 800  # lato lungo massimo dato ai modelli di saliency


def resize_for_saliency(image_bgr, max_side=MAX_SALIENCY_SIDE):
    """Riduce l'immagine se supera max_side sul lato lungo (non ingrandisce mai).
    Necessario: alcune foto arrivano fino a ~20 megapixel e DeepGaze (due copie
    in RAM, base + fine-tuned) su quella risoluzione esaurisce la memoria."""
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image_bgr
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_saliency_at_full_resolution(model_name, image_bgr):
    """Calcola la saliency su una versione ridotta dell'immagine, poi riporta
    la mappa alla risoluzione originale (necessaria per indicizzare le box YOLO,
    che restano sulle coordinate dell'immagine originale)."""
    h, w = image_bgr.shape[:2]
    small = resize_for_saliency(image_bgr)
    saliency_small = compute_saliency(model_name, small)
    if saliency_small.shape[:2] == (h, w):
        return saliency_small
    return cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Punteggio box <- saliency map
# ---------------------------------------------------------------------------

def score_box(saliency_map, box):
    """Punteggio di una box = media della saliency map al suo interno."""
    x1, y1, x2, y2 = box
    h, w = saliency_map.shape
    x1c, x2c = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1c, y2c = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    if x2c <= x1c or y2c <= y1c:
        return 0.0
    return float(saliency_map[y1c:y2c, x1c:x2c].mean())


# ---------------------------------------------------------------------------
# Ground truth dal ranking dei click
# ---------------------------------------------------------------------------

def ground_truth_rank(boxes, clicks):
    """Assegna a ogni box l'ordine del primo click che cade al suo interno.
    Box mai cliccate: None (escluse dal confronto)."""
    ordered_clicks = sorted(clicks, key=lambda c: c["order"])
    ranks = [None] * len(boxes)
    for click in ordered_clicks:
        for i, b in enumerate(boxes):
            if ranks[i] is not None:
                continue
            x1, y1, x2, y2 = b["box"]
            if x1 <= click["x"] <= x2 and y1 <= click["y"] <= y2:
                ranks[i] = click["order"]
    return ranks


# ---------------------------------------------------------------------------
# Confronto ranking
# ---------------------------------------------------------------------------

def compare_rankings(model_scores, gt_ranks):
    """Spearman e Kendall tau-b tra punteggio del modello e ordine di verità,
    solo sulle box effettivamente cliccate. Rank 1 = primo click = piu' importante,
    quindi si confronta con -rank per allineare il verso della correlazione."""
    pairs = [(s, r) for s, r in zip(model_scores, gt_ranks) if r is not None]
    if len(pairs) < 2:
        return float("nan"), float("nan")
    scores, ranks = zip(*pairs)
    rho, _ = spearmanr(scores, [-r for r in ranks])
    tau, _ = kendalltau(scores, [-r for r in ranks])
    return float(rho), float(tau)


# ---------------------------------------------------------------------------
# Orchestrazione
# ---------------------------------------------------------------------------

def evaluate_ranking(images_dir, clicks_path):
    with open(clicks_path) as f:
        all_clicks = json.load(f)

    per_model_scores = {name: [] for name in AVAILABLE_MODELS}

    for name, clicks in all_clicks.items():
        path = os.path.join(images_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  ATTENZIONE: immagine non trovata: {name}, saltata.")
            continue

        boxes = detect_objects(image_bgr)
        if len(boxes) < 2:
            print(f"  {name}: meno di 2 oggetti rilevati, saltata dal confronto ranking.")
            continue

        gt_ranks = ground_truth_rank(boxes, clicks)
        n_clicked = sum(r is not None for r in gt_ranks)
        if n_clicked < 2:
            print(f"  {name}: meno di 2 box cliccate, saltata dal confronto ranking.")
            continue

        for model_name in AVAILABLE_MODELS:
            try:
                saliency = compute_saliency_at_full_resolution(model_name, image_bgr)
            except Exception as error:
                print(f"  {model_name} fallito su {name}: {error}")
                continue
            scores = [score_box(saliency, b["box"]) for b in boxes]
            rho, tau = compare_rankings(scores, gt_ranks)
            per_model_scores[model_name].append((rho, tau))

        print(f"  {name}: {len(boxes)} oggetti rilevati, ranking confrontato su {n_clicked} box cliccate")

    return per_model_scores


def summarize(per_model_scores):
    summary = []
    for model_name, values in per_model_scores.items():
        if not values:
            continue
        rhos = np.array([v[0] for v in values], dtype=np.float64)
        taus = np.array([v[1] for v in values], dtype=np.float64)
        rhos_valid = rhos[~np.isnan(rhos)]
        taus_valid = taus[~np.isnan(taus)]
        summary.append({
            "model": model_name,
            "n_images": len(values),
            "Spearman": round(float(rhos_valid.mean()), 4) if rhos_valid.size else float("nan"),
            "Kendall": round(float(taus_valid.mean()), 4) if taus_valid.size else float("nan"),
        })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Valuta il ranking oggetti guidato dalla saliency, confrontato con l'ordine dei click."
    )
    parser.add_argument("--images", default="img")
    parser.add_argument("--clicks", default="clicks.json")
    args = parser.parse_args()

    print(f"Valutazione ranking su '{args.images}' con ground truth da '{args.clicks}'\n")
    per_model = evaluate_ranking(args.images, args.clicks)
    summary = summarize(per_model)

    header = f"\n{'Modello':<28}{'N img':>8}{'Spearman':>12}{'Kendall':>12}"
    print(header)
    print("-" * len(header.strip()))
    for e in summary:
        print(f"{e['model']:<28}{e['n_images']:>8}{e['Spearman']:>12.4f}{e['Kendall']:>12.4f}")
