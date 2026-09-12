import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

"""
evaluate.py --- valuta tutti i modelli di saliency contro i click raccolti.

Uso:
    python evaluate.py --images img/ --clicks clicks.json

Confronta ogni modello del registro con la ground truth costruita dai click
di collect.py, usando tre metriche standard della letteratura sulla
saliency: CC, NSS, AUC-Judd.
"""

import argparse
import json

import numpy as np
import cv2

from src.registry import AVAILABLE_MODELS, compute_saliency


EPS = 1e-12
SIGMA_FRAC = 0.05  # stesso valore usato in collect.py: ~1 grado di angolo visivo
MAX_SALIENCY_SIDE = 800  # lato lungo massimo dato ai modelli di saliency


# ---------------------------------------------------------------------------
# Ridimensionamento per i modelli pesanti (DeepGaze, MSI-Net)
# ---------------------------------------------------------------------------

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
    la mappa alla risoluzione originale (necessaria per confrontarla pixel per
    pixel con la ground truth costruita dai click a piena risoluzione)."""
    h, w = image_bgr.shape[:2]
    small = resize_for_saliency(image_bgr)
    saliency_small = compute_saliency(model_name, small)
    if saliency_small.shape[:2] == (h, w):
        return saliency_small
    return cv2.resize(saliency_small, (w, h), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Ground truth dai click (stessa logica di collect.py/osie_ground_truth.py)
# ---------------------------------------------------------------------------

def clicks_to_heatmap(clicks, height, width):
    """Ogni click diventa una Gaussiana, tutte sommate e normalizzate."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    if not clicks:
        return heatmap
    sigma = SIGMA_FRAC * width
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)
    for c in clicks:
        d2 = (x_grid - c["x"]) ** 2 + (y_grid - c["y"]) ** 2
        heatmap += np.exp(-d2 / (2.0 * sigma ** 2))
    m = heatmap.max()
    return heatmap / m if m > 0 else heatmap


def clicks_to_fixation_map(clicks, height, width):
    """Mappa binaria: 1 sui pixel cliccati (per NSS e AUC-Judd)."""
    fmap = np.zeros((height, width), dtype=np.float32)
    for c in clicks:
        x = int(np.clip(c["x"], 0, width - 1))
        y = int(np.clip(c["y"], 0, height - 1))
        fmap[y, x] = 1.0
    return fmap


# ---------------------------------------------------------------------------
# Metriche -- formule standard, verificate contro valori noti analiticamente
# ---------------------------------------------------------------------------

def _standardize(m):
    """Media 0, deviazione standard 1: rende confrontabili mappe su scale diverse."""
    m = m.astype(np.float64)
    std = m.std()
    return (m - m.mean()) / std if std > EPS else np.zeros_like(m)


def cc(saliency, ground_truth):
    """Correlazione di Pearson. 1 = mappe identiche nell'andamento, -1 = opposte."""
    return float((_standardize(saliency) * _standardize(ground_truth)).mean())


def nss(saliency, fixation_map):
    """Saliency media (standardizzata) esattamente sui punti cliccati.
    0 = casuale, positivo = i click cadono su zone piu' salienti della media."""
    fixations = fixation_map > 0
    if not fixations.any():
        return float("nan")
    return float(_standardize(saliency)[fixations].mean())


def auc_judd(saliency, fixation_map):
    """Area sotto la curva ROC: la mappa come classificatore dei punti cliccati.
    0.5 = casuale, 1 = separazione perfetta."""
    fixations = fixation_map > 0
    if not fixations.any():
        return float("nan")

    s = saliency.astype(np.float64)
    lo, hi = s.min(), s.max()
    if hi - lo < EPS:
        return 0.5
    s = (s - lo) / (hi - lo)

    fix_vals = np.sort(s[fixations])[::-1]
    all_vals = np.sort(s.ravel())
    n_fix, n_pix = fix_vals.size, all_vals.size
    n_non_fix = n_pix - n_fix
    if n_non_fix <= 0:
        return 0.5

    tp, fp = [0.0], [0.0]
    for i, t in enumerate(fix_vals):
        above = n_pix - np.searchsorted(all_vals, t, side="left")
        tp.append((i + 1) / n_fix)
        fp.append((above - (i + 1)) / n_non_fix)
    tp.append(1.0)
    fp.append(1.0)
    tp, fp = np.array(tp), np.array(fp)
    return float(np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1]) / 2.0))


METRICS = {"CC": cc, "NSS": nss, "AUC-Judd": auc_judd}
NEEDS_FIXATIONS = {"CC": False, "NSS": True, "AUC-Judd": True}


# ---------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------

def evaluate(images_dir, clicks_path):
    with open(clicks_path) as f:
        all_clicks = json.load(f)

    rows = []
    for name, clicks in all_clicks.items():
        path = os.path.join(images_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"  ATTENZIONE: immagine non trovata: {name}, saltata.")
            continue

        h, w = image_bgr.shape[:2]
        gt_heatmap = clicks_to_heatmap(clicks, h, w)
        gt_fixations = clicks_to_fixation_map(clicks, h, w)

        for model_name in AVAILABLE_MODELS:
            try:
                saliency = compute_saliency_at_full_resolution(model_name, image_bgr)
            except Exception as error:
                print(f"  {model_name} fallito su {name}: {error}")
                continue

            row = {"image": name, "model": model_name}
            for metric_name, fn in METRICS.items():
                reference = gt_fixations if NEEDS_FIXATIONS[metric_name] else gt_heatmap
                row[metric_name] = round(fn(saliency, reference), 4)
            rows.append(row)

        print(f"  {name}: valutata su {len(AVAILABLE_MODELS)} modelli")

    return rows


def summarize(rows):
    models = sorted(set(r["model"] for r in rows))
    summary = []
    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        entry = {"model": model, "n_images": len(model_rows)}
        for metric_name in METRICS:
            values = np.array([r[metric_name] for r in model_rows], dtype=np.float64)
            valid = values[~np.isnan(values)]
            entry[metric_name] = round(float(valid.mean()), 4) if valid.size else float("nan")
        summary.append(entry)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuta i modelli NOVA contro i click raccolti.")
    parser.add_argument("--images", default="img")
    parser.add_argument("--clicks", default="clicks.json")
    args = parser.parse_args()

    print(f"Valutazione su '{args.images}' con ground truth da '{args.clicks}'\n")
    rows = evaluate(args.images, args.clicks)
    summary = summarize(rows)

    header = f"\n{'Modello':<24}{'CC':>10}{'NSS':>10}{'AUC-Judd':>10}"
    print(header)
    print("-" * len(header.strip()))
    for e in summary:
        print(f"{e['model']:<24}{e['CC']:>10.4f}{e['NSS']:>10.4f}{e['AUC-Judd']:>10.4f}")
