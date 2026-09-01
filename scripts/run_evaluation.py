"""
Valutazione dei modelli contro la ground truth annotata.

Uso:
    python scripts/run_evaluation.py

Produce in results/:
    metrics_per_image.csv   una riga per (immagine, modello)
    metrics_summary.csv     media e deviazione standard per modello
"""

import argparse
import csv
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import AVAILABLE_MODELS, compute_saliency
from src.annotation.store import load_all_annotations
from src.annotation.ground_truth import (
    aggregate_annotators_heatmap, clicks_to_fixation_map, all_clicks_of,
)
from src.evaluation.metrics import evaluate_all, INTRINSIC_METRICS


def evaluate_dataset(images_dir, annotations_dir, models=None):
    """Valuta tutti i modelli su tutte le immagini annotate."""
    models = models or AVAILABLE_MODELS
    annotations = load_all_annotations(annotations_dir)

    if not annotations:
        raise SystemExit(
            f"Nessuna annotazione in '{annotations_dir}'. "
            "Esegui prima 'python annotate.py'."
        )

    rows = []

    for image_name, record in sorted(annotations.items()):
        image_bgr = cv2.imread(os.path.join(images_dir, image_name))
        if image_bgr is None:
            print(f"  ATTENZIONE: immagine non trovata, saltata: {image_name}")
            continue

        height, width = image_bgr.shape[:2]

        # Le coordinate dei click sono relative alla risoluzione mostrata
        # all'annotatore. Se l'immagine e' cambiata, il confronto sarebbe
        # falsato senza che nulla lo segnali.
        if (height, width) != tuple(record["image_size"]):
            print(f"  ATTENZIONE: {image_name} e' {(height, width)} ma le annotazioni "
                  f"sono per {tuple(record['image_size'])}. Saltata.")
            continue

        annotators = record["annotators"]
        gt_heatmap = aggregate_annotators_heatmap(annotators, (height, width))
        gt_fixations = clicks_to_fixation_map(all_clicks_of(annotators), (height, width))

        for model_name in models:
            saliency = compute_saliency(model_name, image_bgr)
            metrics = evaluate_all(saliency, gt_heatmap, gt_fixations)

            row = {
                "image": image_name,
                "model": model_name,
                "n_annotators": len(annotators),
                "n_clicks": len(all_clicks_of(annotators)),
            }
            row.update({k: round(v, 4) for k, v in metrics.items()})
            rows.append(row)

        print(f"  {image_name}: {len(models)} modelli, {len(annotators)} annotatori")

    return rows


def summarize(rows, models=None):
    """Media e deviazione standard per modello.

    I NaN delle metriche non definite vengono esclusi: senza il filtro, un
    solo NaN renderebbe NaN l'intera colonna.
    """
    models = models or AVAILABLE_MODELS
    summary = []

    for model_name in models:
        model_rows = [r for r in rows if r["model"] == model_name]
        if not model_rows:
            continue

        entry = {"model": model_name, "n_images": len(model_rows)}

        for metric in INTRINSIC_METRICS:
            values = np.array([r[metric] for r in model_rows], dtype=np.float64)
            valid = values[~np.isnan(values)]
            if valid.size == 0:
                entry[f"{metric}_mean"] = float("nan")
                entry[f"{metric}_std"] = float("nan")
            else:
                entry[f"{metric}_mean"] = round(float(valid.mean()), 4)
                entry[f"{metric}_std"] = round(float(valid.std()), 4)

        summary.append(entry)

    return summary


def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(summary):
    header = f"{'Modello':<20}" + "".join(f"{m:>12}" for m in INTRINSIC_METRICS)
    print("\n" + header)
    print("-" * len(header))
    for entry in summary:
        line = f"{entry['model']:<20}"
        for metric in INTRINSIC_METRICS:
            line += f"{entry[f'{metric}_mean']:>12.4f}"
        print(line)
    print("\nNota: per KL un valore BASSO indica una prestazione migliore.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione dei modelli NOVA.")
    parser.add_argument("--images", default="data/eval")
    parser.add_argument("--annotations", default="data/annotations")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    print(f"Valutazione su '{args.images}' con annotazioni da '{args.annotations}'\n")

    rows = evaluate_dataset(args.images, args.annotations)
    summary = summarize(rows)

    write_csv(rows, os.path.join(args.out, "metrics_per_image.csv"))
    write_csv(summary, os.path.join(args.out, "metrics_summary.csv"))

    print_table(summary)
    print(f"\nRisultati salvati in '{args.out}/'.")
