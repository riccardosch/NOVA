"""
Suite di test del progetto NOVA.

Strategia: ogni verifica usa casi in cui la risposta corretta e' nota per
costruzione o per ragionamento matematico, non per confronto con un'altra
implementazione. Un test che verifica solo l'accordo con un'altra libreria
non dice se si e' capita la metrica.

Uso:
    python tests/test_all.py
"""

import sys
import os
import shutil
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.registry import AVAILABLE_MODELS, compute_saliency
from src.baselines.center_prior import center_prior_saliency
from src.baselines.spectral_residual import spectral_residual_saliency
from src.baselines.itti_koch import itti_koch_saliency
from src.annotation.store import AnnotationStore, load_all_annotations, MIN_CLICKS, MAX_CLICKS
from src.annotation.ground_truth import (
    clicks_to_fixation_map, clicks_to_heatmap, aggregate_annotators_heatmap,
)
from src.evaluation.metrics import (
    cc, sim, kl_divergence, nss, auc_judd, evaluate_all,
    rank_data, spearman, kendall_tau, top_k_accuracy,
)
from src.ranking.objects import (
    crop_box, score_box, rank_by_saliency, rank_by_clicks, aligned_scores,
)
from src.visualization import overlay_saliency, mark_peak


def check(condition, message):
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    return condition


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


def make_color_image(h=240, w=320, cx=90, cy=160, s=28):
    """Quadrato rosso su sfondo verde rumoroso.

    I due sono quasi ISOLUMINANTI in scala di grigi (94 vs 98): l'oggetto e'
    distinguibile solo dal colore.
    """
    rng = np.random.RandomState(1)
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 1] = (rng.rand(h, w) * 50 + 90).astype(np.uint8)
    img[:, :, 0] = (rng.rand(h, w) * 30 + 60).astype(np.uint8)
    img[:, :, 2] = (rng.rand(h, w) * 30 + 60).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    img[cy - s:cy + s, cx - s:cx + s] = [40, 40, 220]
    return img, (cx - s, cx + s, cy - s, cy + s)


def gaussian_map(h=60, w=80, cx=50, cy=30, sigma=8.0):
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------------------

def test_contract():
    print("\nContratto comune dei modelli")
    img, _ = make_color_image()
    results = []
    for name in AVAILABLE_MODELS:
        sal = compute_saliency(name, img)
        results.append(check(
            sal.dtype == np.float32 and sal.shape == img.shape[:2]
            and 0.0 <= sal.min() and sal.max() <= 1.0,
            f"{name}: float32, shape {sal.shape}, range [0,1]"))

    try:
        compute_saliency("Inesistente", img)
        results.append(check(False, "modello sconosciuto: doveva sollevare KeyError"))
    except KeyError:
        results.append(check(True, "modello sconosciuto: KeyError con elenco dei validi"))
    return results


def test_models_behaviour():
    print("\nComportamento dei modelli")
    results = []

    sal = center_prior_saliency((240, 320))
    py, px = np.unravel_index(np.argmax(sal), sal.shape)
    results.append(check((px, py) == (160, 120),
                         f"center prior: picco esattamente al centro ({px}, {py})"))

    img, (x0, x1, y0, y1) = make_color_image()
    py, px = np.unravel_index(np.argmax(itti_koch_saliency(img)), img.shape[:2])
    results.append(check(x0 <= px <= x1 and y0 <= py <= y1,
                         f"Itti-Koch trova l'oggetto saliente per colore ({px}, {py})"))

    # Limite noto: Spectral Residual lavora in scala di grigi, quindi e'
    # cieco a un oggetto che si distingue solo per tinta.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    square = gray[y0:y1, x0:x1].mean()
    mask = np.ones(gray.shape, bool)
    mask[y0:y1, x0:x1] = False
    results.append(check(abs(square - gray[mask].mean()) < 10,
                         f"quadrato ({square:.0f}) e sfondo ({gray[mask].mean():.0f}) "
                         f"isoluminanti: SR non puo' vederlo, Itti-Koch si'"))

    uniform = np.full((100, 140, 3), 128, np.uint8)
    results.append(check(np.isfinite(itti_koch_saliency(uniform)).all(),
                         "immagine uniforme: nessun NaN"))

    odd = (np.random.RandomState(3).rand(97, 211, 3) * 255).astype(np.uint8)
    results.append(check(itti_koch_saliency(odd).shape == (97, 211)
                         and spectral_residual_saliency(odd).shape == (97, 211),
                         "shape dispari 97x211 preservata"))
    return results


def test_annotation():
    print("\nAnnotazione: persistenza e vincoli")
    results = []
    tmp = tempfile.mkdtemp()
    try:
        store = AnnotationStore("Riccardo", tmp)
        results.append(check(store.annotator == "riccardo", "nome normalizzato"))

        store.save_annotation("img.jpg", (480, 640), [(100, 200), (300, 150), (500, 400)])
        results.append(check(os.path.exists(store.path), "file JSON scritto subito"))

        # Rilettura da istanza nuova: simula la ripresa di una sessione.
        reloaded = AnnotationStore("riccardo", tmp)
        record = reloaded._data["annotations"]["img.jpg"]
        results.append(check(record["image_size"] == [480, 640], "dimensioni conservate"))
        results.append(check([c["order"] for c in record["clicks"]] == [1, 2, 3],
                             "ordine dei click assegnato correttamente"))

        for bad, label in [([(1, 1), (2, 2)], f"meno di {MIN_CLICKS}"),
                           ([(i, i) for i in range(MAX_CLICKS + 1)], f"piu' di {MAX_CLICKS}")]:
            try:
                store.save_annotation("x.jpg", (100, 100), bad)
                results.append(check(False, f"{label} click: doveva fallire"))
            except ValueError:
                results.append(check(True, f"{label} click: rifiutati"))

        AnnotationStore("bruno", tmp).save_annotation("img.jpg", (480, 640),
                                                      [(1, 1), (2, 2), (3, 3)])
        aggregated = load_all_annotations(tmp)
        results.append(check(set(aggregated["img.jpg"]["annotators"]) == {"riccardo", "bruno"},
                             "aggregazione di piu' annotatori"))
    finally:
        shutil.rmtree(tmp)
    return results


def test_ground_truth():
    print("\nGround truth: click -> mappe")
    results = []

    fmap = clicks_to_fixation_map([{"x": 10, "y": 20, "order": 1}], (100, 120))
    results.append(check(fmap.sum() == 1.0 and fmap[20, 10] == 1.0,
                         "mappa binaria: pixel acceso in [y, x]"))
    results.append(check(clicks_to_fixation_map([{"x": 999, "y": -5, "order": 1}],
                                                (100, 120)).sum() == 1.0,
                         "coordinate fuori bordo ricondotte dentro"))

    heatmap = clicks_to_heatmap([{"x": 60, "y": 50, "order": 1}], (100, 120))
    py, px = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    results.append(check((px, py) == (60, 50) and close(heatmap.max(), 1.0, 1e-5),
                         "heatmap: picco sul click, normalizzata"))
    results.append(check(heatmap[50, 60] > heatmap[50, 80] > heatmap[50, 110],
                         "decadimento monotono allontanandosi dal click"))
    results.append(check(clicks_to_heatmap([], (100, 120)).max() == 0.0,
                         "nessun click: mappa vuota, nessun crash"))

    # Il consenso fra annotatori deve emergere da solo.
    annotators = {
        "a": [{"x": 30, "y": 50, "order": 1}],
        "b": [{"x": 32, "y": 51, "order": 1}],
        "c": [{"x": 100, "y": 50, "order": 1}],
    }
    agg = aggregate_annotators_heatmap(annotators, (100, 130))
    results.append(check(agg[50, 31] > agg[50, 100],
                         "zona di consenso piu' saliente del punto isolato"))
    return results


def test_intrinsic_metrics():
    print("\nMetriche intrinseche")
    g = gaussian_map()
    fix = np.zeros((60, 80), np.float32)
    fix[30, 50] = 1.0
    results = []

    results.append(check(close(cc(g, g), 1.0), "CC di una mappa con se stessa = 1"))
    results.append(check(close(cc(g, -g), -1.0), "CC con l'opposto = -1"))
    results.append(check(close(cc(g, g * 7 + 3), 1.0), "CC invariante a scala e offset"))
    results.append(check(close(sim(g, g), 1.0), "SIM di una mappa con se stessa = 1"))
    results.append(check(close(kl_divergence(g, g), 0.0, 1e-9),
                         "KL di distribuzioni identiche = 0"))
    results.append(check(kl_divergence(gaussian_map(cx=15, cy=50), g) > 0,
                         "KL di distribuzioni diverse > 0"))
    results.append(check(nss(g, fix) > 3.0, "NSS alto con fissazione sul picco"))
    results.append(check(np.isnan(nss(g, np.zeros((60, 80), np.float32))),
                         "NSS senza fissazioni = NaN (non definita)"))
    results.append(check(auc_judd(g, fix) > 0.95, "AUC ~ 1 con fissazione sul massimo"))
    results.append(check(close(auc_judd(g, fix), auc_judd(g ** 3, fix), 1e-9),
                         "AUC invariante a trasformazioni monotone"))

    # Con UNA fissazione l'AUC ha area 1 - fp/2: minimo 0.5, non 0.
    lowest = np.unravel_index(np.argmin(g), g.shape)
    on_min = np.zeros((60, 80), np.float32)
    on_min[lowest] = 1.0
    results.append(check(close(auc_judd(g, on_min), 0.5, 0.02),
                         "una fissazione sul minimo: AUC = 0.5, limite inferiore"))

    try:
        cc(g, gaussian_map(50, 70))
        results.append(check(False, "shape diverse: doveva sollevare ValueError"))
    except ValueError:
        results.append(check(True, "shape incompatibili: ValueError"))

    metrics = evaluate_all(g, g, fix)
    results.append(check(len(metrics) == 5 and close(metrics["CC"], 1.0),
                         "evaluate_all calcola tutte e cinque le metriche"))
    return results


def test_ranking_metrics():
    print("\nMetriche di ranking")
    return [
        check(np.allclose(rank_data([10, 20, 20, 40]), [1, 2.5, 2.5, 4]),
              "ranghi con pari merito: media delle posizioni"),
        check(close(spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0),
              "Spearman con ordinamento identico = 1"),
        check(close(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0),
              "Spearman con ordinamento invertito = -1"),
        check(close(spearman([1, 2, 3, 4], [10, 200, 3000, 40000]), 1.0),
              "Spearman = 1 su relazione monotona non lineare"),
        check(close(spearman([1, 2, 3, 4], [1, 2, 4, 3]), 0.8),
              "Spearman: caso verificabile a mano = 0.8"),
        check(close(kendall_tau([1, 2, 3, 4], [1, 2, 4, 3]), 4.0 / 6.0),
              "Kendall: una coppia invertita su sei = 2/3"),
        check(np.isnan(kendall_tau([5, 5, 5], [5, 5, 5])),
              "Kendall con tutte le coppie pari = NaN (non definita)"),
        check(close(top_k_accuracy([0.2, 0.1, 0.8], [0.1, 0.3, 0.9]), 1.0),
              "Top-1 = 1 quando il modello indica l'oggetto giusto"),
        check(close(top_k_accuracy([0.9, 0.1, 0.5], [0.1, 0.3, 0.9]), 0.0),
              "Top-1 = 0 quando sbaglia"),
    ]


def test_ranking():
    print("\nRanking degli oggetti")
    results = []
    saliency = np.zeros((100, 140), np.float32)
    saliency[10:30, 10:30] = 1.0   # solo il primo box e' saliente

    boxes = [(5, 5, 35, 35), (60, 40, 100, 80), (110, 10, 135, 40)]
    ranking = rank_by_saliency(saliency, boxes)
    results.append(check(ranking[0]["index"] == 0,
                         "il box sulla zona saliente e' primo"))
    results.append(check([e["rank"] for e in ranking] == [1, 2, 3],
                         "ranghi assegnati in ordine"))

    results.append(check(crop_box(saliency, (200, 200, 250, 250)).size == 0,
                         "box fuori immagine: ritaglio vuoto"))
    results.append(check(score_box(saliency, (200, 200, 250, 250)) == 0.0,
                         "box fuori immagine: punteggio 0"))

    # Il bias dell'area: mean e sum si comportano in modo opposto.
    flat = np.full((100, 140), 0.05, np.float32)
    flat[15:25, 15:25] = 1.0
    small, large = (10, 10, 30, 30), (50, 10, 110, 70)
    results.append(check(score_box(flat, small, "mean") > score_box(flat, large, "mean"),
                         "MEAN: il box piccolo e saliente vince"))
    results.append(check(score_box(flat, large, "sum") > score_box(flat, small, "sum"),
                         "SUM: vince il box grande (bias dell'area)"))

    clicks = [{"x": 20, "y": 20, "order": 1}, {"x": 80, "y": 60, "order": 2}]
    reference = rank_by_clicks(clicks, boxes)
    results.append(check(reference[0]["index"] == 0 and close(reference[0]["score"], 1.0),
                         "ranking di riferimento: click di ordine 1 vale 1.0"))

    # Box annidati: il click va al piu' piccolo (piu' specifico).
    nested = [(0, 0, 100, 100), (20, 20, 40, 40)]
    nested_rank = rank_by_clicks([{"x": 30, "y": 30, "order": 1}], nested)
    results.append(check(nested_rank[0]["index"] == 1,
                         "box annidati: il click va al piu' piccolo"))

    predicted, ref = aligned_scores(ranking, reference, len(boxes))
    results.append(check(int(np.argmax(predicted)) == 0 and int(np.argmax(ref)) == 0,
                         "i due vettori sono allineati per indice di box"))
    return results


def test_visualization():
    print("\nVisualizzazione")
    image = np.full((100, 140, 3), 128, np.uint8)
    saliency = np.zeros((100, 140), np.float32)
    saliency[50, 70] = 1.0

    overlay = overlay_saliency(image, saliency)
    marked, (px, py) = mark_peak(image, saliency)

    return [
        check(overlay.shape == image.shape and overlay.dtype == np.uint8,
              "overlay: shape e dtype corretti"),
        check((px, py) == (70, 50), f"picco individuato in ({px}, {py})"),
        check(np.array_equal(image, np.full((100, 140, 3), 128, np.uint8)),
              "l'immagine originale non viene modificata (copia difensiva)"),
    ]


if __name__ == "__main__":
    all_results = []
    all_results += test_contract()
    all_results += test_models_behaviour()
    all_results += test_annotation()
    all_results += test_ground_truth()
    all_results += test_intrinsic_metrics()
    all_results += test_ranking_metrics()
    all_results += test_ranking()
    all_results += test_visualization()

    passed, total = sum(all_results), len(all_results)
    print(f"\n{'=' * 60}")
    print(f"Risultato: {passed}/{total} test superati")
    print(f"{'=' * 60}")
    sys.exit(0 if passed == total else 1)
