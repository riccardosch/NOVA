"""
Persistenza delle annotazioni a click.

Un file JSON per annotatore (piu' persone possono annotare in parallelo
senza conflitti). Salvataggio incrementale: ogni immagine confermata viene
scritta subito, cosi' una sessione interrotta non perde il lavoro fatto.
"""

import json
import os
from datetime import datetime, timezone


MIN_CLICKS = 3   # sotto, non esiste un ranking sensato
MAX_CLICKS = 6   # sopra, si cercano cose da cliccare invece di notarle


class AnnotationStore:
    """Legge e scrive le annotazioni di un singolo annotatore."""

    def __init__(self, annotator, annotations_dir="data/annotations"):
        if not annotator or not annotator.strip():
            raise ValueError("Il nome dell'annotatore non puo' essere vuoto.")

        # Normalizzazione: evita che "Riccardo" e "riccardo " diventino due
        # annotatori distinti.
        self.annotator = annotator.strip().lower().replace(" ", "_")
        self.annotations_dir = annotations_dir
        self.path = os.path.join(annotations_dir, f"{self.annotator}.json")
        self._data = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "annotator": self.annotator,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "annotations": {},
        }

    def _save(self):
        """Scrittura atomica: file temporaneo, poi rinomina.

        os.replace e' atomico: un'interruzione a meta' scrittura non puo'
        corrompere il file esistente lasciando un JSON troncato.
        """
        os.makedirs(self.annotations_dir, exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def save_annotation(self, image_name, image_size, clicks):
        """Registra i click di un'immagine e salva subito.

        image_size viene memorizzato insieme ai click: senza, le coordinate
        sarebbero prive di significato se l'immagine venisse ridimensionata.
        L'ordine si deriva dalla posizione nella lista, cosi' il chiamante
        non puo' sbagliarlo.
        """
        if not MIN_CLICKS <= len(clicks) <= MAX_CLICKS:
            raise ValueError(
                f"Servono da {MIN_CLICKS} a {MAX_CLICKS} click, ricevuti {len(clicks)}."
            )
        self._data["annotations"][image_name] = {
            "image_size": list(image_size),
            "clicks": [
                {"x": int(x), "y": int(y), "order": i + 1}
                for i, (x, y) in enumerate(clicks)
            ],
            "annotated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()

    def annotated_images(self):
        return set(self._data["annotations"].keys())

    def count(self):
        return len(self._data["annotations"])


def load_all_annotations(annotations_dir="data/annotations"):
    """Unisce i file di tutti gli annotatori, raggruppando per immagine.

    Returns
    -------
    dict
        {nome_immagine: {"image_size": [h, w], "annotators": {nome: clicks}}}
    """
    aggregated = {}
    if not os.path.isdir(annotations_dir):
        return aggregated

    for filename in sorted(os.listdir(annotations_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(annotations_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        annotator = data.get("annotator", filename[:-5])
        for image_name, record in data.get("annotations", {}).items():
            if image_name not in aggregated:
                aggregated[image_name] = {
                    "image_size": record["image_size"],
                    "annotators": {},
                }
            aggregated[image_name]["annotators"][annotator] = record["clicks"]

    return aggregated
