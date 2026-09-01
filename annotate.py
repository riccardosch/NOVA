"""
NOVA --- strumento di annotazione a click.

Uso:
    python annotate.py --images data/eval

Protocollo: l'annotatore clicca, IN ORDINE DI IMPORTANZA DECRESCENTE, da 3
a 6 volte per immagine. Un solo compito produce due ground truth: le
posizioni (per NSS/AUC-Judd) e l'ordine (per il ranking).
"""

import argparse
import os

import cv2
import gradio as gr

from src.annotation.store import AnnotationStore, MIN_CLICKS, MAX_CLICKS


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

ISTRUZIONI = f"""
## Istruzioni

Guarda l'immagine e clicca, **in ordine di importanza decrescente**, sugli
elementi che attirano di più la tua attenzione.

- Il **primo click** indica l'elemento più importante
- Minimo **{MIN_CLICKS}** click, massimo **{MAX_CLICKS}**
- Segui il tuo istinto: la prima impressione conta

Quando hai finito, premi **Salva e continua**.
"""


def list_images(images_dir):
    if not os.path.isdir(images_dir):
        return []
    return sorted(f for f in os.listdir(images_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS))


def draw_clicks(image_rgb, clicks):
    """Disegna i click numerati.

    Il numero rende visibile l'ORDINE, che e' meta' del dato raccolto:
    senza, l'annotatore perde il conto di cosa ha gia' cliccato.
    """
    annotated = image_rgb.copy()

    for index, (x, y) in enumerate(clicks, start=1):
        cv2.circle(annotated, (x, y), 16, (0, 0, 0), -1)
        cv2.circle(annotated, (x, y), 14, (255, 80, 40), -1)

        text = str(index)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(annotated, text, (x - tw // 2, y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


class AnnotationSession:
    """Stato della sessione: quale immagine, quali click, quale annotatore.

    Tenuto separato dall'interfaccia cosi' e' testabile senza browser.
    """

    def __init__(self, images_dir, annotations_dir="data/annotations"):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_names = list_images(images_dir)
        self.store = None
        self.index = 0
        self.clicks = []
        self.current_image = None

    def start(self, annotator):
        """Avvia la sessione saltando le immagini gia' annotate.

        Chi riapre il browser il giorno dopo riprende da dove aveva
        lasciato, senza rifare il lavoro.
        """
        self.store = AnnotationStore(annotator, self.annotations_dir)
        done = self.store.annotated_images()

        self.index = 0
        while self.index < len(self.image_names) and self.image_names[self.index] in done:
            self.index += 1

        self.clicks = []
        return self._load_current()

    def _load_current(self):
        if self.finished:
            self.current_image = None
            return None

        path = os.path.join(self.images_dir, self.image_names[self.index])
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            self.current_image = None
            return None

        self.current_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.current_image

    @property
    def finished(self):
        return self.index >= len(self.image_names)

    @property
    def current_name(self):
        return None if self.finished else self.image_names[self.index]

    def add_click(self, x, y):
        if len(self.clicks) >= MAX_CLICKS:
            return False
        self.clicks.append((int(x), int(y)))
        return True

    def undo_click(self):
        if self.clicks:
            self.clicks.pop()

    def save_and_advance(self):
        """Salva e passa oltre. Restituisce (successo, messaggio).

        Coppia invece di eccezione: in un'interfaccia, un vincolo non
        rispettato e' una condizione normale da comunicare, non un bug.
        Si salva PRIMA di avanzare: se il salvataggio fallisse, l'indice non
        avanzerebbe e il lavoro non andrebbe perso.
        """
        if self.finished:
            return False, "Sessione già completata."
        if len(self.clicks) < MIN_CLICKS:
            return False, f"Servono almeno {MIN_CLICKS} click (ne hai {len(self.clicks)})."

        height, width = self.current_image.shape[:2]
        self.store.save_annotation(self.current_name, (height, width), self.clicks)

        self.index += 1
        self.clicks = []
        self._load_current()
        return True, "Salvato."

    def skip(self):
        """Salta senza salvare: utile se l'immagine e' inutilizzabile."""
        if not self.finished:
            self.index += 1
            self.clicks = []
            self._load_current()

    def view(self):
        if self.current_image is None:
            return None
        return draw_clicks(self.current_image, self.clicks)

    def status(self):
        if self.store is None:
            return "Inserisci il tuo nome e premi Inizia."
        if self.finished:
            return f"Tutte le immagini annotate. Totale salvate: {self.store.count()}."
        return (f"Immagine {self.index + 1} di {len(self.image_names)} — "
                f"{self.current_name} — click: {len(self.clicks)}/{MAX_CLICKS} "
                f"(minimo {MIN_CLICKS})")


def build_interface(images_dir, annotations_dir="data/annotations"):
    session = AnnotationSession(images_dir, annotations_dir)

    with gr.Blocks(title="NOVA --- Annotazione") as demo:
        gr.Markdown("# NOVA --- Raccolta ground truth")
        gr.Markdown(ISTRUZIONI)

        with gr.Row():
            annotator_box = gr.Textbox(label="Nome annotatore",
                                       placeholder="es. riccardo", scale=3)
            start_button = gr.Button("Inizia", variant="primary", scale=1)

        status_box = gr.Textbox(label="Stato", interactive=False, value=session.status())
        image_display = gr.Image(label="Clicca in ordine di importanza",
                                 type="numpy", interactive=False, height=520)

        with gr.Row():
            undo_button = gr.Button("Annulla ultimo click")
            skip_button = gr.Button("Salta immagine")
            save_button = gr.Button("Salva e continua", variant="primary")

        def on_start(annotator):
            try:
                session.start(annotator)
            except ValueError as error:
                return None, str(error)
            return session.view(), session.status()

        def on_click(event: gr.SelectData):
            if session.current_image is None:
                return None, session.status()
            # event.index e' [x, y]: colonna, riga. Ordine opposto a numpy.
            x, y = event.index[0], event.index[1]
            if not session.add_click(x, y):
                return session.view(), f"Massimo {MAX_CLICKS} click raggiunto."
            return session.view(), session.status()

        def on_undo():
            session.undo_click()
            return session.view(), session.status()

        def on_skip():
            session.skip()
            return session.view(), session.status()

        def on_save():
            ok, message = session.save_and_advance()
            return session.view(), session.status() if ok else message

        start_button.click(on_start, inputs=[annotator_box],
                           outputs=[image_display, status_box])
        image_display.select(on_click, outputs=[image_display, status_box])
        undo_button.click(on_undo, outputs=[image_display, status_box])
        skip_button.click(on_skip, outputs=[image_display, status_box])
        save_button.click(on_save, outputs=[image_display, status_box])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotazione e analisi a click per NOVA.")
    parser.add_argument("--images", default="data/eval")
    parser.add_argument("--out", default="data/annotations")
    args = parser.parse_args()

    import uvicorn
    from app import app
    print("Avvio di NOVA Web App Unificata su http://localhost:7860 ...")
    uvicorn.run(app, host="0.0.0.0", port=7860)

