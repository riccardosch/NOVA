

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # forza CPU — evita segfault TF/PyTorch

import cv2
import numpy as np
import gradio as gr

from src.registry import AVAILABLE_MODELS, compute_saliency
from src.gaze_visual import overlay_saliency, saliency_to_heatmap, mark_peak

MAX_SIDE = 800


def _resize_if_needed(image_bgr):

    h, w = image_bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image_bgr


def analyze(image_rgb, model_name):

    if image_rgb is None:
        return None, None

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_bgr = _resize_if_needed(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    saliency = compute_saliency(model_name, image_bgr)

    overlay = overlay_saliency(image_rgb, saliency, alpha=0.5)
    overlay, _ = mark_peak(overlay, saliency)

    heatmap = saliency_to_heatmap(saliency)

    return overlay, heatmap


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="numpy", label="Immagine"),
        gr.Radio(choices=AVAILABLE_MODELS, value=AVAILABLE_MODELS[0], label="Modello"),
    ],
    outputs=[
        gr.Image(label="Overlay"),
        gr.Image(label="Heatmap"),
    ],
    title="NOVA — Visual Saliency",
    description="Carica un'immagine e scegli un modello per generare la mappa di saliency.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
