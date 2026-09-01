"""
NOVA --- Interfaccia web unificata e semplificata per la visual saliency & gaze annotation.

Uso:
    python app.py

Apre un server web locale a http://localhost:7860 con interfaccia basata su Shadcn UI.
"""

import base64
import os
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from src.registry import AVAILABLE_MODELS, compute_saliency
from src.visualization import overlay_saliency, saliency_to_heatmap, mark_peak
from src.annotation.store import AnnotationStore, MIN_CLICKS, MAX_CLICKS


MAX_SIDE = 800
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "data", "img")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "annotations")
TEMPLATES_DIR = os.path.join(BASE_DIR, "src", "templates")


def get_all_project_images() -> dict:
    """Scansiona la cartella data/img e restituisce nome -> path assoluto."""
    images = {}
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    if os.path.exists(IMG_DIR):
        for fname in sorted(os.listdir(IMG_DIR)):
            if fname.lower().endswith(valid_exts):
                images[fname] = os.path.join(IMG_DIR, fname)
    return images


app = FastAPI(title="NOVA Saliency & Gaze Platform")



class AnalysisRequest(BaseModel):
    image_base64: str
    model: str = "Spectral Residual"
    alpha: float = 0.5
    show_peak: bool = True


class ClickItem(BaseModel):
    x: int
    y: int


class AnnotationSaveRequest(BaseModel):
    annotator: str
    image_name: str
    image_size: List[int]
    clicks: List[ClickItem]


def base64_to_cv2(b64_str: str) -> np.ndarray:
    """Converte una stringa Base64 in un'immagine OpenCV BGR."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Impossibile decodificare l'immagine.")
    return image_bgr


def cv2_to_base64(image_bgr: np.ndarray, ext: str = ".png") -> str:
    """Converte un'immagine OpenCV in una stringa Base64 data URL."""
    _, buffer = cv2.imencode(ext, image_bgr)
    b64_data = base64.b64encode(buffer).decode('utf-8')
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64_data}"


def prepare_image(image_rgb: np.ndarray):
    """Ridimensiona se troppo grande e converte in BGR."""
    height, width = image_rgb.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb,
                               (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
    return image_rgb, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


@app.get("/", response_class=HTMLResponse)
def get_index():
    """Serve la Single Page Application unified con stile Shadcn UI."""
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>NOVA Web App</h1><p>index.html non trovato.</p>"


@app.get("/api/samples")
def get_samples():
    """Restituisce l'elenco delle immagini di esempio disponibili nel progetto."""
    all_images = get_all_project_images()
    samples = [
        {"name": fname, "path": f"/api/sample/{fname}"}
        for fname in all_images.keys()
    ]
    return {"samples": samples}


@app.get("/api/models")
def get_models():
    """Restituisce l'elenco dei modelli di saliency disponibili."""
    return {"models": AVAILABLE_MODELS}


@app.get("/api/sample/{filename}")
def get_sample_file(filename: str):
    """Restituisce il file immagine cercandolo tra le immagini del progetto."""
    all_images = get_all_project_images()
    if filename in all_images and os.path.exists(all_images[filename]):
        return FileResponse(all_images[filename])
    raise HTTPException(status_code=404, detail=f"Immagine '{filename}' non trovata nel progetto.")


@app.post("/api/analyze")
def analyze_image(req: AnalysisRequest):
    """Calcola la saliency map dell'immagine ricevuta in Base64."""
    try:
        image_bgr = base64_to_cv2(req.image_base64)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        image_rgb, image_bgr = prepare_image(image_rgb)
        
        model_name = req.model if req.model in AVAILABLE_MODELS else AVAILABLE_MODELS[0]
        saliency = compute_saliency(model_name, image_bgr)
        
        overlay = overlay_saliency(image_rgb, saliency, alpha=req.alpha)
        heatmap = saliency_to_heatmap(saliency)
        
        peak_x, peak_y = 0, 0
        if req.show_peak:
            overlay, (peak_x, peak_y) = mark_peak(overlay, saliency)
            
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        
        return {
            "success": True,
            "model": model_name,
            "overlay": cv2_to_base64(overlay_bgr),
            "heatmap": cv2_to_base64(heatmap_bgr),
            "width": image_rgb.shape[1],
            "height": image_rgb.shape[0],
            "peak": {"x": int(peak_x), "y": int(peak_y)}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/annotate/session")
def get_annotation_session(annotator: str = "riccardo"):
    """Inizializza o riprende la sessione di annotazione restituendo tutte le immagini e il loro stato."""
    all_images_dict = get_all_project_images()
    all_images = list(all_images_dict.keys())
    
    store = AnnotationStore(annotator, ANNOTATIONS_DIR)
    annotated = store.annotated_images()
    
    items = []
    first_unannotated_index = -1
    
    for idx, img_name in enumerate(all_images):
        is_annotated = img_name in annotated
        clicks = []
        if is_annotated and img_name in store._data.get("annotations", {}):
            clicks = store._data["annotations"][img_name].get("clicks", [])
            
        items.append({
            "name": img_name,
            "annotated": is_annotated,
            "clicks": clicks
        })
        
        if not is_annotated and first_unannotated_index == -1:
            first_unannotated_index = idx

    return {
        "annotator": annotator,
        "images": items,
        "count": store.count(),
        "first_unannotated_index": first_unannotated_index if first_unannotated_index != -1 else 0
    }




@app.post("/api/annotate/save")
def save_annotation(req: AnnotationSaveRequest):
    """Salva i click dell'annotatore per un'immagine."""
    if len(req.clicks) < MIN_CLICKS:
        return {"success": False, "error": f"Servono almeno {MIN_CLICKS} click."}
    if len(req.clicks) > MAX_CLICKS:
        return {"success": False, "error": f"Massimo {MAX_CLICKS} click permessi."}
        
    store = AnnotationStore(req.annotator, ANNOTATIONS_DIR)
    clicks_tuple = [(c.x, c.y) for c in req.clicks]
    
    try:
        store.save_annotation(req.image_name, (req.image_size[0], req.image_size[1]), clicks_tuple)
        return {"success": True, "count": store.count()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def find_available_port(start_port: int = 7860, max_attempts: int = 20) -> int:
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port


if __name__ == "__main__":
    port = find_available_port(7860)
    print(f"\n========================================================")
    print(f"🚀 Avvio NOVA Web App (Shadcn UI Style - Teal Edition)")
    print(f"👉 Apri il browser a: http://localhost:{port}")
    print(f"========================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

