# NOVA — Gaze-guided Salient Object Ranking

Project for **Industrial Applications of Computer Vision**
University of Padova — A.Y. 2025/26

**Authors:** Riccardo Schiavo, Eros Patarini

---

## What it is

Compares six **visual saliency prediction** models (predicting which areas
of an image draw human visual attention) against a ground truth built from
scratch using mouse clicks as a proxy for gaze — the same methodological
principle behind the SALICON dataset — and uses that data to fine-tune one
of the pre-trained models to the project's specific domain.

**Scenario:** first-person images (smart-glasses style), motivated by
assistive technology.

The project runs two distinct experiments on the same set of 23 annotated
photographs:

1. **Pixel-level evaluation** (`evaluate.py`): compares each model's
   saliency map against the click-based ground truth using CC, NSS and
   AUC-Judd.
2. **Object ranking** (`rank_objects.py`): detects objects with YOLOv8,
   ranks them by importance according to each saliency map, and compares
   that ranking against the real click order, using Spearman's ρ and
   Kendall's τ-b.

The two experiments measure different things and can produce different
rankings — this is one of the results discussed in the documentation (see
below).

## Models

| Model | Type | Notes |
|---|---|---|
| **Center Prior** | baseline | Fixed Gaussian at the center, ignores image content — minimum term of comparison |
| **Spectral Residual** (Hou & Zhang, 2007) | classical CV | FFT + spectral residual, implemented from scratch |
| **Itti-Koch** (Itti et al., 1998) | classical CV | Multi-scale center-surround, implemented from scratch |
| **MSI-Net** (Kroner et al., 2020) | deep learning | Pre-trained on SALICON, TensorFlow/Keras via TensorFlow Hub |
| **DeepGaze IIE** (Linardos et al., 2021) | deep learning | Pre-trained, PyTorch |
| **DeepGaze IIE (fine-tuned)** | deep learning | DeepGaze IIE variant fine-tuned on OSIE (`finetune_deepgaze.py`) |

All models expose the same interface via `src/registry.py`: they take an
image and return a `float32` map in `[0, 1]`, with the same dimensions as
the input. Adding a model only requires editing `src/registry.py` — the
rest of the project picks it up automatically.

Object detection for the ranking experiment uses **YOLOv8n** (pre-trained
COCO weights, no training) via the
[`ultralytics`](https://github.com/ultralytics/ultralytics) library
(AGPL-3.0 license).

## Results

On 23 annotated photographs (12 usable for the ranking comparison, due to
the minimum number of objects and clicks required):

**Pixel-level metrics** (`evaluate.py`)

| Model | CC | NSS | AUC-Judd |
|---|---|---|---|
| DeepGaze IIE | 0.406 | 1.996 | **0.910** |
| DeepGaze IIE (fine-tuned) | **0.402** | **2.010** | 0.909 |
| Spectral Residual | 0.313 | 1.303 | 0.852 |
| Center Prior | 0.274 | 0.820 | 0.807 |
| Itti-Koch | 0.216 | 1.237 | 0.835 |
| MSI-Net | 0.212 | 0.641 | 0.776 |

**Object ranking** (`rank_objects.py`)

| Model | Spearman ρ | Kendall τ-b |
|---|---|---|
| MSI-Net | **0.648** | **0.617** |
| Center Prior | 0.485 | 0.468 |
| DeepGaze IIE | 0.485 | 0.468 |
| DeepGaze IIE (fine-tuned) | 0.485 | 0.468 |
| Itti-Koch | 0.363 | 0.327 |
| Spectral Residual | 0.043 | 0.030 |

The two experiments produce different rankings: DeepGaze IIE is best at the
pixel level, MSI-Net is best on object ranking. The full interpretation —
including the statistical limits of the comparison (small sample, rank
correlations computed on very few points per image) — is in the study
document.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The three classical models work out of the box. The deep models and the
ranking experiment need additional dependencies:

```bash
# MSI-Net
pip install tensorflow tensorflow_hub

# DeepGaze IIE
pip install torch einops
pip install git+https://github.com/matthias-k/DeepGaze.git
pip install git+https://github.com/openai/CLIP.git

# Object ranking
pip install ultralytics
```

Pre-trained weights (MSI-Net, DeepGaze IIE, YOLOv8n) download
automatically on first use of each model. The OSIE dataset (used only for
fine-tuning) and the already-downloaded/fine-tuned weights are included
directly in the repository for environment portability — a deliberate
choice made at the cost of a larger repository size.

> Environment note: keep `setuptools` below version 71
> (`pip install "setuptools<71"`) if installing `ultralytics` after
> `tensorflow_hub`, otherwise the latter stops being importable (removal
> of `pkg_resources`).

## Usage

| Command | What it does |
|---|---|
| `python3 app.py` | Web interface: upload an image, pick a model, see the overlay and heatmap |
| `python3 collect.py --images img/` | Click collection: shows images one at a time, 3-6 clicks in order of importance per image |
| `python3 finetune_deepgaze.py --inspect-only` | Lists DeepGaze IIE's parameters, to choose the unfreeze pattern |
| `python3 finetune_deepgaze.py --osie <path> --unfreeze-pattern "saliency_networks" --max-images 200 --epochs 5` | Runs fine-tuning and saves `deepgaze_finetuned.pt` |
| `python3 evaluate.py --images img/ --clicks clicks.json` | Pixel-level evaluation: CC, NSS, AUC-Judd for each model |
| `python3 rank_objects.py --images img/ --clicks clicks.json` | Object ranking: YOLO detection + ranking comparison |

`collect.py` saves everything to a single `clicks.json` in the project
root, updated after each completed image. Reopening it resumes from the
images not yet annotated.

## Structure

```
app.py                     comparison interface (layout only, no computation logic)
collect.py                 click collection for the evaluation ground truth
osie_ground_truth.py       converts OSIE fixations into heatmaps (for training)
finetune_deepgaze.py       fine-tunes DeepGaze IIE on OSIE
finetune_msinet.py         attempted fine-tuning of MSI-Net (documented dead end: frozen checkpoint)
evaluate.py                pixel-level evaluation (CC, NSS, AUC-Judd)
rank_objects.py            object ranking (YOLO + ranking comparison)
clicks.json                clicks collected on the 23 evaluation images
deepgaze_finetuned.pt      DeepGaze IIE weights after fine-tuning
requirements.txt
img/                       the 23 evaluation photographs
predicting-human-gaze-beyond-pixels/   OSIE dataset (fixations.mat + stimuli)
src/
├── registry.py             central registry: which models exist
├── gaze_visual.py          heatmap, overlay, peak-saliency marker
├── baselines/              non-learned models, from scratch
│   ├── center_prior.py
│   ├── spectral_residual.py
│   └── itti_koch.py
└── models/                 wrappers for the pre-trained deep models
    ├── msi_net.py
    └── deepgaze.py
```

**Why `registry.py` and `gaze_visual.py` are kept separate from the app**:
each model has a slightly different interface to load/call; the registry
unifies access in one place. `app.py`, `evaluate.py` and `rank_objects.py`
don't know anything about *how* a model computes saliency — they just ask
for the result.

## Documentation

The project includes three separate documents, each serving a different
purpose:

- **`NOVA_Diario_Sviluppo.pdf`** — development log: decisions, dead ends,
  bugs encountered and how they were resolved (including two
  implementation defects found and fixed during the work: a double
  normalization in the DeepGaze IIE wrapper, and unfrozen normalization
  statistics during fine-tuning).
- **`NOVA_Documento_Studio.pdf`** — theoretical background (visual
  attention, metrics, model families) and a critical reading of the
  experimental results, with explicitly stated limitations.
- **`NOVA_Documentazione_Tecnica.pdf`** — file-by-file explanation of the
  code, in dependency order, with quantitative verification of
  implementation choices where relevant.

## Known limitations

- **Small evaluation sample**: 23 photographs (12 for object ranking), a
  single annotator per image. No difference between closely-ranked models
  has been verified with a statistical test.
- **TensorFlow/PyTorch conflict**: on a machine without a dedicated GPU,
  the two frameworks conflict when both try to initialize CUDA (causing a
  crash). Every entry point sets `CUDA_VISIBLE_DEVICES=""` at the top of
  the file to force CPU execution.
- **A batch normalization statistics defect** during fine-tuning was
  found and fixed during the project (`model.train()` was updating
  statistics even in frozen modules). After the fix, fine-tuning
  DeepGaze IIE shows a real, if modest, benefit: the fine-tuned variant
  surpasses the base model on NSS and stays on par on CC and AUC-Judd.
  Details in the development log and the study document.

## References

- L. Itti, C. Koch, E. Niebur, *A Model of Saliency-Based Visual Attention
  for Rapid Scene Analysis*, IEEE TPAMI, 1998.
- X. Hou, L. Zhang, *Saliency Detection: A Spectral Residual Approach*,
  CVPR, 2007.
- M. Jiang et al., *SALICON: Saliency in Context*, CVPR, 2015.
- A. Kroner et al., *Contextual Encoder-Decoder Network for Visual
  Saliency Prediction*, Neural Networks, 2020. —
  [code](https://github.com/alexanderkroner/saliency)
- A. Linardos et al., *DeepGaze IIE: Calibrated Prediction in and
  Out-of-Domain for State-of-the-Art Saliency Modeling*, ICCV, 2021. —
  [code](https://github.com/matthias-k/DeepGaze)
- G. Jocher et al., *Ultralytics YOLOv8*, 2023. —
  [code](https://github.com/ultralytics/ultralytics) (AGPL-3.0 license)
- OSIE dataset: J. Xu et al., *Predicting Human Gaze Beyond Pixels*,
  Journal of Vision, 2014. —
  [code and data](https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels)

## Declaration on the use of AI/LLM tools

*[Draft — to be reviewed and completed together with Eros before
submission, with the actual detail of who used what. The text below
reflects the usage during this phase of development; it should be
integrated with Eros's own usage, if different.]*

During the project, AI assistance tools based on language models
(Anthropic Claude) were used for:

- **code drafting**: initial drafts of some wrappers and scripts (verified
  and corrected by hand — two implementation defects not caught by the
  first AI-assisted draft were later found and fixed, see
  `NOVA_Diario_Sviluppo.pdf`);
- **debugging**: diagnosis of environment errors (dependency conflicts,
  CUDA segfaults) and of two defects in the code (double normalization in
  `src/models/deepgaze.py`; unfrozen batch normalization statistics in
  `finetune_deepgaze.py`), verified with targeted tests written
  specifically to confirm them before fixing them;
- **documentation writing**: the three PDF documents listed above were
  drafted with AI assistance based on the project's actual code and data,
  with independent verification (running code, cross-checking against the
  official documentation of the libraries used) of the technical claims
  reported.

Every design choice described in the documentation has been understood by
the authors and can be argued independently during the oral discussion.
