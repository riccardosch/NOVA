# NOVA — Gaze-guided Salient Object Ranking

Progetto per **Industrial Applications of Computer Vision**
Università degli Studi di Padova — A.A. 2025/26

**Autori:** Riccardo Schiavo, Eros Patarini

---

## Cos'è

Confronta modelli di **visual saliency prediction** (predizione delle zone di un'immagine che attirano l'attenzione visiva umana). L'obiettivo finale del progetto è valutarli contro una ground truth costruita con **click del mouse come proxy del gaze** — lo stesso principio metodologico del dataset SALICON — e usare quei dati per adattare (fine-tuning) uno dei modelli pre-addestrati al dominio specifico del progetto.

**Scenario:** immagini in prima persona (stile smart glasses), con motivazione di assistive technology.

## Stato attuale

✅ Cinque modelli di saliency, tutti funzionanti e verificati.
✅ Interfaccia web per confrontarli su un'immagine a scelta.
✅ Strumento di raccolta click, pronto all'uso.
🔲 Raccolta del dataset reale (foto in prima persona) — non ancora fatta.
🔲 Fine-tuning di un modello sui dati raccolti — in attesa del dataset.
🔲 Valutazione quantitativa e ranking degli oggetti (YOLO) — non ancora implementati.

## Modelli

| Modello | Tipo | Note |
|---|---|---|
| **Center Prior** | baseline | Gaussiana fissa al centro, ignora il contenuto — termine minimo di paragone |
| **Spectral Residual** (Hou & Zhang, 2007) | CV tradizionale | FFT + residuo spettrale, implementato da zero |
| **Itti-Koch** (Itti et al., 1998) | CV tradizionale | Center-surround multi-scala, implementato da zero |
| **MSI-Net** (Kroner et al., 2020) | deep learning | Pre-addestrato su SALICON, TensorFlow/Keras via TensorFlow Hub |
| **DeepGaze IIE** (Linardos et al., 2021) | deep learning | Pre-addestrato, PyTorch |

Tutti espongono la stessa firma tramite `src/registry.py`: ricevono un'immagine e restituiscono una mappa `float32` in `[0, 1]`, delle stesse dimensioni dell'input. Aggiungere un modello significa modificare solo `src/registry.py` — il resto del progetto lo vede automaticamente.

## Installazione

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

I tre modelli classici funzionano subito. I due deep richiedono dipendenze aggiuntive (commentate in `requirements.txt`, non installate di default per non appesantire l'ambiente a chi non ne ha bisogno):

```bash
# MSI-Net
pip install tensorflow tensorflow_hub

# DeepGaze IIE
pip install torch einops
pip install git+https://github.com/matthias-k/DeepGaze.git
pip install git+https://github.com/openai/CLIP.git
```

I pesi pre-addestrati si scaricano automaticamente al primo utilizzo di ciascun modello (qualche centinaio di MB, serve una connessione attiva la prima volta).

## Uso

| Comando | Cosa fa |
|---|---|
| `python app.py` | Interfaccia web: carica un'immagine, scegli un modello dal radio button, vedi overlay e heatmap |
| `python collect.py --images img/` | Raccolta click: mostra le immagini una alla volta, 3-6 click in ordine di importanza per immagine |

`collect.py` salva tutto in un unico `clicks.json` nella root, aggiornato dopo ogni immagine completata. Riaprendolo, riprende dalle immagini non ancora fatte.

## Struttura

```
app.py                     interfaccia di confronto (solo layout, nessuna logica di calcolo)
collect.py                 raccolta click per la ground truth
requirements.txt
img/                       immagini di esempio per la demo
src/
├── registry.py             registro centrale: quali modelli esistono
├── gaze_visual.py          heatmap, overlay, marcatore del punto più saliente
├── baselines/              modelli senza apprendimento, da zero
│   ├── center_prior.py
│   ├── spectral_residual.py
│   └── itti_koch.py
└── models/                 wrapper dei modelli deep pre-addestrati
    ├── msi_net.py
    └── deepgaze.py
```

**Perché `registry.py` e `gaze_visual.py` sono separati dall'app**: ogni modello ha una firma leggermente diversa da caricare/chiamare; il registro uniforma l'accesso in un punto solo. `app.py` non sa nulla di *come* un modello calcola la saliency — chiede solo il risultato.

## Limitazioni note

- **DeepGaze IIE e MSI-Net non possono girare insieme senza un accorgimento**: TensorFlow e PyTorch, sulla stessa macchina senza GPU dedicata, entrano in conflitto inizializzando CUDA contemporaneamente (causa un crash). `app.py` imposta `CUDA_VISIBLE_DEVICES=""` in cima al file per forzare entrambi su CPU ed evitare il problema.
- **Nessuna valutazione quantitativa ancora**: i modelli si possono confrontare solo visivamente, tramite l'app. Le metriche (CC, NSS, AUC-Judd, ecc.) sono un passo successivo, in attesa del dataset annotato.
- **Il dataset di valutazione non esiste ancora**: le tre immagini in `img/` servono solo da demo, non sono il dataset del progetto.

## Riferimenti

- L. Itti, C. Koch, E. Niebur, *A Model of Saliency-Based Visual Attention for Rapid Scene Analysis*, IEEE TPAMI, 1998.
- X. Hou, L. Zhang, *Saliency Detection: A Spectral Residual Approach*, CVPR, 2007.
- M. Jiang et al., *SALICON: Saliency in Context*, CVPR, 2015.
- A. Kroner et al., *Contextual Encoder-Decoder Network for Visual Saliency Prediction*, Neural Networks, 2020. — [codice](https://github.com/alexanderkroner/saliency)
- A. Linardos et al., *DeepGaze IIE*, ICCV, 2021. — [codice](https://github.com/matthias-k/DeepGaze)

## Dichiarazione sull'uso di strumenti AI/LLM

*Da compilare con quanto effettivamente accaduto prima della consegna — dove sono stati usati strumenti AI (progettazione, generazione di codice, debug) e per cosa, in modo specifico e onesto, non genericamente.*
