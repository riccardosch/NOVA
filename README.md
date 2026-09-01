# NOVA — Gaze-guided Salient Object Ranking

Progetto per **Industrial Applications of Computer Vision**
Università degli Studi di Padova — A.A. 2025/26

**Autori:** Riccardo Schiavo, Eros Patarini

---

## Cos'è

Confronta modelli di **visual saliency prediction** (predizione delle zone che attirano
l'attenzione visiva) valutandoli contro una ground truth costruita con **click del mouse come
proxy del gaze** — lo stesso principio metodologico del dataset SALICON.

Come passo finale, le saliency map vengono combinate con dei bounding box per produrre un
**ranking degli oggetti** per importanza visiva.

**Scenario:** immagini in prima persona (smart glasses), con motivazione di assistive technology.

## Input e output

| | |
|---|---|
| **Input** | Una singola immagine RGB |
| **Output 1** | Una saliency map: `float32`, `[0, 1]`, stesse dimensioni dell'immagine |
| **Output 2** | Ranking degli oggetti per importanza visiva |
| **Output 3** | Tabelle di metriche che quantificano l'accordo con la ground truth |

## Modelli

| Modello | Tipo |
|---|---|
| **Center Prior** | Baseline: Gaussiana centrata, ignora il contenuto |
| **Spectral Residual** (Hou & Zhang, 2007) | CV tradizionale: FFT + residuo spettrale |
| **Itti-Koch** (Itti et al., 1998) | CV tradizionale: center-surround multi-scala |

Tutti implementati da zero.

## Metriche

**Intrinseche** — saliency map contro ground truth:
CC, SIM, KL (⚠️ basso = meglio), NSS, AUC-Judd.

**Estrinseche** — ranking degli oggetti:
Spearman, Kendall τ-b, Top-1 accuracy.

Tutte implementate da zero.

## Installazione

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

| Comando | Cosa fa |
|---|---|
| `python app.py` | Interfaccia web: carica un'immagine, scegli un modello, vedi il risultato |
| `python annotate.py --images data/eval` | Interfaccia web di annotazione a click |
| `python scripts/run_evaluation.py` | Valutazione su tutto il dataset → CSV in `results/` |
| `python tests/test_all.py` | 56 test |

## Struttura

```
app.py                     interfaccia di confronto
annotate.py                interfaccia di annotazione
src/
├── registry.py            registro centrale dei modelli
├── visualization.py       heatmap, overlay, marcatori
├── baselines/             i tre modelli, implementati da zero
├── annotation/            persistenza e click → ground truth
├── evaluation/metrics.py  tutte le metriche
└── ranking/objects.py     ranking degli oggetti
scripts/run_evaluation.py  valutazione → tabelle CSV
tests/test_all.py          suite di test
docs/                      documentazione completa (PDF + sorgente LaTeX)
```

**Come aggiungere un modello:** una riga in `src/registry.py`. Comparirà automaticamente
nell'app, nei test e nella valutazione.

## Ground truth

L'annotatore clicca **in ordine di importanza decrescente**, da 3 a 6 volte per immagine. Un solo
compito produce due ground truth: le posizioni (per NSS/AUC-Judd) e l'ordine (per il ranking).

I click diventano Gaussiane con σ pari al **5% della larghezza** — approssima l'ampiezza della
fovea (~1° di angolo visivo). I click di tutti gli annotatori confluiscono in un'unica nuvola: le
zone di consenso emergono naturalmente.

Formato: un file JSON per annotatore in `data/annotations/`, salvataggio incrementale e atomico.

## Limitazioni dichiarate

- **I click non sono fissazioni oculari**: il click è deliberato, la fissazione in parte
  involontaria.
- **Nessun limite di tempo** di osservazione (SALICON usa un protocollo più elaborato).
- **Numero limitato di annotatori** rispetto ai benchmark di riferimento.
- **Spectral Residual è cieco al colore puro**: lavora in scala di grigi, quindi un oggetto
  isoluminante con lo sfondo gli è invisibile (verificato numericamente nei test).
- **Itti-Koch semplificato**: scale center-surround fisse e normalizzazione a singolo passaggio
  invece che iterativa.
- **Assegnazione click → oggetto per euristica**: con box sovrapposti vince il più piccolo.

## Documentazione

`docs/NOVA_Documentazione.pdf` — Parte I: funzionamento generale e architettura. Parte II: il
codice sezionato riga per riga, file per file.

Compilazione del sorgente: `xelatex -shell-escape NOVA_Documentazione.tex` (due volte, per
l'indice). Richiede `minted` e `pygments`.

## Riferimenti

- L. Itti, C. Koch, E. Niebur, *A Model of Saliency-Based Visual Attention for Rapid Scene
  Analysis*, IEEE TPAMI, 1998.
- X. Hou, L. Zhang, *Saliency Detection: A Spectral Residual Approach*, CVPR, 2007.
- M. Jiang et al., *SALICON: Saliency in Context*, CVPR, 2015.

**Librerie:** NumPy (BSD), OpenCV (MIT), Gradio (Apache 2.0).

## Dichiarazione sull'uso di strumenti AI/LLM

*Da compilare con quanto effettivamente accaduto: dove sono stati usati strumenti AI e per cosa.*
