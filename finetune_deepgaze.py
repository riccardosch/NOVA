"""
finetune_deepgaze.py --- fine-tuning dell'ultimo blocco di DeepGaze IIE su OSIE.

Uso (due fasi, nell'ordine):

    # FASE 1 -- diagnostica: mostra i nomi dei parametri allenabili
    python finetune_deepgaze.py --osie predicting-human-gaze-beyond-pixels --inspect-only

    # FASE 2 -- training vero, dopo aver scelto --unfreeze-pattern dalla FASE 1
    python finetune_deepgaze.py --osie predicting-human-gaze-beyond-pixels \
        --unfreeze-pattern "<pattern scelto>" --max-images 200 --epochs 5

Perche' DeepGaze IIE e non MSI-Net: il checkpoint di MSI-Net pubblicato su
Kaggle e' congelato (nessuna variabile allenabile, verificato). I modelli
PyTorch come DeepGaze IIE non hanno questo problema per costruzione: sono
sempre veri oggetti allenabili (nn.Module con parametri veri), non esistono
in una forma "solo inferenza" come i SavedModel TensorFlow pubblicati per
servire previsioni.

Il meccanismo di freeze qui e' anche piu' semplice e affidabile di quello
usato per MSI-Net: si imposta requires_grad=False su tutti i parametri,
poi lo si riaccende solo per quelli scelti. E' il modo standard e
documentato in PyTorch (a differenza del tentativo fragile con
Variable._trainable che avevamo scartato per TensorFlow).
"""

import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osie_ground_truth import load_fixations, points_to_heatmap


TARGET_HEIGHT, TARGET_WIDTH = 240, 320  # risoluzione di lavoro per il training


# ---------------------------------------------------------------------------
# Dati: OSIE -> coppie (immagine, heatmap) in formato PyTorch (NCHW)
# ---------------------------------------------------------------------------

def build_dataset(osie_dir, max_images=None):
    """Costruisce le coppie (immagine, heatmap) pronte per DeepGaze IIE.

    A differenza di MSI-Net (TensorFlow, formato NHWC -- canali per ultimi),
    DeepGaze IIE si aspetta PyTorch, formato NCHW -- canali per primi.

    Returns
    -------
    list[tuple(np.ndarray, np.ndarray)]
        (immagine RGB float32 shape (3, H, W), heatmap float32 shape (H, W))
    """
    mat_path = os.path.join(osie_dir, "data", "eye", "fixations.mat")
    stimuli_dir = os.path.join(osie_dir, "data", "stimuli")

    fixations = load_fixations(mat_path)
    names = list(fixations.keys())
    if max_images is not None:
        names = names[:max_images]

    pairs = []
    skipped = 0
    for name in names:
        path = os.path.join(stimuli_dir, name)
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            skipped += 1
            continue

        original_h, original_w = image_bgr.shape[:2]

        heatmap = points_to_heatmap(fixations[name], (original_h, original_w))
        heatmap = cv2.resize(heatmap, (TARGET_WIDTH, TARGET_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (TARGET_WIDTH, TARGET_HEIGHT),
                                   interpolation=cv2.INTER_AREA)
        # NHW C -> CHW: PyTorch vuole i canali come prima dimensione.
        image_chw = image_resized.transpose(2, 0, 1).astype(np.float32)

        pairs.append((image_chw, heatmap.astype(np.float32)))

    if skipped:
        print(f"  ATTENZIONE: {skipped} immagini elencate in fixations.mat "
              f"non trovate in {stimuli_dir}, saltate.")

    return pairs


def build_centerbias(height, width):
    """Center bias uniforme: stesso identico usato in src/models/deepgaze.py.

    Nessun vantaggio al centro dell'immagine, per non favorire questo
    modello rispetto ai tre matematici, che non ricevono alcun prior.
    """
    import torch
    centerbias = np.zeros((height, width), dtype=np.float64)
    log_sum = np.log(np.exp(centerbias).sum())
    centerbias = centerbias - log_sum
    return torch.tensor(centerbias, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------

def load_trainable_model(device="cpu"):
    """Carica DeepGaze IIE. A differenza di MSI-Net, e' gia' un vero
    oggetto PyTorch allenabile: nessun bypass o caricamento grezzo necessario.
    """
    import deepgaze_pytorch
    model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    return model


def inspect_parameters(model):
    """Stampa tutti i parametri allenabili, coi nomi che preservano la
    gerarchia dei moduli (a differenza dei nomi piatti visti con Keras 3).
    """
    named = list(model.named_parameters())
    print(f"\nParametri allenabili totali: {len(named)}\n")
    for name, p in named:
        print(f"  {name:<70} shape={tuple(p.shape)}")

    print(
        "\nCerca i nomi che sembrano appartenere agli ULTIMI moduli della "
        "rete (spesso le 'teste' di lettura finali, o i layer con indice "
        "piu' alto). La sottostringa comune a quei nomi e' il valore da "
        "passare a --unfreeze-pattern."
    )


def select_trainable_parameters(model, unfreeze_pattern):
    """Congela tutto, poi riaccende solo i parametri il cui nome contiene
    il pattern.

    A differenza del tentativo scartato per TensorFlow (modificare un
    flag interno dopo la creazione, non garantito), in PyTorch impostare
    requires_grad e' il meccanismo standard e documentato -- verificato
    che funziona davvero con un test isolato prima di usarlo qui.

    Returns
    -------
    list[torch.nn.Parameter]
        I parametri rimasti allenabili, da passare all'ottimizzatore.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    for name, p in model.named_parameters():
        if unfreeze_pattern in name:
            p.requires_grad = True
            trainable.append(p)

    return trainable


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, trainable_params, pairs, epochs, batch_size, learning_rate, out_path):
    import torch

    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)
    mse = torch.nn.MSELoss()

    # Il center bias dipende solo dalle dimensioni, non dal contenuto
    # dell'immagine: si costruisce una volta sola e si riusa per tutti i
    # batch, dato che lavoriamo a risoluzione fissa (TARGET_HEIGHT x WIDTH).
    centerbias = build_centerbias(TARGET_HEIGHT, TARGET_WIDTH)

    n = len(pairs)
    model.train()
    # model.train() rimette in modalita' addestramento anche i BatchNorm dei
    # moduli congelati: requires_grad=False protegge i pesi, ma non i buffer
    # (running_mean/running_var), che vengono aggiornati a ogni forward
    # indipendentemente da requires_grad. Li si riporta esplicitamente in
    # eval() per lasciarli davvero intatti.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    for epoch in range(epochs):
        order = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_size_actual = len(idx)

            images = torch.tensor(np.stack([pairs[i][0] for i in idx]))
            targets = torch.tensor(np.stack([pairs[i][1] for i in idx]))
            batch_centerbias = centerbias.unsqueeze(0).repeat(batch_size_actual, 1, 1)

            optimizer.zero_grad()

            log_density = model(images, batch_centerbias)

            # L'uscita e' una log-density (come in src/models/deepgaze.py):
            # va convertita in mappa normale per confrontarla con la
            # heatmap target. Sottrarre il massimo prima di exp() evita
            # overflow numerico.
            log_density_flat = log_density.view(batch_size_actual, -1)
            max_per_image = log_density_flat.max(dim=1, keepdim=True).values
            density = torch.exp(log_density_flat - max_per_image)
            min_per_image = density.min(dim=1, keepdim=True).values
            max_density = density.max(dim=1, keepdim=True).values
            predicted = (density - min_per_image) / (max_density - min_per_image + 1e-8)
            predicted = predicted.view(batch_size_actual, TARGET_HEIGHT, TARGET_WIDTH)

            loss = mse(predicted, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"epoca {epoch + 1}/{epochs}  loss media: {epoch_loss / n_batches:.6f}")

    import torch as torch_module
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch_module.save(model.state_dict(), out_path)
    print(f"\nPesi completi salvati in '{out_path}'.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning di DeepGaze IIE su OSIE.")
    parser.add_argument("--osie", required=True,
                        help="Cartella del repository OSIE clonato.")
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-pattern", type=str, default=None,
                        help="Sottostringa dei nomi dei parametri da NON congelare. "
                             "Obbligatorio a meno di --inspect-only.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--out", default="deepgaze_finetuned.pt")
    args = parser.parse_args()

    print(f"Preparazione dati da OSIE (fino a {args.max_images} immagini)...")
    pairs = build_dataset(args.osie, max_images=args.max_images)
    print(f"Coppie (immagine, heatmap) pronte: {len(pairs)}")

    print("\nCaricamento di DeepGaze IIE...")
    model = load_trainable_model()

    if args.inspect_only:
        inspect_parameters(model)
        raise SystemExit(0)

    if not args.unfreeze_pattern:
        raise SystemExit(
            "Serve --unfreeze-pattern. Lancia prima con --inspect-only per "
            "vedere i nomi dei parametri e scegliere il pattern giusto."
        )

    trainable_params = select_trainable_parameters(model, args.unfreeze_pattern)
    total = sum(1 for _ in model.parameters())
    print(f"Parametri selezionati per il training: {len(trainable_params)} su {total} totali.")
    if len(trainable_params) == 0:
        raise SystemExit(
            f"Nessun parametro corrisponde al pattern '{args.unfreeze_pattern}'. "
            "Ricontrolla l'output di --inspect-only."
        )
    if len(trainable_params) == total:
        print(
            "ATTENZIONE: il pattern corrisponde a TUTTI i parametri -- "
            "stai allenando l'intera rete, non solo l'ultimo blocco."
        )

    train(model, trainable_params, pairs, args.epochs, args.batch_size,
          args.learning_rate, args.out)