# finetune_msinet.py — Fine-tune the MSI-Net saliency model on OSIE data.
# Downloads MSI-Net from TensorFlow Hub, optionally inspects its variables,
# then trains a selected subset of layers on OSIE fixation heatmaps.

import argparse  # Command-line argument parsing
import os        # Filesystem operations
import sys       # System-level utilities (e.g. path manipulation)

import numpy as np  # Numerical operations (arrays, math)
import cv2          # OpenCV — image reading, resizing, color conversion

# Make sure the project root is in the Python path so local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osie_ground_truth import load_fixations, points_to_heatmap


# URL of the pre-trained MSI-Net model on TensorFlow Hub / Kaggle
MODEL_URL = "https://www.kaggle.com/models/alexanderkroner/msi-net/tensorFlow2/salicon/1"
# Input resolution expected by MSI-Net
INPUT_HEIGHT, INPUT_WIDTH = 240, 320


# ---------------------------------------------------------------------------
# Data: OSIE -> (image, heatmap) pairs ready for training
# ---------------------------------------------------------------------------

def build_dataset(osie_dir, max_images=None):
    """Load OSIE images and fixation data, returning (image, heatmap) pairs.

    Each image is resized to (INPUT_HEIGHT, INPUT_WIDTH) and normalized to [0, 1].
    Heatmaps are built at the original resolution first, then resized.
    """
    mat_path = os.path.join(osie_dir, "data", "eye", "fixations.mat")
    stimuli_dir = os.path.join(osie_dir, "data", "stimuli")

    # Load all fixation points grouped by image name
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

        # The heatmap is built at the ORIGINAL resolution (fixation
        # coordinates are in that coordinate system), then resized:
        # building it directly at the small size would produce a
        # Gaussian with an incorrect sigma.
        heatmap = points_to_heatmap(fixations[name], (original_h, original_w))
        heatmap = cv2.resize(heatmap, (INPUT_WIDTH, INPUT_HEIGHT),
                             interpolation=cv2.INTER_AREA)

        # Convert BGR -> RGB and resize the image to the model input size
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT),
                                   interpolation=cv2.INTER_AREA)
        # Normalize pixel values to the [0, 1] range
        image_normalized = image_resized.astype(np.float32) / 255.0

        pairs.append((image_normalized, heatmap.astype(np.float32)))

    if skipped:
        print(f"  WARNING: {skipped} images listed in fixations.mat "
              f"were not found in {stimuli_dir}, skipped.")

    return pairs


# ---------------------------------------------------------------------------
# Model: load in trainable mode + choose which variables to freeze
# ---------------------------------------------------------------------------

def load_trainable_model():
    """Download/load MSI-Net from TF Hub and extract its trainable variables.

    Returns (infer, trainable_vars) where *infer* is the serving signature
    and *trainable_vars* is a list of tf.Variable objects.
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    # Resolve the model URL to a local cache path and load the SavedModel
    local_path = hub.resolve(MODEL_URL)
    loaded = tf.saved_model.load(local_path)
    infer = loaded.signatures["serving_default"]

    trainable_vars = list(infer.trainable_variables)
    if not trainable_vars:
        print(
            "\nWARNING: no trainable variables found even with raw "
            "loading. The model may have been published in a fully "
            "frozen form (constant weights, not variables). In that "
            "case, fine-tuning is not possible with this checkpoint."
        )

    return infer, trainable_vars


def call_signature(infer, batch_images):
    """Run the model's serving signature on a batch of images."""
    outputs = infer(batch_images)
    # The signature may have an arbitrary output key; just grab the first value
    return list(outputs.values())[0]


def inspect_variables(infer, trainable_vars):
    """Print all trainable variable names and the signature's I/O spec.

    Useful for choosing the --unfreeze-pattern before training.
    """
    print(f"\nTotal trainable variables: {len(trainable_vars)}\n")
    for v in trainable_vars:
        print(f"  {v.name:<70} shape={tuple(v.shape)}")

    print("\nSignature spec (expected inputs and outputs):")
    print(infer.pretty_printed_signature())

    print(
        "\nLook at the variable names for those that appear to belong to "
        "the LAST layers of the network (they usually appear last, or "
        "have higher block/index numbers). The common substring of those "
        "names is the value to pass to --unfreeze-pattern."
    )


def select_trainable_variables(all_trainable_vars, unfreeze_pattern):
    """Return only the variables whose name contains *unfreeze_pattern*."""
    return [v for v in all_trainable_vars if unfreeze_pattern in v.name]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(infer, trainable_vars, pairs, epochs, batch_size, learning_rate, out_dir):
    """Run the fine-tuning loop and save the resulting weights.

    Only the variables in *trainable_vars* receive gradient updates;
    all other model weights remain frozen.
    """
    import tensorflow as tf

    # Stack all images and targets into contiguous NumPy arrays
    images = np.stack([p[0] for p in pairs])
    targets = np.stack([p[1] for p in pairs])[..., np.newaxis]  # Add channel dim

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    mse = tf.keras.losses.MeanSquaredError()

    n = len(pairs)
    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        order = np.random.permutation(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch_images = tf.constant(images[idx])
            batch_targets = tf.constant(targets[idx])

            with tf.GradientTape() as tape:
                # The forward pass uses the entire signature: all variables
                # contribute to producing the output, frozen or not — only
                # the gradient is requested selectively, right below.
                predictions = call_signature(infer, batch_images)
                predictions = tf.reshape(predictions, tf.shape(batch_targets))
                loss = mse(batch_targets, predictions)

            # Compute gradients only for the selected (unfrozen) variables
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            epoch_loss += float(loss)
            n_batches += 1

        print(f"epoch {epoch + 1}/{epochs}  average loss: {epoch_loss / n_batches:.6f}")

    # This is not a tf.keras.Model, so model.save() cannot be used.
    # Instead, save the trained variable values directly in a simple
    # format that is easy to reload (weights of the fine-tuned block only).
    os.makedirs(out_dir, exist_ok=True)
    weights = {v.name: v.numpy() for v in trainable_vars}
    np.savez(os.path.join(out_dir, "finetuned_weights.npz"), **{
        # Colons in tensor names (e.g. "conv5/kernel:0") are not valid
        # as file keys; replace them for safety.
        name.replace(":", "_").replace("/", "__"): value
        for name, value in weights.items()
    })
    print(f"\nTrained weights saved to '{out_dir}/finetuned_weights.npz' "
          f"({len(weights)} variables).")


# ---------------------------------------------------------------------------
# Entry point — parse CLI arguments and run fine-tuning (or inspection)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning MSI-Net on OSIE.")
    parser.add_argument("--osie", required=True,
                        help="Folder of the cloned OSIE repository.")
    parser.add_argument("--max-images", type=int, default=200,
                        help="How many OSIE images to use (default: 200, out of ~700 total).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-pattern", type=str, default=None,
                        help="Substring of variable names to NOT freeze. "
                             "Required unless --inspect-only is used.")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Only print the variable names and exit, without training.")
    parser.add_argument("--out", default="msinet_finetuned")
    args = parser.parse_args()

    # Build the (image, heatmap) dataset from OSIE
    print(f"Preparing data from OSIE (up to {args.max_images} images)...")
    pairs = build_dataset(args.osie, max_images=args.max_images)
    print(f"(image, heatmap) pairs ready: {len(pairs)}")

    # Load MSI-Net with direct SavedModel access
    print("\nLoading MSI-Net (direct SavedModel access)...")
    infer, all_trainable_vars = load_trainable_model()

    # If inspect-only mode, print variables and exit
    if args.inspect_only:
        inspect_variables(infer, all_trainable_vars)
        raise SystemExit(0)

    # Require --unfreeze-pattern for actual training
    if not args.unfreeze_pattern:
        raise SystemExit(
            "--unfreeze-pattern is required. Run with --inspect-only first "
            "to see the variable names and choose the right pattern."
        )

    # Select only the variables matching the unfreeze pattern
    trainable_vars = select_trainable_variables(all_trainable_vars, args.unfreeze_pattern)
    total = len(all_trainable_vars)
    print(f"Variables selected for training: {len(trainable_vars)} out of {total} total.")
    if len(trainable_vars) == 0:
        raise SystemExit(
            f"No variable matches the pattern '{args.unfreeze_pattern}'. "
            "Double-check the output of --inspect-only."
        )
    if len(trainable_vars) == total:
        print(
            "WARNING: the pattern matches ALL variables — you are training "
            "the entire network, not just the last block. "
            "Make sure this is intentional."
        )

    # Run the fine-tuning loop
    train(infer, trainable_vars, pairs, args.epochs, args.batch_size,
          args.learning_rate, args.out)
