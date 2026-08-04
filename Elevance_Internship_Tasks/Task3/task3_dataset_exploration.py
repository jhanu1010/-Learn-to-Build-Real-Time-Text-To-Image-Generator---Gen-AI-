"""
INTERNSHIP TASK 3
Load and examine a public image-caption dataset (Oxford-102 Flowers by
default, COCO Captions as an alternative). Report dataset statistics
(number of classes/images, caption-length distribution, image resolutions)
and display sample images together with their text descriptions.

Two run modes:
  DATASET = "flowers"  -> torchvision.datasets.Flowers102 (auto-downloads
                          images; needs internet + GPU/CPU machine, e.g.
                          Colab). If you also have the Reed et al. caption
                          files (10 captions per image), point
                          CAPTIONS_DIR at the extracted "text_c10" folder
                          and real captions will be used; otherwise a
                          template caption is generated per class so the
                          text-exploration code path still runs end-to-end.
  DATASET = "coco"      -> torchvision.datasets.CocoCaptions (needs the
                          COCO images + annotations downloaded locally,
                          set COCO_ROOT / COCO_ANN_FILE).
  DATASET = "offline_demo" -> a tiny procedurally generated shapes+caption
                          dataset (reuses Task 2's shape drawer) so this
                          script is runnable with zero internet access,
                          e.g. to sanity-check the analysis code.

Run:
    python task3_dataset_exploration.py
Outputs:
    outputs/task3_sample_grid.png
    outputs/task3_caption_length_hist.png
    outputs/task3_resolution_scatter.png
    outputs/task3_stats.json
"""

# CELL 1: Imports & config
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

random.seed(42)
np.random.seed(42)

DATASET = "offline_demo"   # "flowers" | "coco" | "offline_demo"
DATA_ROOT = "data"
CAPTIONS_DIR = None        # path to Reed et al. text_c10/ folder, if available
COCO_ROOT = None
COCO_ANN_FILE = None
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)


# CELL 2: Dataset loaders
def load_flowers102():
    """Loads Oxford-102 Flowers via torchvision (downloads on first run)."""
    import torchvision
    from torchvision.datasets import Flowers102

    ds = Flowers102(root=DATA_ROOT, split="train", download=True)
    class_names = [f"class_{i}" for i in range(102)]  # Flowers102 has no built-in names
    records = []
    for idx in range(len(ds)):
        img, label = ds[idx]
        if CAPTIONS_DIR:
            caption = _load_reed_caption(CAPTIONS_DIR, idx)
        else:
            caption = f"A photograph of an Oxford-102 flower of {class_names[label]}."
        records.append({"image": img, "label": label, "caption": caption})
    return records, 102


def _load_reed_caption(captions_dir, idx):
    # Reed et al. captions are stored as class_XXXXX/image_XXXXX.txt with
    # 10 lines (10 captions). Fall back gracefully if not found.
    try:
        files = sorted(os.listdir(captions_dir))
        with open(os.path.join(captions_dir, files[idx])) as f:
            lines = [l.strip() for l in f if l.strip()]
        return random.choice(lines)
    except Exception:
        return "A flower."


def load_coco():
    """Loads MS-COCO Captions from a local copy (images + annotations)."""
    from torchvision.datasets import CocoCaptions

    ds = CocoCaptions(root=COCO_ROOT, annFile=COCO_ANN_FILE)
    records = []
    for idx in range(len(ds)):
        img, captions = ds[idx]
        records.append({"image": img, "label": None, "caption": random.choice(captions)})
    num_classes = 80  # COCO "things" categories
    return records, num_classes


def load_offline_demo(n_per_class=15):
    """Zero-internet fallback: procedurally generated shapes + captions,
    used only to validate the analysis pipeline below."""
    from task2_cgan_shapes import ShapesDataset, LABELS

    ds = ShapesDataset(n_per_class=n_per_class)
    colour_words = ["white", "pale", "bright", "faint"]
    size_words = ["small", "medium-sized", "large", "compact"]
    records = []
    for i in range(len(ds)):
        label = ds.samples[i]
        img_tensor, label_idx = ds[i]
        arr = ((img_tensor.numpy()[0] + 1) * 127.5).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
        caption = (f"A {random.choice(size_words)}, {random.choice(colour_words)} "
                   f"{label} shape centered on a dark background.")
        records.append({"image": img, "label": label_idx, "caption": caption})
    return records, len(LABELS)


def load_dataset():
    if DATASET == "flowers":
        return load_flowers102()
    elif DATASET == "coco":
        return load_coco()
    elif DATASET == "offline_demo":
        return load_offline_demo()
    raise ValueError(f"Unknown DATASET={DATASET}")


# CELL 3: Dataset statistics
def compute_stats(records, num_classes):
    caption_lengths = [len(r["caption"].split()) for r in records]
    resolutions = [r["image"].size for r in records]  # (W, H)
    widths = [w for w, h in resolutions]
    heights = [h for w, h in resolutions]

    stats = {
        "num_samples": len(records),
        "num_classes": num_classes,
        "caption_length": {
            "mean": float(np.mean(caption_lengths)),
            "min": int(np.min(caption_lengths)),
            "max": int(np.max(caption_lengths)),
            "std": float(np.std(caption_lengths)),
        },
        "image_resolution": {
            "mean_width": float(np.mean(widths)),
            "mean_height": float(np.mean(heights)),
            "min_width": int(np.min(widths)),
            "max_width": int(np.max(widths)),
        },
    }
    return stats, caption_lengths, resolutions


# CELL 4: Visualization - sample images with captions
def plot_sample_grid(records, n=9):
    samples = random.sample(records, min(n, len(records)))
    cols = 3
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax, rec in zip(axes, samples):
        ax.imshow(rec["image"])
        ax.set_title(rec["caption"], fontsize=8, wrap=True)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.suptitle(f"Task 3: sample images + captions ({DATASET})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task3_sample_grid.png"))
    plt.close()


def plot_caption_lengths(caption_lengths):
    plt.figure()
    plt.hist(caption_lengths, bins=15)
    plt.xlabel("Caption length (words)")
    plt.ylabel("Count")
    plt.title("Task 3: Caption length distribution")
    plt.savefig(os.path.join(OUT_DIR, "task3_caption_length_hist.png"))
    plt.close()


def plot_resolutions(resolutions):
    widths = [w for w, h in resolutions]
    heights = [h for w, h in resolutions]
    plt.figure()
    plt.scatter(widths, heights, alpha=0.5, s=10)
    plt.xlabel("Width (px)"); plt.ylabel("Height (px)")
    plt.title("Task 3: Image resolution scatter")
    plt.savefig(os.path.join(OUT_DIR, "task3_resolution_scatter.png"))
    plt.close()


if __name__ == "__main__":
    print(f"Loading dataset: {DATASET}")
    records, num_classes = load_dataset()
    stats, caption_lengths, resolutions = compute_stats(records, num_classes)
    print(json.dumps(stats, indent=2))

    with open(os.path.join(OUT_DIR, "task3_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    plot_sample_grid(records)
    plot_caption_lengths(caption_lengths)
    plot_resolutions(resolutions)
    print(f"Saved plots and stats to {OUT_DIR}/")
