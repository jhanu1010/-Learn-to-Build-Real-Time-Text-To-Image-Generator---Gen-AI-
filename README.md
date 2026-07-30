# Learn to Build Real-Time Text-To-Image Generator - Gen-AI


# Real-Time Text-to-Image Generator - Gen AI Internship Project

This repository builds on my training project (a Stable Diffusion + Gradio
text-to-image app) and extends it with 6 internship tasks covering
fine-tuning, conditional GANs, dataset exploration, text preprocessing,
attention mechanisms, and a full end-to-end generation pipeline.

## Folder Structure

```
.
├── README.md
├── training_project/                    # Original training project 
│   ├── colab1.ipynb
│   └── colab_cell_by_cell.py
│
├── Task1_FineTune_Diffusion/            # Task 1
│   ├── task1_finetune_diffusion.py
│   └── task1_finetune_diffusion.ipynb
│
├── Task2_CGAN_Shapes/                   # Task 2
│   ├── task2_cgan_shapes.py
│   ├── task2_cgan_shapes.ipynb
│   ├── task2_samples.png
│   └── task2_training_curve.png
│
├── Task3_Dataset_Exploration/           # Task 3
│   ├── task3_dataset_exploration.py
│   ├── task3_dataset_exploration.ipynb
│   ├── task2_cgan_shapes.py             
│   ├── task3_sample_grid.png
│   ├── task3_caption_length_hist.png
│   ├── task3_resolution_scatter.png
│   └── task3_stats.json
│
├── Task4_Text_Preprocessing/            # Task 4
│   ├── task4_text_preprocessing.py
│   ├── task4_text_preprocessing.ipynb
│   ├── task4_tokenization_example.json
│   └── task4_embedding_similarity.png
│
├── Task5_Attention_GAN/                 # Task 5
│   ├── task5_attention_gan.py
│   ├── task5_attention_gan.ipynb
│   ├── task2_cgan_shapes.py               
│   ├── task4_text_preprocessing.py        
│   ├── task5_samples.png
│   └── task5_training_curve.png
│
├── Task6_Full_Pipeline/                 # Task 6
│   ├── task6_full_pipeline.py
│   ├── task6_full_pipeline.ipynb
│   ├── task2_cgan_shapes.py               
│   ├── task4_text_preprocessing.py        
│   ├── task5_attention_gan.py              
│   ├── task5_generator.pt
│   └── task6_pipeline_demo.png
│
└── datasets/                             # Datasets used across tasks
    ├── custom_domain_dataset_task1/
    ├── shapes_dataset_task2/
    └── shapes_captioned_dataset_task3/
```

Each task folder contains both a plain `.py` script (cell-marked, same
style as `training_project/colab_cell_by_cell.py`) and a matching `.ipynb`
notebook (same style as `training_project/colab1.ipynb`), plus the actual
output plots/files produced by running it. Task folders that build on an
earlier task include a copy of that task's file so each folder can be run
on its own, without needing the whole repo checked out in a specific order.

---

## Task 1 - Fine-tune a Pre-trained Text-to-Image Model on a Custom Dataset

**What it does:** Takes the pre-trained Stable Diffusion model from the
training project and fine-tunes it on a small custom domain dataset (e.g.
a specific art style, or medical-style imagery) so it learns to produce
domain-specific visuals tied to a unique keyword.

**Approach:** LoRA (Low-Rank Adaptation) fine-tuning of the UNet's
cross-attention layers — the standard, memory-efficient way to adapt
Stable Diffusion to a new visual style without retraining the full model.
Each training image is paired with a caption containing a special token
(`sks-domainart`) so the fine-tuned style is bound to that keyword.

**Files:** `Task1_FineTune_Diffusion/task1_finetune_diffusion.py` /
`.ipynb`. Includes a placeholder dataset generator (so the expected
image+caption folder format is demonstrated even before real domain
images are added) and a before/after image comparison script.

**Note:** requires internet access (to download `runwayml/stable-diffusion-v1-5`
from Hugging Face) and a GPU — run this one in Google Colab.

---

## Task 2 - Conditional GAN (CGAN) for Basic Shapes

**What it does:** A GAN that generates basic shapes conditioned on a text
label — give it "circle", "square", "triangle", or "star" and it
generates that shape, instead of a random image. This introduces
conditional inputs into GAN training.

**Approach:** A generator and discriminator that both take a label
embedding alongside the usual noise vector / image input, trained on a
procedurally generated 32×32 grayscale shapes dataset (500 images per
class, drawn with PIL — no external dataset needed).

**Files:** `Task2_CGAN_Shapes/task2_cgan_shapes.py` / `.ipynb`. Trained
for 30 epochs; outputs a sample grid per label and a loss curve. Also
includes a small `generate_from_text()` helper that maps a free-text
prompt (e.g. "draw me a circle") to the right label.

---

## Task 3 - Explore a Public Image-Caption Dataset

**What it does:** Loads and examines a public dataset (Oxford-102 Flowers
or COCO Captions), computing statistics — number of classes, caption
length distribution, image resolution — and visualizing sample images
alongside their text descriptions.

**Approach:** `torchvision.datasets.Flowers102` / `CocoCaptions` as the
two real dataset options (switchable via a `DATASET` config flag), with a
built-in offline demo dataset (reusing Task 2's shape generator, with
templated captions) so the analysis code can be verified without internet
access, then pointed at the real dataset when running with internet.

**Files:** `Task3_Dataset_Exploration/task3_dataset_exploration.py` /
`.ipynb`. Outputs a sample image+caption grid, a caption-length histogram,
a resolution scatter plot, and a JSON stats summary.

---

## Task 4 - Text Preprocessing with Hugging Face Transformers

**What it does:** Preprocesses text captions into tokenized and encoded
representations (embeddings) — the same kind of input a text-to-image
model needs to condition its generation on the text.

**Approach:** Hugging Face's CLIP tokenizer + text encoder
(`openai/clip-vit-base-patch32`) — the same text encoder family Stable
Diffusion itself uses. Falls back to a deterministic hashing-based
embedding if the model can't be downloaded (no internet), so the
`TextEncoder.encode()` interface used by Tasks 5/6 always works.

**Files:** `Task4_Text_Preprocessing/task4_text_preprocessing.py` /
`.ipynb`. Outputs an example tokenization (token ids/tokens for sample
captions), the resulting embeddings, and a cosine-similarity heatmap
between caption embeddings as a sanity check.

---

## Task 5 - Attention-Augmented GAN

**What it does:** Improves the Task 2 CGAN with attention so the model
can focus on relevant parts of the input text and image, producing
better-conditioned images.

**Approach:** Adds two attention mechanisms on top of Task 2's
generator/discriminator:
- **Self-attention** (SAGAN-style) — lets different spatial positions in
  a feature map attend to each other, for more globally coherent shapes.
- **Cross-attention** — lets each spatial position attend directly to the
  Task 4 text embedding, the same mechanism used in Stable Diffusion's
  U-Net to condition on a text prompt.

**Files:** `Task5_Attention_GAN/task5_attention_gan.py` / `.ipynb`.
Trained the same way as Task 2 but conditioned on real caption embeddings
instead of just a label index; outputs a sample grid and training curve.

---

## Task 6 - Full Text-to-Image Generation Pipeline

**What it does:** Combines everything above into one end-to-end pipeline:
text preprocessing (Task 4) → embedding → attention-based GAN generation
(Task 5) → image, wrapped behind a single `generate(prompt)` call —
mirroring how `StableDiffusionGenerator` works in the training project.

**Approach:** A `TextToImagePipeline` class that loads the Task 4 text
encoder and Task 5 generator, resolves a free-text prompt to a shape
category, encodes its caption, and generates the corresponding image. Also
includes an optional Gradio UI (same library as the training project) for
an interactive demo.

**Files:** `Task6_Full_Pipeline/task6_full_pipeline.py` / `.ipynb`. Run
with `--ui` to launch the Gradio demo. Outputs a sample gallery generated
from four different text prompts.

---

## Datasets

The `datasets/` folder contains the actual data files used:
- `custom_domain_dataset_task1/` — 8 image+caption pairs for Task 1's LoRA fine-tuning demo
- `shapes_dataset_task2/` — 400 labeled shape images used by Tasks 2 and 5
- `shapes_captioned_dataset_task3/` — 100 image+caption pairs used by Task 3's exploration

## Setup

```bash
pip install torch torchvision matplotlib pillow numpy
pip install transformers diffusers accelerate gradio   # for Tasks 1, 4, 5, 6 with full models
```

## Running a Task

```bash
cd Task2_CGAN_Shapes
python task2_cgan_shapes.py
```

or open the matching `.ipynb` in Jupyter/Google Colab and run all cells.

**Note:** Tasks 1, 3, and 4 use real pre-trained models / public datasets
when internet access is available (e.g. in Google Colab) and fall back to
lightweight offline versions otherwise — no code changes needed either way.
