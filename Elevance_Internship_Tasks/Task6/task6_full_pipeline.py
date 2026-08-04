"""
INTERNSHIP TASK 6
Full text-to-image generation pipeline that integrates every earlier task
into one class, mirroring the structure of the training project's
StableDiffusionGenerator + StableDiffusionUI (colab_cell_by_cell.py):

    caption dataset (Task 3)
        -> text tokenization/encoding (Task 4)
            -> attention-conditioned GAN generation (Task 5, built on Task 2)
                -> Gradio demo UI (same library as the training project)

This simulates, at small scale, the real text-to-image workflow the
training project's Stable Diffusion app implements at full scale.

Run:
    python task6_full_pipeline.py            # trains + saves demo images
    python task6_full_pipeline.py --ui        # also launches a Gradio UI
Outputs:
    outputs/task6_pipeline_demo.png
    outputs/task6_pipeline_generator.pt
"""

# CELL 1: Imports & config
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from task2_cgan_shapes import LABELS, LABEL2IDX, LATENT_DIM as SHAPE_LATENT_DIM, NUM_CLASSES
from task4_text_preprocessing import TextEncoder
from task5_attention_gan import (
    AttnGenerator, train_attention_gan, CAPTION_TEMPLATES, LATENT_DIM,
)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CELL 2: End-to-end pipeline class (mirrors StableDiffusionGenerator's role)
class TextToImagePipeline:
    """Ties together: text encoder (Task 4) -> attention-conditioned
    generator (Task 5), exposing a single generate(prompt) -> PIL.Image
    call, the same interface shape as StableDiffusionGenerator.generate()
    in the training project."""

    def __init__(self, generator: AttnGenerator, text_encoder: TextEncoder):
        self.generator = generator.to(DEVICE).eval()
        self.text_encoder = text_encoder

    @classmethod
    def train_from_scratch(cls):
        G, encoder, _ = train_attention_gan()
        return cls(G, encoder)

    @classmethod
    def load(cls, weights_path=os.path.join(OUT_DIR, "task5_generator.pt")):
        encoder = TextEncoder()
        G = AttnGenerator(LATENT_DIM, encoder.embed_dim, NUM_CLASSES)
        G.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        return cls(G, encoder)

    def _resolve_label_and_caption(self, prompt: str):
        prompt_lower = prompt.lower()
        label = next((l for l in LABELS if l in prompt_lower), None)
        if label is None:
            raise ValueError(
                f"Prompt must mention one of the known shape categories {LABELS}. "
                f"(Task 6 conditions on the same label space as Tasks 2/3/5; swap "
                f"in a real captioned dataset, per Task 3, to lift this restriction.)"
            )
        return label, CAPTION_TEMPLATES[label]

    def generate(self, prompt: str, seed: int | None = None):
        label, canonical_caption = self._resolve_label_and_caption(prompt)
        text_emb = torch.tensor(
            self.text_encoder.encode([canonical_caption]), dtype=torch.float32, device=DEVICE
        )
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn(1, LATENT_DIM, device=DEVICE)
        label_t = torch.tensor([LABEL2IDX[label]], device=DEVICE)
        with torch.no_grad():
            img = self.generator(noise, label_t, text_emb)[0, 0].cpu().numpy()
        img = ((img + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image
        return Image.fromarray(img, mode="L"), label


# CELL 3: Demo run (train + generate a small gallery of prompts)
DEMO_PROMPTS = [
    "generate a circle",
    "draw me a square",
    "I need a triangle image",
    "make a star please",
]


def run_demo(pipeline: TextToImagePipeline):
    fig, axes = plt.subplots(1, len(DEMO_PROMPTS), figsize=(len(DEMO_PROMPTS) * 3, 3))
    for ax, prompt in zip(axes, DEMO_PROMPTS):
        img, label = pipeline.generate(prompt, seed=0)
        ax.imshow(img, cmap="gray")
        ax.set_title(f'"{prompt}"\n-> {label}', fontsize=9)
        ax.axis("off")
    plt.suptitle("Task 6: full text -> tokenize/encode -> attention-GAN -> image pipeline")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task6_pipeline_demo.png"))
    plt.close()
    print(f"Saved end-to-end pipeline demo to {OUT_DIR}/task6_pipeline_demo.png")


# CELL 4: Optional Gradio UI (same library as the training project's app)
def launch_ui(pipeline: TextToImagePipeline):
    import gradio as gr

    def infer(prompt, seed):
        img, label = pipeline.generate(prompt, seed=int(seed))
        return img, f"Matched category: {label}"

    with gr.Blocks(title="Task 6: Text-to-Image Pipeline") as demo:
        gr.Markdown("## Internship Task 6 - Full Text-to-Image Pipeline\n"
                    f"Try a prompt containing one of: {', '.join(LABELS)}")
        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", value="generate a circle")
            seed_box = gr.Number(label="Seed", value=0)
        run_btn = gr.Button("Generate")
        output_img = gr.Image(label="Generated image")
        output_label = gr.Textbox(label="Resolved label")
        run_btn.click(infer, inputs=[prompt_box, seed_box], outputs=[output_img, output_label])
    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    weights_path = os.path.join(OUT_DIR, "task5_generator.pt")
    if os.path.exists(weights_path):
        print("Loading pre-trained generator from Task 5...")
        pipeline = TextToImagePipeline.load(weights_path)
    else:
        print("No saved Task 5 weights found - training the attention-GAN now...")
        pipeline = TextToImagePipeline.train_from_scratch()

    run_demo(pipeline)

    if "--ui" in sys.argv:
        launch_ui(pipeline)
