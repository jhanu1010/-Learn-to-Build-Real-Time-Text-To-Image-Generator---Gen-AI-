"""
INTERNSHIP TASK 1
Fine-tune the pre-trained text-to-image model from the training project
(runwayml/stable-diffusion-v1-5, loaded by StableDiffusionGenerator in
colab_cell_by_cell.py) on a small custom domain-specific dataset, so it
learns to produce domain-specific visuals (e.g. a particular art style, or
medical-imaging-style outputs) associated with a special token.

Approach: LoRA (Low-Rank Adaptation) fine-tuning of the UNet's
cross-attention layers. This is the standard, GPU-memory-friendly way to
adapt Stable Diffusion to a new visual domain (vs. full fine-tuning, which
needs far more VRAM and data) and is the technique used in production
tools like DreamBooth-LoRA / civitai-style style-LoRAs.

⚠ This script needs internet access (to download the base SD1.5 weights
from Hugging Face) and a GPU (a T4 in Colab is enough for ~500-1000 steps).
It is written to be run there; it will raise a clear error if the base
model cannot be downloaded in a restricted, offline environment.

Custom dataset format expected (edit CUSTOM_DATA_DIR):
    custom_dataset/
        image_001.png
        image_001.txt   <- caption, e.g. "a hospital chest x-ray, TOKEN style"
        image_002.png
        image_002.txt
        ...
A tiny example generator (make_placeholder_dataset) is included so the
folder structure and captioning convention are demonstrated even before you
drop in your own domain images (art / medical imagery / etc).

Run:
    python task1_finetune_diffusion.py
Outputs:
    outputs/task1_lora_weights/         (trained LoRA adapter, loadable
                                          back into StableDiffusionGenerator)
    outputs/task1_before_after.png      (base model vs fine-tuned model,
                                          same prompt/seed)
"""

# CELL 1: Imports & config
import os
import glob
import random

SPECIAL_TOKEN = "sks-domainart"   # unique token bound to the custom domain
MODEL_ID = "runwayml/stable-diffusion-v1-5"
CUSTOM_DATA_DIR = "data/custom_domain_dataset"
OUT_DIR = "outputs"
LORA_OUT_DIR = os.path.join(OUT_DIR, "task1_lora_weights")
RESOLUTION = 512
TRAIN_STEPS = 800
LR = 1e-4
LORA_RANK = 4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CUSTOM_DATA_DIR, exist_ok=True)


# CELL 2: Placeholder dataset generator (shows the expected folder format)
def make_placeholder_dataset(n=8):
    """Creates a tiny procedurally-drawn 'domain' dataset (stylised
    concentric-shape 'artwork') purely so the folder layout and captioning
    convention can be inspected without needing real domain images yet.
    Replace the contents of CUSTOM_DATA_DIR with your real artwork /
    medical-imaging dataset (10-30 images is typically enough for LoRA)."""
    from PIL import Image, ImageDraw
    import numpy as np

    random.seed(SEED)
    for i in range(n):
        img = Image.new("RGB", (256, 256), color=(10, 10, 30))
        draw = ImageDraw.Draw(img)
        cx, cy = 128, 128
        for r in range(100, 10, -15):
            colour = tuple(int(c) for c in np.random.randint(80, 255, size=3))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=4)
        path_img = os.path.join(CUSTOM_DATA_DIR, f"image_{i:03d}.png")
        path_txt = os.path.join(CUSTOM_DATA_DIR, f"image_{i:03d}.txt")
        img.save(path_img)
        with open(path_txt, "w") as f:
            f.write(f"an abstract concentric circle artwork, {SPECIAL_TOKEN} style")
    print(f"Wrote {n} placeholder image/caption pairs to {CUSTOM_DATA_DIR}/")


# CELL 3: Dataset class for the fine-tuning loop
def build_dataset(tokenizer, resolution=RESOLUTION):
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    class DomainDataset(Dataset):
        def __init__(self, data_dir):
            self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")) +
                                       glob.glob(os.path.join(data_dir, "*.jpg")))
            self.tfm = transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            caption = open(txt_path).read().strip() if os.path.exists(txt_path) else SPECIAL_TOKEN
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.tfm(image)
            input_ids = tokenizer(caption, padding="max_length", truncation=True,
                                   max_length=tokenizer.model_max_length,
                                   return_tensors="pt").input_ids[0]
            return {"pixel_values": pixel_values, "input_ids": input_ids}

    return DomainDataset(CUSTOM_DATA_DIR)


# CELL 4: LoRA fine-tuning loop
def finetune_lora():
    """
    Attaches LoRA adapters to the UNet cross-attention layers of the
    pre-trained Stable Diffusion pipeline (the same pipeline class,
    StableDiffusionGenerator, used in the training project) and trains
    them on the custom domain dataset. Requires internet + GPU.
    """
    import torch
    import torch.nn.functional as F
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from diffusers.loaders import AttnProcsLayers
    from diffusers.models.attention_processor import LoRAAttnProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected. LoRA fine-tuning of Stable Diffusion "
              "on CPU is impractically slow; run this on a GPU runtime "
              "(e.g. Colab T4) for a real training run. Continuing anyway "
              "with a tiny number of steps purely to validate the code path.")

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, safety_checker=None)
    pipe.to(device)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Attach LoRA adapters to every cross-attention processor in the UNet.
    lora_attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        cross_attention_dim = (None if name.endswith("attn1.processor")
                                else pipe.unet.config.cross_attention_dim)
        if name.startswith("mid_block"):
            hidden_size = pipe.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipe.unet.config.block_out_channels))[block_id]
        else:  # down_blocks
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipe.unet.config.block_out_channels[block_id]
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=LORA_RANK
        )
    pipe.unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(pipe.unet.attn_processors)

    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=LR)
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    dataset = build_dataset(pipe.tokenizer)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found in {CUSTOM_DATA_DIR}. Run make_placeholder_dataset() "
            f"first, or copy your own domain images + .txt captions there."
        )
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    global_step = 0
    pipe.unet.train()
    while global_step < TRAIN_STEPS:
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, dtype=pipe.unet.dtype)
            input_ids = batch["input_ids"].to(device)

            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                       (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = pipe.text_encoder(input_ids)[0]
            model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = F.mse_loss(model_pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 50 == 0:
                print(f"step {global_step}/{TRAIN_STEPS} | loss {loss.item():.4f}")
            if global_step >= TRAIN_STEPS:
                break

    os.makedirs(LORA_OUT_DIR, exist_ok=True)
    pipe.unet.save_attn_procs(LORA_OUT_DIR)
    print(f"Saved LoRA adapter weights to {LORA_OUT_DIR}/")
    return pipe


# CELL 5: Before/after comparison (extends StableDiffusionGenerator from the
# training project's colab_cell_by_cell.py)
def compare_before_after(prompt=f"a serene mountain landscape, {SPECIAL_TOKEN} style"):
    import torch
    from diffusers import StableDiffusionPipeline
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(SEED)

    base_pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, safety_checker=None).to(device)
    base_image = base_pipe(prompt, num_inference_steps=25, generator=generator).images[0]

    base_pipe.unet.load_attn_procs(LORA_OUT_DIR)
    generator = torch.Generator(device=device).manual_seed(SEED)
    finetuned_image = base_pipe(prompt, num_inference_steps=25, generator=generator).images[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(base_image); axes[0].set_title("Base SD1.5"); axes[0].axis("off")
    axes[1].imshow(finetuned_image); axes[1].set_title("Fine-tuned (LoRA)"); axes[1].axis("off")
    plt.suptitle(f'Prompt: "{prompt}"')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task1_before_after.png"))
    print(f"Saved before/after comparison to {OUT_DIR}/task1_before_after.png")


if __name__ == "__main__":
    if not glob.glob(os.path.join(CUSTOM_DATA_DIR, "*.png")):
        make_placeholder_dataset()
    try:
        finetune_lora()
        compare_before_after()
    except Exception as e:
        print(f"Fine-tuning requires internet access to download "
              f"'{MODEL_ID}' from Hugging Face and (ideally) a GPU. "
              f"Run this script in Google Colab / a machine with internet "
              f"and a GPU runtime. Original error: {e}")
