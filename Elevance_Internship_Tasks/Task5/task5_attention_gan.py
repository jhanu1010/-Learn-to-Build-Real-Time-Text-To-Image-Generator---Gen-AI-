"""
INTERNSHIP TASK 5
Improve the Task 2 CGAN with attention: a self-attention block in the
generator/discriminator feature maps (SAGAN-style, "which pixels relate to
which other pixels") plus a cross-attention block that lets each spatial
location attend directly to the Task-4 text embedding ("which words does
this region of the image correspond to"). This is the same attention
principle used inside Stable Diffusion's U-Net (from the training project),
scaled down to a small GAN so it can be trained quickly for the internship.

Run:
    python task5_attention_gan.py
Outputs:
    outputs/task5_samples.png
    outputs/task5_training_curve.png
    outputs/task5_generator.pt
"""

# CELL 1: Imports & config
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from task2_cgan_shapes import ShapesDataset, LABELS, LABEL2IDX, IMG_SIZE, NUM_CLASSES
from task4_text_preprocessing import TextEncoder

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

LATENT_DIM = 64
TEXT_DIM = 128           # matches TextEncoder's fallback embed_dim; auto-detected below anyway
BATCH_SIZE = 64
EPOCHS = 30
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

CAPTION_TEMPLATES = {
    "circle": "a smooth round circle shape",
    "square": "a square shape with four equal sides",
    "triangle": "a triangle shape with three sharp corners",
    "star": "a pointed five sided star shape",
}


# CELL 2: Self-attention block (SAGAN-style, over spatial feature maps)
class SelfAttention2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)   # (b, hw, c//8)
        k = self.key(x).view(b, -1, h * w)                      # (b, c//8, hw)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)            # (b, hw, hw)
        v = self.value(x).view(b, c, h * w)                      # (b, c, hw)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


# CELL 3: Cross-attention block (spatial features attend to text embedding)
class CrossAttention2d(nn.Module):
    """Each spatial location (query) attends over the text-embedding tokens
    (key/value), analogous to the cross-attention layers in Stable
    Diffusion's U-Net that condition on CLIP text embeddings."""

    def __init__(self, channels, text_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        assert channels % heads == 0, "channels must be divisible by heads"
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x, text_emb):
        # x: (b, c, h, w) ; text_emb: (b, text_dim) treated as a single
        # "token" per caption (extendable to per-word tokens).
        b, c, h, w = x.shape
        text_tokens = text_emb.unsqueeze(1)  # (b, 1, text_dim)

        q = self.to_q(x).view(b, self.heads, self.head_dim, h * w)          # (b, heads, hd, hw)
        k = self.to_k(text_tokens).view(b, 1, self.heads, self.head_dim).permute(0, 2, 3, 1)  # (b, heads, hd, 1)
        v = self.to_v(text_tokens).view(b, 1, self.heads, self.head_dim).permute(0, 2, 1, 3)  # (b, heads, 1, hd)

        attn = torch.einsum("bhdn,bhdt->bhnt", q, k) / (self.head_dim ** 0.5)  # (b, heads, hw, 1)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnt,bhtd->bhnd", attn, v)  # (b, heads, hw, hd)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(out)


# CELL 4: Attention-augmented Generator
class AttnGenerator(nn.Module):
    def __init__(self, latent_dim, text_dim, num_classes, img_size=IMG_SIZE, label_embed_dim=16):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, label_embed_dim)
        self.init_size = img_size // 4
        self.fc = nn.Linear(latent_dim + label_embed_dim, 128 * self.init_size ** 2)

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(128), nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
        )
        self.cross_attn = CrossAttention2d(128, text_dim, heads=4)
        self.self_attn = SelfAttention2d(128)
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, padding=1), nn.Tanh(),
        )

    def forward(self, noise, labels, text_emb):
        label_e = self.label_embed(labels)
        x = torch.cat([noise, label_e], dim=1)
        x = self.fc(x).view(x.size(0), 128, self.init_size, self.init_size)
        x = self.block1(x)
        x = self.cross_attn(x, text_emb)   # attend to the caption
        x = self.self_attn(x)              # attend within the image itself
        return self.block2(x)


# CELL 5: Attention-augmented Discriminator
class AttnDiscriminator(nn.Module):
    def __init__(self, text_dim, num_classes, img_size=IMG_SIZE, label_embed_dim=16):
        super().__init__()
        self.img_size = img_size
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)

        self.block1 = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(64, 0.8),
        )
        self.self_attn = SelfAttention2d(64)
        self.cross_attn = CrossAttention2d(64, text_dim, heads=4)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(128, 0.8),
        )
        ds_size = img_size // 2 ** 3
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels, text_emb):
        label_map = self.label_embed(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        x = self.block1(x)
        x = self.self_attn(x)
        x = self.cross_attn(x, text_emb)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.adv_layer(x)


# CELL 6: Training loop
def train_attention_gan():
    encoder = TextEncoder()
    text_dim = encoder.embed_dim

    dataset = ShapesDataset(n_per_class=500)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Pre-compute one caption embedding per class (cached lookup table).
    class_captions = [CAPTION_TEMPLATES[label] for label in LABELS]
    class_text_emb = torch.tensor(encoder.encode(class_captions), dtype=torch.float32, device=DEVICE)

    G = AttnGenerator(LATENT_DIM, text_dim, NUM_CLASSES).to(DEVICE)
    D = AttnDiscriminator(text_dim, NUM_CLASSES).to(DEVICE)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    g_losses, d_losses = [], []
    for epoch in range(EPOCHS):
        eg, ed = 0.0, 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            bs = imgs.size(0)
            valid = torch.ones(bs, 1, device=DEVICE)
            fake = torch.zeros(bs, 1, device=DEVICE)
            text_emb = class_text_emb[labels]

            opt_G.zero_grad()
            noise = torch.randn(bs, LATENT_DIM, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            gen_text_emb = class_text_emb[gen_labels]
            gen_imgs = G(noise, gen_labels, gen_text_emb)
            g_loss = criterion(D(gen_imgs, gen_labels, gen_text_emb), valid)
            g_loss.backward()
            opt_G.step()

            opt_D.zero_grad()
            real_loss = criterion(D(imgs, labels, text_emb), valid)
            fake_loss = criterion(D(gen_imgs.detach(), gen_labels, gen_text_emb), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_D.step()

            eg += g_loss.item(); ed += d_loss.item()

        g_losses.append(eg / len(loader)); d_losses.append(ed / len(loader))
        print(f"Epoch {epoch+1}/{EPOCHS} | G: {g_losses[-1]:.4f} | D: {d_losses[-1]:.4f}")

    torch.save(G.state_dict(), os.path.join(OUT_DIR, "task5_generator.pt"))
    plt.figure()
    plt.plot(g_losses, label="Generator"); plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Task 5: Attention-GAN training curve")
    plt.savefig(os.path.join(OUT_DIR, "task5_training_curve.png")); plt.close()

    return G, encoder, class_text_emb


# CELL 7: Sample visualization
def sample_and_plot(G, class_text_emb, n_per_label=4):
    G.eval()
    fig, axes = plt.subplots(NUM_CLASSES, n_per_label, figsize=(n_per_label * 2, NUM_CLASSES * 2))
    with torch.no_grad():
        for row, label in enumerate(LABELS):
            noise = torch.randn(n_per_label, LATENT_DIM, device=DEVICE)
            labels_t = torch.full((n_per_label,), LABEL2IDX[label], dtype=torch.long, device=DEVICE)
            text_emb = class_text_emb[labels_t]
            imgs = G(noise, labels_t, text_emb).cpu().numpy()
            for col in range(n_per_label):
                ax = axes[row, col]
                ax.imshow((imgs[col, 0] + 1) / 2, cmap="gray")
                ax.axis("off")
            axes[row, 0].set_title(label, loc="left", fontsize=9)
    plt.suptitle("Task 5: Attention-GAN samples conditioned on text embedding")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task5_samples.png")); plt.close()
    print(f"Saved sample grid to {OUT_DIR}/task5_samples.png")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    G, encoder, class_text_emb = train_attention_gan()
    sample_and_plot(G, class_text_emb)
