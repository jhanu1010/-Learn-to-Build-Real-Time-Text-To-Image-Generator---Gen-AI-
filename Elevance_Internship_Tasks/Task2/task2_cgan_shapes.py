"""
INTERNSHIP TASK 2
Conditional GAN (CGAN) that uses textual labels/categories to produce basic
visuals. Given a label such as "square", "circle", or "triangle", the
generator produces an image of the corresponding shape.

This extends the training-project (Stable Diffusion generator app) by
introducing *conditional* generation with a lightweight GAN built from
scratch in PyTorch, which demonstrates the core idea (label -> embedding ->
conditioned generator) that underlies text-to-image models like the one in
the training project.

Run:
    python task2_cgan_shapes.py
Outputs:
    outputs/task2_samples.png        (grid of generated shapes per label)
    outputs/task2_training_curve.png (loss curve)
    outputs/task2_generator.pt       (trained generator weights)
"""

# CELL 1: Imports & config
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMG_SIZE = 32
LABELS = ["circle", "square", "triangle", "star"]
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)
LATENT_DIM = 64
EMBED_DIM = 16
BATCH_SIZE = 64
EPOCHS = 30
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# CELL 2: Synthetic labeled-shape dataset
# In place of a hand-labeled image bank, we procedurally draw each shape.
# This keeps Task 2 self-contained and reproducible, while the CGAN training
# logic below is identical to what you'd use with a real labeled dataset.
class ShapesDataset(Dataset):
    def __init__(self, n_per_class: int = 500, img_size: int = IMG_SIZE):
        self.img_size = img_size
        self.samples = []
        for label in LABELS:
            for _ in range(n_per_class):
                self.samples.append(label)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _draw(self, label: str) -> Image.Image:
        size = self.img_size
        img = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(img)
        pad = random.randint(3, 6)
        box = [pad, pad, size - pad, size - pad]
        if label == "circle":
            draw.ellipse(box, fill=255)
        elif label == "square":
            draw.rectangle(box, fill=255)
        elif label == "triangle":
            x0, y0, x1, y1 = box
            draw.polygon([(size / 2, y0), (x0, y1), (x1, y1)], fill=255)
        elif label == "star":
            cx, cy, r = size / 2, size / 2, size / 2 - pad
            pts = []
            for i in range(10):
                ang = np.pi / 2 + i * np.pi / 5
                rad = r if i % 2 == 0 else r * 0.45
                pts.append((cx + rad * np.cos(ang), cy - rad * np.sin(ang)))
            draw.polygon(pts, fill=255)
        return img

    def __getitem__(self, idx):
        label = self.samples[idx]
        img = self._draw(label)
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        return tensor, LABEL2IDX[label]


# CELL 3: Generator (conditioned on label embedding)
class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.init_size = img_size // 4
        self.fc = nn.Linear(latent_dim + embed_dim, 128 * self.init_size ** 2)
        self.net = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_emb = self.label_embed(labels)
        x = torch.cat([noise, label_emb], dim=1)
        x = self.fc(x).view(x.size(0), 128, self.init_size, self.init_size)
        return self.net(x)


# CELL 4: Discriminator (conditioned on label embedding)
class Discriminator(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size

        def block(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c, 0.8))
            return layers

        self.net = nn.Sequential(
            *block(2, 32, bn=False),
            *block(32, 64),
            *block(64, 128),
        )
        ds_size = img_size // 2 ** 3
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        label_map = self.label_embed(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.adv_layer(x)


# CELL 5: Training loop
def train_cgan():
    dataset = ShapesDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    g_losses, d_losses = [], []
    for epoch in range(EPOCHS):
        epoch_g, epoch_d = 0.0, 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            bs = imgs.size(0)
            valid = torch.ones(bs, 1, device=DEVICE)
            fake = torch.zeros(bs, 1, device=DEVICE)

            # Train Generator
            opt_G.zero_grad()
            noise = torch.randn(bs, LATENT_DIM, device=DEVICE)
            gen_labels = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            gen_imgs = G(noise, gen_labels)
            g_loss = criterion(D(gen_imgs, gen_labels), valid)
            g_loss.backward()
            opt_G.step()

            # Train Discriminator
            opt_D.zero_grad()
            real_loss = criterion(D(imgs, labels), valid)
            fake_loss = criterion(D(gen_imgs.detach(), gen_labels), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_D.step()

            epoch_g += g_loss.item()
            epoch_d += d_loss.item()

        g_losses.append(epoch_g / len(loader))
        d_losses.append(epoch_d / len(loader))
        print(f"Epoch {epoch+1}/{EPOCHS} | G loss: {g_losses[-1]:.4f} | D loss: {d_losses[-1]:.4f}")

    torch.save(G.state_dict(), os.path.join(OUT_DIR, "task2_generator.pt"))

    plt.figure()
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Task 2: CGAN Training Curve")
    plt.savefig(os.path.join(OUT_DIR, "task2_training_curve.png"))
    plt.close()

    return G


# CELL 6: Generate & visualize a sample per label
def sample_and_plot(G, n_per_label=4):
    G.eval()
    fig, axes = plt.subplots(NUM_CLASSES, n_per_label, figsize=(n_per_label * 2, NUM_CLASSES * 2))
    with torch.no_grad():
        for row, label in enumerate(LABELS):
            noise = torch.randn(n_per_label, LATENT_DIM, device=DEVICE)
            labels = torch.full((n_per_label,), LABEL2IDX[label], dtype=torch.long, device=DEVICE)
            imgs = G(noise, labels).cpu().numpy()
            for col in range(n_per_label):
                ax = axes[row, col]
                ax.imshow((imgs[col, 0] + 1) / 2, cmap="gray")
                ax.axis("off")
                if col == 0:
                    ax.set_ylabel(label)
        for row, label in enumerate(LABELS):
            axes[row, 0].set_title(label, loc="left")
    plt.suptitle("Task 2: CGAN samples conditioned on text label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task2_samples.png"))
    plt.close()
    print(f"Saved sample grid to {OUT_DIR}/task2_samples.png")


# CELL 7: Text-label -> shape inference helper (mirrors "prompt in, image out")
def generate_from_text(G, text: str):
    """Very small text interface: looks up the shape keyword in the prompt."""
    text = text.lower()
    match = next((label for label in LABELS if label in text), None)
    if match is None:
        raise ValueError(f"No known shape keyword found in '{text}'. Known labels: {LABELS}")
    noise = torch.randn(1, LATENT_DIM, device=DEVICE)
    label_tensor = torch.tensor([LABEL2IDX[match]], device=DEVICE)
    with torch.no_grad():
        img = G(noise, label_tensor).cpu().numpy()[0, 0]
    return (img + 1) / 2, match


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    G = train_cgan()
    sample_and_plot(G)
    demo_img, demo_label = generate_from_text(G, "please draw a circle for me")
    print(f"Text-to-shape demo produced a '{demo_label}' image with shape {demo_img.shape}")
