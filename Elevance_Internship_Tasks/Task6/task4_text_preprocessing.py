"""
INTERNSHIP TASK 4
Preprocess text descriptions into tokenized + encoded representations using
Hugging Face Transformers, producing the text embeddings that a
text-to-image model (Task 5/6) uses as conditioning input.

Primary path: CLIP's text tower (openai/clip-vit-base-patch32), the same
family of text encoder Stable Diffusion itself uses (matching the training
project). Falls back to a deterministic bag-of-words hashing embedding if
model weights cannot be downloaded (e.g. no internet in this environment),
so the rest of the pipeline (Task 5/6) always has a well-defined
`TextEncoder.encode(list[str]) -> np.ndarray[N, D]` interface to build on.

Run:
    python task4_text_preprocessing.py
Outputs:
    outputs/task4_tokenization_example.json
    outputs/task4_embeddings.npy
    outputs/task4_embedding_similarity.png
"""

# CELL 1: Imports & config
import os
import json
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBED_DIM_FALLBACK = 128

SAMPLE_CAPTIONS = [
    "a small white circle on a dark background",
    "a large yellow square with sharp corners",
    "a red triangle pointing upward",
    "a bright blue star with five points",
    "this pink flower has overlapping petals and yellow florets",
]


# CELL 2: Text encoder wrapper (HF Transformers, with offline fallback)
class TextEncoder:
    """Tokenizes + encodes text into fixed-size embeddings.

    encode() always returns a numpy array of shape (N, D) so downstream
    GAN/diffusion conditioning code (Task 5/6) doesn't need to know which
    backend produced the embedding.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.backend = None
        self.tokenizer = None
        self.model = None
        self.embed_dim = None
        self._try_load_hf_model()

    def _try_load_hf_model(self):
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
            import torch

            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            self.model = CLIPTextModel.from_pretrained(self.model_name)
            self.model.eval()
            self.embed_dim = self.model.config.hidden_size
            self.backend = "huggingface_clip"
            print(f"Loaded {self.model_name} (embed_dim={self.embed_dim})")
        except Exception as e:
            print(f"Could not load '{self.model_name}' from Hugging Face "
                  f"({e}). Falling back to a lightweight offline hashing "
                  f"encoder so the pipeline still runs end-to-end. "
                  f"In an environment with internet access (e.g. Colab), "
                  f"this will automatically use the real CLIP text encoder.")
            self.backend = "offline_hash"
            self.embed_dim = EMBED_DIM_FALLBACK

    def tokenize(self, texts):
        """Returns a serializable dict of token ids/attention masks (HF
        backend) or of simple whitespace tokens (offline backend), useful
        for inspecting how a caption is represented before encoding."""
        if self.backend == "huggingface_clip":
            enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].tolist(),
                "attention_mask": enc["attention_mask"].tolist(),
                "tokens": [self.tokenizer.convert_ids_to_tokens(ids) for ids in enc["input_ids"]],
            }
        else:
            tokenized = [t.lower().replace(",", "").split() for t in texts]
            return {"tokens": tokenized}

    def encode(self, texts):
        """Returns embeddings as a numpy array of shape (len(texts), D)."""
        if self.backend == "huggingface_clip":
            import torch

            enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**enc)
            # Use the pooled [EOS] representation, standard for CLIP text towers.
            pooled = out.pooler_output
            return pooled.cpu().numpy()
        else:
            return np.stack([self._hash_embed(t) for t in texts])

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-words hashing embedding (offline fallback).
        Not a substitute for a learned encoder, but keeps the interface and
        downstream shapes identical so Task 5/6 code doesn't need to branch."""
        vec = np.zeros(self.embed_dim, dtype=np.float32)
        for word in text.lower().replace(",", "").split():
            h = hash(word) % self.embed_dim
            vec[h] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# CELL 3: Similarity check - sanity-test that encoded captions cluster sensibly
def plot_similarity_matrix(embeddings, labels):
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim = norm @ norm.T
    plt.figure(figsize=(6, 5))
    plt.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title("Task 4: caption embedding similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "task4_embedding_similarity.png"))
    plt.close()


if __name__ == "__main__":
    encoder = TextEncoder()
    print(f"Backend in use: {encoder.backend}")

    tokenized = encoder.tokenize(SAMPLE_CAPTIONS)
    with open(os.path.join(OUT_DIR, "task4_tokenization_example.json"), "w") as f:
        json.dump({"captions": SAMPLE_CAPTIONS, "tokenized": tokenized}, f, indent=2)

    embeddings = encoder.encode(SAMPLE_CAPTIONS)
    print(f"Embeddings shape: {embeddings.shape}")
    np.save(os.path.join(OUT_DIR, "task4_embeddings.npy"), embeddings)

    plot_similarity_matrix(embeddings, SAMPLE_CAPTIONS)
    print(f"Saved tokenization example, embeddings, and similarity plot to {OUT_DIR}/")
