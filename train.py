# train.py
import math
import os
import pickle
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
import numpy as np
import torch
from model import GPT, GPTConfig

PRESETS = {
    "tiny": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=64, max_iters=3000),
    "small": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=32, max_iters=5000),
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=256, batch_size=32, max_iters=8000),
    "large": dict(n_layer=12, n_head=12, n_embd=768, block_size=512, batch_size=12, max_iters=10000),
}

P = PRESETS[sys.argv[1] if len(sys.argv) > 1 else "small"]
batch_size, block_size, max_iters = P["batch_size"], P["block_size"], P["max_iters"]
eval_interval, eval_iters = 250, 40
learning_rate, min_lr, warmup_iters = 6e-4, 6e-5, 200
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
torch.manual_seed(1337)
print("device:", device)

# ---------- data ----------
meta = pickle.load(open("data/meta.pkl", "rb"))
data = {s: np.memmap(f"data/{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
assert len(data["val"]) > block_size + 1, "Not enough data: add more text or use the tiny preset"


def get_batch(split):
    d = data[split]
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(d[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------- model ----------
cfg = GPTConfig(
    vocab_size=meta["vocab_size"],
    block_size=block_size,
    n_layer=P["n_layer"],
    n_head=P["n_head"],
    n_embd=P["n_embd"],
    dropout=dropout,
)
model = GPT(cfg).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

decay = [p for p in model.parameters() if p.dim() >= 2]  # weight matrices
no_decay = [p for p in model.parameters() if p.dim() < 2]  # biases and LayerNorm
opt = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=learning_rate,
    betas=(0.9, 0.95),
)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters  # linear warmup
    ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)  # cosine decay


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with amp:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------- training loop ----------
os.makedirs("out", exist_ok=True)
best_val, t0 = float("inf"), time.time()

for it in range(max_iters):
    for g in opt.param_groups:
        g["lr"] = get_lr(it)

    if it % eval_interval == 0 or it == max_iters - 1:
        L = estimate_loss()
        print(f"step {it:5d} | train {L['train']:.3f} | val {L['val']:.3f} | {time.time() - t0:.0f}s")
        if L["val"] < best_val:  # keep only the best model
            best_val = L["val"]
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, "out/ckpt.pt")

    x, y = get_batch("train")
    with amp:
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

print("done. best val loss:", round(best_val, 3))