# sft.py
import json
import os
import pickle
import random
import sys
from contextlib import nullcontext
import torch
import torch.nn as nn
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
use_lora = "no-lora" not in sys.argv  # default to True for stable fine-tuning

# Determine data path (fallback to data/sft.jsonl if not provided or if argument is 'lora')
data_path = "data/sft.jsonl"
for arg in sys.argv[1:]:
    if arg not in ("lora", "no-lora") and not arg.startswith("-"):
        data_path = arg
        break

tok = pickle.load(open("data/tok.pkl", "rb"))


# ---------- LoRA ----------
class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False  # freeze original weights
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zeros: starts as the original model
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model, r=8):
    for p in model.parameters():
        p.requires_grad = False
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r)
        blk.attn.proj = LoRALinear(blk.attn.proj, r)


def merge_lora(model):  # fold adapters back into base weights
    for blk in model.blocks:
        for name in ("qkv", "proj"):
            layer = getattr(blk.attn, name)
            if isinstance(layer, LoRALinear):
                layer.base.weight.data += (layer.B @ layer.A) * layer.scale
                setattr(blk.attn, name, layer.base)


# ---------- load the pretrained model ----------
ckpt = torch.load("out/ckpt.pt", map_location="cpu")
cfg = GPTConfig(**ckpt["cfg"])
model = GPT(cfg)
model.load_state_dict(ckpt["model"])
if use_lora:
    add_lora(model)
model.to(device)


# ---------- build training examples ----------
def build(prompt, response):
    p_ids = tok.encode(f"### User:\n{prompt}\n### Assistant:\n")
    r_ids = tok.encode(response + "<|end|>")
    ids = (p_ids + r_ids)[: cfg.block_size + 1]
    labels = ([-100] * len(p_ids) + r_ids)[: cfg.block_size + 1]  # no loss on the prompt
    return ids, labels


rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
examples = [build(r["prompt"], r["response"]) for r in rows]
print(f"Loaded {len(examples)} examples from {data_path} (use_lora={use_lora}, device={device})")


def make_batch(exs, bs):
    batch = random.sample(exs, min(bs, len(exs)))
    L = max(len(ids) for ids, _ in batch) - 1
    x = torch.zeros(len(batch), L, dtype=torch.long)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    for k, (ids, labels) in enumerate(batch):
        n = len(ids) - 1
        x[k, :n] = torch.tensor(ids[:-1])
        y[k, :n] = torch.tensor(labels[1:])
    return x.to(device), y.to(device)


# ---------- train ----------
steps, bs = 1000, 8
lr = 2e-3 if use_lora else 3e-4
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

os.makedirs("out", exist_ok=True)
model.train()
for step in range(steps):
    x, y = make_batch(examples, bs)
    with amp:
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if step % 50 == 0 or step == steps - 1:
        print(f"step {step:4d} | loss {loss.item():.3f}")

if use_lora:
    merge_lora(model)
torch.save({"model": model.state_dict(), "cfg": ckpt["cfg"]}, "out/sft.pt")
print("Saved fine-tuned checkpoint to out/sft.pt")

STOP_PATTERNS = ["<|end|>", "<|", "\n###", "### User:", "###"]


# ---------- try it ----------
def ask(question, max_new=100):
    model.eval()
    prompt = f"### User:\n{question}\n### Assistant:\n"
    idx = torch.tensor([tok.encode(prompt)], device=device)
    new_tokens = []
    with torch.no_grad():
        for _ in range(max_new):
            idx_cond = idx[:, -model.cfg.block_size:]
            logits, _ = model(idx_cond)
            probs = torch.softmax(logits[:, -1, :] / 0.6, dim=-1)
            v, _ = torch.topk(probs, 40)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            new_tokens.append(next_id.item())
            text = tok.decode(new_tokens)
            if any(stop in text for stop in STOP_PATTERNS):
                break

    ans = tok.decode(new_tokens)
    for stop in STOP_PATTERNS:
        if stop in ans:
            ans = ans.split(stop)[0]
    return ans.strip()


print("\nTesting sample prompts:")
for p in ["Hello", "What are your opening hours?", "How do I reset my password?"]:
    print(f"User: {p}")
    print(f"Assistant: {ask(p)}\n")