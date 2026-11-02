import torch
from model import GPT, GPTConfig
model = GPT(GPTConfig(vocab_size=2000))
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
x = torch.randint(0, 2000, (2, 16))
logits, loss = model(x, x)
print(logits.shape, loss.item())
