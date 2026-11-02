# sample.py
import pickle
import sys
import torch
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "out/ckpt.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig(**ckpt["cfg"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tok = pickle.load(open("data/tok.pkl", "rb"))
prompt = "Once upon a time"  # change this to start of your own text
idx = torch.tensor([tok.encode(prompt)], device=device)
out = model.generate(idx, max_new_tokens=200, temperature=0.8, top_k=40)
print(tok.decode(out[0].tolist()))