# chat.py - usage: python chat.py [out/sft.pt]
import pickle
import sys
import torch
from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "out/sft.pt"

print(f"Loading checkpoint from {ckpt_path} onto {device}...")
ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(GPTConfig(**ckpt["cfg"])).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tok = pickle.load(open("data/tok.pkl", "rb"))
print("Model ready! Ask a question. Press Enter on an empty line or Ctrl+C to quit.\n")

STOP_PATTERNS = ["<|end|>", "<|", "\n###", "### User:", "###"]


def generate_reply(question, max_new_tokens=100, temperature=0.6, top_k=40):
    prompt = f"### User:\n{question}\n### Assistant:\n"
    idx = torch.tensor([tok.encode(prompt)], device=device)
    new_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.cfg.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            new_tokens.append(next_id.item())

            # Decode so far to see if a stop marker was hit
            partial_text = tok.decode(new_tokens)
            if any(stop in partial_text for stop in STOP_PATTERNS):
                break

    reply = tok.decode(new_tokens)
    for stop in STOP_PATTERNS:
        if stop in reply:
            reply = reply.split(stop)[0]
    return reply.strip()


try:
    while True:
        question = input("You: ").strip()
        if not question:
            break

        response = generate_reply(question)
        print(f"Model: {response}\n")
except (KeyboardInterrupt, EOFError):
    print("\nGoodbye!")