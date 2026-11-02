import pickle
import numpy as np 
from bpe import BPETokenizer

VOCAB_SIZE = 2000

text = open("data/input.txt", encoding="utf-8").read()
tok = BPETokenizer()
tok.train(text, VOCAB_SIZE)

ids = np.array(tok.encode(text), dtype=np.uint16)
n = int(0.9 * len(ids))
ids[:n].tofile("data/train.bin")
ids[n:].tofile("data/val.bin")

pickle.dump(tok, open("data/tok.pkl", "wb"))
pickle.dump({"vocab_size": len(tok.vocab)}, open("data/meta.pkl", "wb"))

print(f"{len(ids):,} tokens | vocab {len(tok.vocab)} | train {n:,} | val {len(ids) - n:,}")