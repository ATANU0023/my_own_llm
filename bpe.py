import re
from collections import Counter

PAT = re.compile(r"\s?\w+|\s?[^\w\s]+|\s")


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.cache = {}

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text, vocab_size):
        words = Counter(PAT.findall(text))  # piece -> count
        seqs = {w: list(w.encode("utf-8")) for w in words}  # piece -> byte ids
        for new_id in range(256, vocab_size):
            stats = Counter()
            for w, count in words.items():
                s = seqs[w]
                for pair in zip(s, s[1:]):
                    stats[pair] += count
            if not stats:
                break
            best = max(stats, key=stats.get)  # most frequent pair
            for w in seqs:
                seqs[w] = self._merge(seqs[w], best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if new_id % 200 == 0:
                print(f"merge {new_id - 255}/{vocab_size - 256}")

    def _encode_piece(self, piece):
        if piece in self.cache:
            return self.cache[piece]
        ids = list(piece.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        self.cache[piece] = ids
        return ids

    def encode(self, text):
        out = []
        for piece in PAT.findall(text):
            out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
