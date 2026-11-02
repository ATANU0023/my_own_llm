import pathlib
import re

def clean(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

files = sorted(pathlib.Path("raw").glob("**/*.txt"))
parts = [clean(f.read_text(encoding="utf-8", errors="ignore")) for f in files]

pathlib.Path("data").mkdir(exist_ok=True)
pathlib.Path("data/input.txt").write_text("\n\n".join(parts), encoding="utf-8")
print(f"{len(files)} files merged, {sum(len(p) for p in parts):,} characters")
