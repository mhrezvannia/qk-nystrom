"""Download Fashion-MNIST and MNIST idx files (GitHub mirrors) and convert to data/*.npz."""
import gzip, urllib.request, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
SRC = {"fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
       "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/"}
FILES = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"), "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}
for name, base in SRC.items():
    for split, (fx, fy) in FILES.items():
        for f in (fx, fy):
            p = RAW / f"{name}-{f}"
            if not p.exists(): urllib.request.urlretrieve(base + f, p)
        x = np.frombuffer(gzip.open(RAW / f"{name}-{fx}").read(), np.uint8, offset=16).reshape(-1, 784)
        y = np.frombuffer(gzip.open(RAW / f"{name}-{fy}").read(), np.uint8, offset=8)
        np.savez_compressed(ROOT / "data" / f"{name}_{split}.npz", x=x, y=y); print(name, split, x.shape)
