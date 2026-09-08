"""Datasets, splits and leakage-safe preprocessing.

Leakage policy (enforced by assertions and tests/test_leakage.py):
  * PCA and min-max scaling are fitted on TRAINING data only (Preprocessor.fit raises if it sees test ids).
  * Landmark selection and parameter training use training data only.
  * Hyper-parameters (C, gamma, scaling range a) are chosen on a validation split carved from the training set.
  * The official test split is touched exactly once per (task, seed), for the final numbers.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

FASHION_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


@dataclass
class Task:
    dataset: str          # 'fashion' | 'mnist' | 'breast_cancer'
    classes: tuple        # (neg, pos) labels for image datasets; () for tabular
    name: str

    @property
    def key(self):
        return f"{self.dataset}_{self.classes[0]}v{self.classes[1]}" if self.classes else self.dataset


def make_task(spec: dict) -> Task:
    ds = spec["dataset"]
    if ds in ("fashion", "mnist"):
        a, b = spec["classes"]
        nm = f"{FASHION_NAMES[a]} vs {FASHION_NAMES[b]}" if ds == "fashion" else f"{a} vs {b}"
        return Task(ds, (a, b), nm)
    return Task(ds, (), ds)


def _load_npz(name, split):
    z = np.load(DATA_DIR / f"{name}_{split}.npz")
    return z["x"].astype(np.float32), z["y"].astype(int)


def load_split(task: Task, n_train: int, seed: int, test_fraction: float = 0.3):
    """Return X_train, y_train, X_test, y_test (raw features, labels in {0,1}).

    Image tasks: n_train stratified samples from the OFFICIAL train split; the ENTIRE official test split for
    the two classes is the test set.  Tabular: stratified train/test split with the given seed."""
    rng = np.random.default_rng(seed)
    if task.dataset in ("fashion", "mnist"):
        xtr, ytr = _load_npz(task.dataset, "train"); xte, yte = _load_npz(task.dataset, "test")
        a, b = task.classes
        mtr = (ytr == a) | (ytr == b); mte = (yte == a) | (yte == b)
        Xtr, Ytr = xtr[mtr], (ytr[mtr] == b).astype(int)
        Xte, Yte = xte[mte], (yte[mte] == b).astype(int)
        # stratified sub-sample of the training pool
        idx = []
        for c in (0, 1):
            pool = np.where(Ytr == c)[0]
            idx.append(rng.choice(pool, n_train // 2, replace=False))
        idx = np.concatenate(idx); rng.shuffle(idx)
        return Xtr[idx], Ytr[idx], Xte, Yte
    if task.dataset == "breast_cancer":
        d = load_breast_cancer()
        Xtr, Xte, ytr, yte = train_test_split(d.data.astype(np.float32), d.target, test_size=test_fraction,
                                              random_state=seed, stratify=d.target)
        if n_train < len(ytr):
            sub = np.random.default_rng(seed).choice(len(ytr), n_train, replace=False)
            Xtr, ytr = Xtr[sub], ytr[sub]
        return Xtr, ytr, Xte, yte
    raise ValueError(task.dataset)


class Preprocessor:
    """PCA(k) -> per-feature min-max to [0, 1] (fitted on training data only) -> scale to [0, a]."""

    def __init__(self, n_components: int = 8):
        self.k = n_components
        self._fitted_on = None

    def fit(self, X_train: np.ndarray, train_tag: str = "train"):
        assert train_tag == "train", "Preprocessor must be fitted on training data only"
        self.pca = PCA(n_components=self.k, random_state=0).fit(X_train)
        Z = self.pca.transform(X_train)
        self.lo, self.hi = Z.min(0), Z.max(0)
        self._fitted_on = X_train.shape
        self.explained_variance_ratio_ = float(self.pca.explained_variance_ratio_.sum())
        return self

    def transform_unit(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted_on is not None, "fit on training data first"
        Z = self.pca.transform(X)
        return (Z - self.lo) / np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)

    @staticmethod
    def to_angles(U: np.ndarray, a: float) -> np.ndarray:
        """Scale unit-range features to angles in [0, a]."""
        return a * U
