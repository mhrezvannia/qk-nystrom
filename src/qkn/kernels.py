"""Quantum fidelity kernels, noise models, classical kernels, and circuit-evaluation accounting."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel


# ----------------------------------------------------------------------------- quantum kernel
def fidelity_kernel(states_A: np.ndarray, states_B: np.ndarray) -> np.ndarray:
    """K_ij = |<phi(a_i)|phi(b_j)>|^2 from precomputed statevectors."""
    return np.abs(states_A @ states_B.conj().T) ** 2


def shot_noise(K: np.ndarray, shots: int | None, rng: np.random.Generator, symmetric: bool = False) -> np.ndarray:
    """Finite-shot estimate of each fidelity via the inversion test: p_hat = Binomial(S, K)/S."""
    if shots is None or shots <= 0:
        return K.copy()
    Kc = np.clip(K, 0.0, 1.0)
    Kh = rng.binomial(shots, Kc) / shots
    if symmetric:  # each unordered pair is estimated once
        iu = np.triu_indices_from(Kh, k=1)
        Kh[(iu[1], iu[0])] = Kh[iu]
        np.fill_diagonal(Kh, 1.0)  # diagonal fidelity is known to be 1 and need not be measured
    return Kh


def depolarize(K: np.ndarray, lam: float, n_qubits: int) -> np.ndarray:
    """Global depolarising channel on the inversion-test state: p(0..0) -> (1-lam) K + lam / 2^n."""
    return (1 - lam) * K + lam / (2 ** n_qubits)


# ----------------------------------------------------------------------------- classical kernels
def kernel_matrix(kind: str, A: np.ndarray, B: np.ndarray, gamma: float | None = None) -> np.ndarray:
    if kind == "linear":
        return linear_kernel(A, B)
    if kind == "rbf":
        return rbf_kernel(A, B, gamma=gamma)
    if kind == "product_cosine":
        return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=2)
    raise ValueError(kind)


def random_fourier_features(A: np.ndarray, D: int, gamma: float, rng: np.random.Generator):
    """Rahimi & Recht random Fourier features for the RBF kernel exp(-gamma ||a-b||^2)."""
    d = A.shape[1]
    Wf = rng.normal(0, np.sqrt(2 * gamma), (d, D))
    b = rng.uniform(0, 2 * np.pi, D)
    return np.sqrt(2.0 / D) * np.cos(A @ Wf + b), (Wf, b)


def rff_transform(A, Wf, b):
    return np.sqrt(2.0 / Wf.shape[1]) * np.cos(A @ Wf + b)


# ----------------------------------------------------------------------------- accounting
@dataclass
class CircuitCounter:
    """Counts distinct fidelity estimations (each one circuit configuration executed S times)."""
    train_train: int = 0
    landmark_landmark: int = 0
    train_landmark: int = 0
    test_landmark: int = 0
    test_train: int = 0
    shots: int | None = None

    @staticmethod
    def exact(n_train: int, n_test: int, shots=None) -> "CircuitCounter":
        return CircuitCounter(train_train=n_train * (n_train - 1) // 2, test_train=n_test * n_train, shots=shots)

    @staticmethod
    def nystrom(n_train: int, n_test: int, m: int, landmarks_are_data: bool, shots=None) -> "CircuitCounter":
        c = CircuitCounter(landmark_landmark=m * (m - 1) // 2, test_landmark=n_test * m, shots=shots)
        # if landmarks are training points, the m x m block is a subset of the n_train x m block
        c.train_landmark = n_train * m - (m * (m + 1) // 2 if landmarks_are_data else 0)
        return c

    @property
    def train_total(self) -> int:
        return self.train_train + self.landmark_landmark + self.train_landmark

    @property
    def test_total(self) -> int:
        return self.test_landmark + self.test_train

    @property
    def total(self) -> int:
        return self.train_total + self.test_total

    def shot_weighted(self) -> int | None:
        return None if self.shots is None else self.total * self.shots

    def as_dict(self) -> dict:
        return dict(circ_train=self.train_total, circ_test=self.test_total, circ_total=self.total,
                    circ_shot_weighted=self.shot_weighted())
