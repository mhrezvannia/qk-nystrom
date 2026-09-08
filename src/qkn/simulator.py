"""Batched exact statevector simulator for the structured circuits used in this study.

Convention: qubit 0 is the most-significant bit of the basis index (PennyLane convention).
States are arrays of shape (B, 2**n), one row per data sample.  All data-dependent gates used
here are diagonal in the computational basis (RZ, ZZ) and are applied as per-sample phase
vectors; parameterised single-qubit gates (RY) and CNOT/CZ are applied by tensor reshaping.
Verified against PennyLane ``default.qubit`` in tests/test_simulator.py.
"""
from __future__ import annotations
import numpy as np


class Sim:
    def __init__(self, n: int):
        self.n = n
        self.dim = 2 ** n
        idx = np.arange(self.dim)
        # bits[b, j] = value of qubit j in basis state b (qubit 0 = MSB)
        self.bits = ((idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
        self.sgn = (1 - 2 * self.bits).astype(np.float64)  # Z eigenvalue (+1 for |0>, -1 for |1>)
        self._cnot_perm = {}

    # ---------- state preparation ----------
    def zero(self, B: int) -> np.ndarray:
        psi = np.zeros((B, self.dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        return psi

    def plus(self, B: int) -> np.ndarray:
        return np.full((B, self.dim), 1.0 / np.sqrt(self.dim), dtype=np.complex128)

    # ---------- diagonal data gates ----------
    def rz_data(self, psi: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Apply RZ(x_j) on qubit j for every j; X has shape (B, n). RZ(phi)=diag(e^{-i phi/2}, e^{i phi/2})."""
        phase = np.exp(-0.5j * (X @ self.sgn.T))  # (B, dim)
        return psi * phase

    def zz_data(self, psi: np.ndarray, angles: np.ndarray, pairs) -> np.ndarray:
        """Apply exp(-i angle_p/2 Z_j Z_k) for each pair p=(j,k); angles has shape (B, len(pairs))."""
        zz = np.stack([self.sgn[:, j] * self.sgn[:, k] for (j, k) in pairs], axis=1)  # (dim, P)
        phase = np.exp(-0.5j * (angles @ zz.T))
        return psi * phase

    # ---------- parameterised single-qubit gates ----------
    def _apply_1q(self, psi: np.ndarray, U: np.ndarray, q: int) -> np.ndarray:
        B = psi.shape[0]
        left, right = 2 ** q, 2 ** (self.n - q - 1)
        t = psi.reshape(B, left, 2, right)
        t = np.einsum('ab,Blbr->Blar', U, t)
        return t.reshape(B, self.dim)

    @staticmethod
    def ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)

    @staticmethod
    def rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)

    def _kron_all(self, mats) -> np.ndarray:
        M = np.array([[1.0 + 0j]])
        for U in mats:
            M = np.kron(M, U)
        return M

    def ry_layer(self, psi: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """RY(thetas[j]) on qubit j (thetas broadcastable to length n).  Implemented as one dense
        (2^n x 2^n) tensor-product matrix, which is faster than n sequential gates for n <= 10."""
        thetas = np.broadcast_to(np.asarray(thetas, dtype=float), (self.n,))
        M = self._kron_all([self.ry(t) for t in thetas])  # qubit 0 = MSB -> leftmost factor
        return psi @ M.T

    def h_layer(self, psi: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_Hn"):
            self._Hn = self._kron_all([self.H] * self.n)
        return psi @ self._Hn.T

    # ---------- entanglers ----------
    def cnot(self, psi: np.ndarray, c: int, t: int) -> np.ndarray:
        key = (c, t)
        if key not in self._cnot_perm:
            perm = np.arange(self.dim)
            mask = self.bits[:, c] == 1
            perm[mask] = perm[mask] ^ (1 << (self.n - 1 - t))
            self._cnot_perm[key] = perm
        return psi[:, self._cnot_perm[key]]

    def cnot_ring(self, psi: np.ndarray) -> np.ndarray:
        for q in range(self.n - 1):
            psi = self.cnot(psi, q, q + 1)
        return self.cnot(psi, self.n - 1, 0)

    def cz_ring(self, psi: np.ndarray) -> np.ndarray:
        sign = np.ones(self.dim)
        for q in range(self.n):
            j, k = q, (q + 1) % self.n
            sign *= np.where((self.bits[:, j] == 1) & (self.bits[:, k] == 1), -1.0, 1.0)
        return psi * sign

    # ---------- measurement distribution ----------
    def probs(self, psi: np.ndarray) -> np.ndarray:
        return np.abs(psi) ** 2
