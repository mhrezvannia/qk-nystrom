"""Quantum feature maps studied in this work.

Every map exposes ``states(X, theta) -> (B, 2**n)`` complex array and ``n_params``.
All maps operate on angle-encoded inputs X in R^{B x n} (already scaled to [0, a]).

Maps
----
ProductCosine      -- the previous manuscript's construction: |phi(x)> = RZ(x) |+>^n
                      (equivalently H, RZ(x), then ANY data-independent unitary).  Closed form
                      K = prod_j cos^2((x_j - x'_j)/2).  Serves as baseline / ablation arm.
CovariantTorus     -- U(1)^n-covariant kernel with trainable fiducial state:
                      |phi(x)> = D(x) V(theta) |0>,  D(x) = (x)RZ(x_j).
                      K(x,x') = |<0|V^dag D(x'-x) V|0>|^2 = |sum_b p_b e^{i (x'-x).s_b/2}|^2,
                      where p_b = |<b|V|0>|^2 is the Z-basis distribution of the fiducial state
                      and s_b in {+-1}^n.  Shift-invariant (covariant under torus translations).
                      Parameters act BEFORE the data, so they do not cancel.
ReUploading        -- data re-uploading: |phi(x)> = RZ(x) prod_{l=L..1}[E RY(theta_l) RZ(x)] H^n|0>
                      (a data layer follows every parameterised/entangling block, so nothing cancels).
ZZReUploading      -- re-uploading with data-dependent ZZ entanglers exp(-i x_j x_k Z_j Z_k / 2)
                      on a ring (IQP-type), plus trainable RY between data blocks.
"""
from __future__ import annotations
import numpy as np
from .simulator import Sim


class FeatureMap:
    name = "base"
    n_params = 0

    def __init__(self, n: int, layers: int = 1, entangle: bool = True):
        self.n, self.L, self.entangle = n, layers, entangle
        self.sim = Sim(n)

    def states(self, X: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def init_params(self, rng: np.random.Generator, scheme: str = "uniform") -> np.ndarray:
        if self.n_params == 0:
            return np.zeros(0)
        if scheme == "uniform":
            return rng.uniform(0, 2 * np.pi, self.n_params)
        if scheme == "small":
            return rng.normal(0, 0.1, self.n_params)
        if scheme == "zeros":
            return np.zeros(self.n_params)
        raise ValueError(scheme)

    def describe(self) -> dict:
        return dict(map=self.name, n_qubits=self.n, layers=self.L, entangle=self.entangle, n_params=self.n_params)


class ProductCosine(FeatureMap):
    name = "product_cosine"

    def states(self, X, theta=None):
        return self.sim.rz_data(self.sim.plus(X.shape[0]), X)

    @staticmethod
    def closed_form(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.prod(np.cos((A[:, None, :] - B[None, :, :]) / 2) ** 2, axis=2)


class CovariantTorus(FeatureMap):
    """Fiducial circuit V(theta): L blocks of [RY(theta_{l,j}) on each qubit, then CNOT ring if entangle].
    Data enters last, as D(x) = (x) RZ(x_j).  Note that a leading H layer would be a special case
    (theta = pi/2, no entanglement) giving the ProductCosine kernel."""
    name = "covariant_torus"

    def __init__(self, n, layers=2, entangle=True):
        super().__init__(n, layers, entangle)
        self.n_params = layers * n

    def fiducial(self, theta):
        psi = self.sim.zero(1)
        th = np.asarray(theta).reshape(self.L, self.n)
        for l in range(self.L):
            psi = self.sim.ry_layer(psi, th[l])
            if self.entangle:
                psi = self.sim.cnot_ring(psi)
        return psi  # (1, dim)

    def states(self, X, theta):
        fid = self.fiducial(theta)  # (1, dim)
        psi = np.repeat(fid, X.shape[0], axis=0)
        return self.sim.rz_data(psi, X)

    def closed_form(self, A, B, theta):
        """K(a,b) = |sum_b p_b exp(i (b-a).s_b / 2)|^2 with p the fiducial Z-distribution."""
        p = self.sim.probs(self.fiducial(theta))[0]  # (dim,)
        delta = (B[None, :, :] - A[:, None, :])  # (NA, NB, n)
        phase = np.exp(0.5j * np.tensordot(delta, self.sim.sgn.T, axes=([2], [0])))  # (NA, NB, dim)
        return np.abs(phase @ p) ** 2


class ReUploading(FeatureMap):
    name = "reuploading"

    def __init__(self, n, layers=2, entangle=True):
        super().__init__(n, layers, entangle)
        self.n_params = layers * n

    def states(self, X, theta):
        th = np.asarray(theta).reshape(self.L, self.n)
        psi = self.sim.h_layer(self.sim.zero(X.shape[0]))
        for l in range(self.L):
            psi = self.sim.rz_data(psi, X)
            psi = self.sim.ry_layer(psi, th[l])
            if self.entangle:
                psi = self.sim.cnot_ring(psi)
        return self.sim.rz_data(psi, X)  # final data layer: nothing after it is data-independent


class ZZReUploading(FeatureMap):
    name = "zz_reuploading"

    def __init__(self, n, layers=2, entangle=True):
        super().__init__(n, layers, entangle)
        self.n_params = layers * n
        self.pairs = [(q, (q + 1) % n) for q in range(n)]

    def _zz(self, psi, X):
        if not self.entangle:
            return psi
        ang = np.stack([X[:, j] * X[:, k] for j, k in self.pairs], axis=1)
        return self.sim.zz_data(psi, ang, self.pairs)

    def states(self, X, theta):
        th = np.asarray(theta).reshape(self.L, self.n)
        psi = self.sim.h_layer(self.sim.zero(X.shape[0]))
        for l in range(self.L):
            psi = self.sim.rz_data(psi, X)
            psi = self._zz(psi, X)
            psi = self.sim.ry_layer(psi, th[l])
        psi = self.sim.rz_data(psi, X)
        return self._zz(psi, X)


REGISTRY = {c.name: c for c in [ProductCosine, CovariantTorus, ReUploading, ZZReUploading]}


def make_map(name: str, n: int, layers: int = 2, entangle: bool = True) -> FeatureMap:
    return REGISTRY[name](n, layers=layers, entangle=entangle)
