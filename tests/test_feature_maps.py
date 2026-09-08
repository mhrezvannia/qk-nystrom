"""Phase-D mathematical validation tests.  A candidate feature map may enter the experimental campaign
only if every test in this file passes for it (see experiments/run_validation.py for the numeric report)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qkn.feature_maps import make_map, ProductCosine, CovariantTorus
from qkn.kernels import fidelity_kernel

N_Q = 8
rng = np.random.default_rng(123)
X = rng.uniform(0, 2 * np.pi, (40, N_Q))
Xp = rng.uniform(0, 2 * np.pi, (40, N_Q))
CANDIDATES = ["covariant_torus", "reuploading", "zz_reuploading"]


def K_of(name, theta=None, entangle=True):
    fm = make_map(name, N_Q, layers=2, entangle=entangle)
    th = fm.init_params(np.random.default_rng(7)) if theta is None else theta
    S = fm.states(X, th)
    return fidelity_kernel(S, S), fm, th


def test_previous_construction_cancels():
    """Regression test of the review finding: RY(theta)+CNOT after RZ encoding do not change the kernel."""
    from qkn.simulator import Sim
    sim = Sim(N_Q); psi0 = sim.rz_data(sim.plus(40), X)
    K0 = fidelity_kernel(psi0, psi0)
    for th in [0.0, 1.0, 2.5]:
        psi = sim.ry_layer(sim.cnot_ring(psi0), th)
        assert np.allclose(fidelity_kernel(psi, psi), K0, atol=1e-12)
    assert np.allclose(K0, ProductCosine.closed_form(X, X), atol=1e-12)


def test_theta_changes_kernel():
    for name in CANDIDATES:
        K1, fm, th = K_of(name)
        K2 = fidelity_kernel(fm.states(X, th + 0.7), fm.states(X, th + 0.7))
        assert np.abs(K1 - K2).max() > 1e-3, f"{name}: theta has no effect"


def test_entanglement_changes_kernel():
    for name in CANDIDATES:
        fm_e = make_map(name, N_Q, 2, True); fm_n = make_map(name, N_Q, 2, False)
        th = fm_e.init_params(np.random.default_rng(7))
        Ke = fidelity_kernel(fm_e.states(X, th), fm_e.states(X, th))
        Kn = fidelity_kernel(fm_n.states(X, th), fm_n.states(X, th))
        assert np.abs(Ke - Kn).max() > 1e-3, f"{name}: entanglement has no effect"


def test_psd_symmetric_unit_diagonal():
    for name in CANDIDATES + ["product_cosine"]:
        K, fm, th = K_of(name)
        assert np.allclose(K, K.T, atol=1e-12)
        assert np.allclose(np.diag(K), 1.0, atol=1e-10)
        assert np.linalg.eigvalsh(K).min() > -1e-9


def test_not_trivial_or_global_phase():
    for name in CANDIDATES:
        K, fm, th = K_of(name)
        off = K[~np.eye(40, dtype=bool)]
        assert off.std() > 1e-3 and off.mean() < 0.99, f"{name}: kernel is (nearly) constant"


def test_encoding_on_zero_state_is_trivial_regression():
    """The old two-moons circuit: RZ on |0> is a global phase -> constant kernel.  Kept as a guard."""
    from qkn.simulator import Sim
    sim = Sim(4); psi = sim.rz_data(sim.zero(20), X[:20, :4])
    assert np.allclose(fidelity_kernel(psi, psi), 1.0)


def test_covariant_torus_closed_form_and_shift_invariance():
    fm = CovariantTorus(N_Q, 2, True); th = fm.init_params(np.random.default_rng(3))
    K_sim = fidelity_kernel(fm.states(X, th), fm.states(Xp, th))
    K_cf = fm.closed_form(X, Xp, th)
    assert np.allclose(K_sim, K_cf, atol=1e-10)
    h = rng.uniform(0, 2 * np.pi, N_Q)
    K_shift = fidelity_kernel(fm.states(X + h, th), fm.states(Xp + h, th))
    assert np.allclose(K_shift, K_sim, atol=1e-10)          # covariance under torus translations
    # a re-uploading map is NOT shift invariant (sanity: the property is specific to the covariant map)
    fr = make_map("reuploading", N_Q, 2, True); thr = fr.init_params(np.random.default_rng(3))
    Kr = fidelity_kernel(fr.states(X, thr), fr.states(Xp, thr))
    Kr_s = fidelity_kernel(fr.states(X + h, thr), fr.states(Xp + h, thr))
    assert np.abs(Kr - Kr_s).max() > 1e-3


def test_covariant_torus_product_fiducial_gives_product_kernel():
    fm = CovariantTorus(N_Q, 1, entangle=False); th = rng.uniform(0, 2 * np.pi, N_Q)
    K = fidelity_kernel(fm.states(X, th), fm.states(Xp, th))
    m = np.cos(th)  # <Z_j> of RY(theta_j)|0>
    D = Xp[None, :, :] - X[:, None, :]
    K_pred = np.prod(np.cos(D / 2) ** 2 + (m ** 2)[None, None, :] * np.sin(D / 2) ** 2, axis=2)
    assert np.allclose(K, K_pred, atol=1e-10)
    # theta = pi/2 on every qubit (|+>^n) recovers the product-cosine kernel exactly
    K2 = fidelity_kernel(fm.states(X, np.full(N_Q, np.pi / 2)), fm.states(Xp, np.full(N_Q, np.pi / 2)))
    assert np.allclose(K2, ProductCosine.closed_form(X, Xp), atol=1e-10)


def test_numerical_stability():
    for name in CANDIDATES:
        K, fm, th = K_of(name)
        assert np.isfinite(K).all() and K.max() <= 1 + 1e-9 and K.min() >= -1e-12
