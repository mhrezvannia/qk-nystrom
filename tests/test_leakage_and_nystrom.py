import numpy as np, sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from qkn.data import Preprocessor
from qkn.nystrom import Nystrom, select_landmarks, approximation_metrics
from qkn.kernels import kernel_matrix, CircuitCounter, shot_noise

def test_preprocessor_refuses_non_train():
    with pytest.raises(AssertionError):
        Preprocessor(2).fit(np.random.rand(10, 5), train_tag="test")

def test_preprocessor_unit_range_on_train_only():
    rng = np.random.default_rng(0); Xtr = rng.normal(size=(100, 20)); Xte = rng.normal(size=(50, 20)) + 3
    pp = Preprocessor(4).fit(Xtr)
    U = pp.transform_unit(Xtr); assert np.allclose(U.min(0), 0) and np.allclose(U.max(0), 1)
    Ute = pp.transform_unit(Xte); assert (Ute.max() > 1) or (Ute.min() < 0)  # test may leave [0,1] -> proves no refit

def test_nystrom_exact_when_m_equals_n():
    rng = np.random.default_rng(1); X = rng.normal(size=(50, 3)); K = kernel_matrix("rbf", X, X, gamma=0.5)
    ny = Nystrom(eps=0.0).fit(K); Kt = ny.transform(K)
    assert approximation_metrics(K, Kt)["E_F"] < 1e-8

def test_landmark_strategies_shapes():
    rng = np.random.default_rng(2); X = rng.normal(size=(80, 4)); K = kernel_matrix("rbf", X, X, gamma=0.5)
    for s in ["uniform", "kmeanspp", "kmeans_centroids", "kmeans_nearest", "leverage"]:
        Z, idx = select_landmarks(X, 12, s, rng, K_full=K)
        assert Z.shape == (12, 4)
        if idx is not None: assert len(np.unique(idx)) == 12

def test_circuit_counter_symmetric():
    c = CircuitCounter.exact(600, 2000); assert c.train_train == 600*599//2 and c.test_train == 2000*600
    n = CircuitCounter.nystrom(600, 2000, 60, landmarks_are_data=False)
    assert n.train_total == 600*60 + 60*59//2 and n.test_total == 2000*60

def test_shot_noise_unbiased():
    rng = np.random.default_rng(0); K = np.full((200, 200), 0.3); Kh = shot_noise(K, 1000, rng)
    assert abs(Kh.mean() - 0.3) < 0.005
