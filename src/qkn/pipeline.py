"""Shared, leakage-safe experimental pipeline used by every experiment script."""
from __future__ import annotations
import time, json, platform
import numpy as np, yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from .data import make_task, load_split, Preprocessor
from .feature_maps import make_map
from .kernels import fidelity_kernel, kernel_matrix, CircuitCounter
from .optimization import train_kta
from .metrics import kta, classification_metrics

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "configs" / "main.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare(task_spec: dict, seed: int, cfg: dict) -> dict:
    """Load, split, and preprocess one (task, seed).  Returns unit-scaled features; angles are made later."""
    task = make_task(task_spec)
    Xtr_raw, ytr, Xte_raw, yte = load_split(task, cfg["data"]["n_train"], seed, cfg["data"]["tabular_test_fraction"])
    pp = Preprocessor(cfg["preprocessing"]["pca_components"]).fit(Xtr_raw, "train")
    Utr, Ute = pp.transform_unit(Xtr_raw), pp.transform_unit(Xte_raw)
    # validation split for hyper-parameter selection (from training data only)
    i_fit, i_val = train_test_split(np.arange(len(ytr)), test_size=cfg["data"]["val_fraction"],
                                    random_state=seed, stratify=ytr)
    return dict(task=task, seed=seed, U_train=Utr, y_train=ytr, U_test=Ute, y_test=yte, i_fit=i_fit, i_val=i_val,
                n_train=len(ytr), n_test=len(yte), class_counts_train=np.bincount(ytr).tolist(),
                class_counts_test=np.bincount(yte).tolist(), pca_evr=pp.explained_variance_ratio_)


def svm_precomputed(K_train, y_train, K_test_train, y_test, C):
    clf = SVC(kernel="precomputed", C=C).fit(K_train, y_train)
    return classification_metrics(y_test, clf.predict(K_test_train))


def select_C(K_fit, y_fit, K_val_fit, y_val, C_grid):
    best = (-1, None)
    for C in C_grid:
        acc = svm_precomputed(K_fit, y_fit, K_val_fit, y_val, C)["acc"]
        if acc > best[0]:
            best = (acc, C)
    return best[1], best[0]


# ----------------------------------------------------------------------------- classical baselines
def classical_baselines(P: dict, cfg: dict) -> dict:
    """Linear SVM, RBF SVM, logistic regression on the same unit-scaled PCA features. Hyper-parameters on val."""
    Utr, ytr, Ute, yte = P["U_train"], P["y_train"], P["U_test"], P["y_test"]
    i_fit, i_val = P["i_fit"], P["i_val"]
    out = {}
    # linear
    C_lin, _ = select_C(kernel_matrix("linear", Utr[i_fit], Utr[i_fit]), ytr[i_fit],
                        kernel_matrix("linear", Utr[i_val], Utr[i_fit]), ytr[i_val], cfg["svm"]["C_grid"])
    t0 = time.time()
    r = svm_precomputed(kernel_matrix("linear", Utr, Utr), ytr, kernel_matrix("linear", Ute, Utr), yte, C_lin)
    out["svm_linear"] = dict(r, C=C_lin, runtime=time.time() - t0)
    # rbf: gamma relative to 1/(d*var)
    g0 = 1.0 / (Utr.shape[1] * Utr.var())
    best = (-1, None, None)
    for gf in cfg["svm"]["rbf_gamma_grid"]:
        g = gf * g0
        C, acc = select_C(kernel_matrix("rbf", Utr[i_fit], Utr[i_fit], g), ytr[i_fit],
                          kernel_matrix("rbf", Utr[i_val], Utr[i_fit], g), ytr[i_val], cfg["svm"]["C_grid"])
        if acc > best[0]:
            best = (acc, g, C)
    _, g, C = best
    t0 = time.time()
    r = svm_precomputed(kernel_matrix("rbf", Utr, Utr, g), ytr, kernel_matrix("rbf", Ute, Utr, g), yte, C)
    out["svm_rbf"] = dict(r, C=C, gamma=g, runtime=time.time() - t0)
    # logistic regression
    best = (-1, None)
    for C in cfg["svm"]["logreg_C_grid"]:
        acc = LogisticRegression(C=C, max_iter=5000).fit(Utr[i_fit], ytr[i_fit]).score(Utr[i_val], ytr[i_val])
        if acc > best[0]:
            best = (acc, C)
    t0 = time.time()
    lr = LogisticRegression(C=best[1], max_iter=5000).fit(Utr, ytr)
    out["logreg"] = dict(classification_metrics(yte, lr.predict(Ute)), C=best[1], runtime=time.time() - t0)
    out["_rbf_gamma"] = g
    return out


# ----------------------------------------------------------------------------- quantum kernel tuning
def tune_quantum_kernel(P: dict, cfg: dict, map_name: str, layers: int, entangle: bool, train_theta: bool = True,
                        scaling_ranges=None, theta_init=None, rank_lambda: float = 0.0):
    """Select scaling range a (and train theta by KTA for each a) on the validation split; then C on val.
    Returns dict with theta, a, C, val_acc, KTA values and trajectory.  Test data are not touched."""
    Utr, ytr, i_fit, i_val = P["U_train"], P["y_train"], P["i_fit"], P["i_val"]
    ranges = scaling_ranges or cfg["preprocessing"]["scaling_ranges"]
    fm = make_map(map_name, cfg["data"]["n_qubits"], layers, entangle)
    rng = np.random.default_rng(P["seed"] * 1000 + 17)
    best = None
    for a in ranges:
        Xfit, Xval = Preprocessor.to_angles(Utr[i_fit], a), Preprocessor.to_angles(Utr[i_val], a)
        if fm.n_params == 0:
            th = np.zeros(0); tr = dict(kta_init=np.nan, kta_final=np.nan, trajectory=[], n_evals=0)
        elif train_theta:
            tr = train_kta(fm, Xfit, ytr[i_fit], rng, cfg["feature_map"]["kta_train_subsample"],
                           cfg["feature_map"]["kta_maxiter"], cfg["feature_map"]["theta_init"],
                           cfg["feature_map"]["kta_restarts"], rank_lambda=rank_lambda)
            th = tr["theta"]
        else:
            th = fm.init_params(rng, cfg["feature_map"]["theta_init"]) if theta_init is None else theta_init
            tr = dict(kta_init=np.nan, kta_final=np.nan, trajectory=[], n_evals=0)
        S_fit, S_val = fm.states(Xfit, th), fm.states(Xval, th)
        K_ff, K_vf = fidelity_kernel(S_fit, S_fit), fidelity_kernel(S_val, S_fit)
        C, acc = select_C(K_ff, ytr[i_fit], K_vf, ytr[i_val], cfg["svm"]["C_grid"])
        cand = dict(a=a, theta=th, C=C, val_acc=acc, kta_fit=kta(K_ff, ytr[i_fit]), **{k: tr[k] for k in ("kta_init", "kta_final", "n_evals")},
                    trajectory=tr["trajectory"])
        if best is None or acc > best["val_acc"] or (acc == best["val_acc"] and cand["kta_fit"] > best["kta_fit"]):
            best = cand
    best["fmap"] = fm
    return best


def quantum_states(P: dict, fm, theta, a):
    Xtr = Preprocessor.to_angles(P["U_train"], a); Xte = Preprocessor.to_angles(P["U_test"], a)
    return fm.states(Xtr, theta), fm.states(Xte, theta)


def env_info(cfg: dict) -> dict:
    import numpy, scipy, sklearn
    return dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, platform=platform.platform(), machine=platform.machine(),
                config_hash=hash(json.dumps(cfg, sort_keys=True, default=str)) & 0xffffffff)
