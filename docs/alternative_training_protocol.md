# Pre-registered protocol — alternative training objective (written before the experiment was run)

**Why this experiment.** The Associate Editor and Reviewer 1 noted that the conclusion "KTA training lowers test accuracy" rests on one objective (centred KTA) and one optimiser. The ablation showed that KTA maximisation halves the effective rank of the kernel; the natural alternative is an objective that rewards alignment while penalising rank collapse.

**Choice of alternative (Option B, regularised KTA).** Chosen over trajectory-based validation selection because it fits the existing `train_kta` code path (same optimiser, same fit-split-only data, same call signature) and because its regulariser targets the mechanism we observed (effective-rank collapse), so a null result is informative. Trajectory selection would additionally change *when* training stops, confounding objective and stopping rule.

## Objective
J_λ(θ) = KTA_c(K_θ(X_sub), y_sub) + λ · log(erank(K_θ(X_sub))) / log(n_sub),

where KTA_c is the centred kernel–target alignment already used, erank(K) = exp(H(p)) with p_i = λ_i(K)/Σλ_i(K) the entropy effective rank (the same statistic reported as `eff_rank` throughout the paper), and n_sub = 256 so that the regulariser lies in [0, 1]. λ = 0 recovers plain KTA.

## Data
- Training set for the objective: the same n_sub = 256 points drawn from the **fit split** (80 % of the 600-point training sample) with the seed-controlled RNG used by `train_kta`.
- Validation set: the 20 % validation split of the training sample — used **only** to choose λ and C.
- Test set: the official test split, touched once per (task, seed) after all choices are fixed.

## Optimisation
- Optimiser: L-BFGS-B, finite-difference gradients (eps = 1e-4), `maxiter` = 60, one restart, θ₀ ~ U(0, 2π) — identical to `train_kta` so that only the objective changes.
- Parameter bounds: none (angles are periodic).
- Stopping: L-BFGS-B default tolerances or 60 iterations.

## Grid and selection rule
- λ ∈ {0.1, 0.3, 1.0}. For each λ the fiducial state is trained on the fit split; C ∈ {0.01, 0.1, 1, 10, 100} is chosen on the validation split; the λ with the highest validation accuracy is kept (ties → larger λ, i.e. more regularisation).
- A second variant, **validation-selected training** (Option C), chooses among {untrained random θ, KTA-trained, regularised-KTA-trained} by validation accuracy (ties → untrained, the cheapest). This is pure model selection and uses no test information.

## Design
- Tasks: the three ablation tasks (fashion_0v1, fashion_2v4, mnist_3v8); seeds 0–9; a = π fixed (as in the ablation); map = covariant_torus, L = 2, entangled.
- Reported per (task, seed, variant): test accuracy (Wilson CI), balanced accuracy, F1, KTA (train), effective rank, energy in top-60 eigenvalues, off-diagonal mean, Nyström (m = 60, k-means centroids) accuracy and E_F, E₂, cond of the applied inverse, kernel distance to the KTA-trained kernel, selected λ, θ, number of objective evaluations, KTA before/after.
- Statistics: paired t-test and Wilcoxon across the 10 seeds per task, Cohen d_z, bootstrap CI; Holm correction over the ablation family (same policy as the main paper).

## Decision rules (fixed now)
- If the regularised objective is **not** better than the untrained kernel on the majority of tasks (paired mean difference ≤ 0), the manuscript keeps a generalised statement: "both alignment-based objectives lowered accuracy relative to the untrained kernel", bounded by the p-values.
- If the regularised objective **matches** the untrained kernel (|Δ| < 0.5 pt and not significant) while plain KTA is lower, the claim is narrowed to "plain KTA maximisation" everywhere it appears, including the Abstract.
- If the regularised objective is **better** than the untrained kernel on the majority of tasks, this becomes a central result and the Abstract, Contributions and Discussion are rewritten accordingly.
- The validation-selected variant is reported descriptively (what a practitioner would obtain); it is not used to pick the narrative.
