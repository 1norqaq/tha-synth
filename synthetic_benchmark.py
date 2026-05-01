"""
Synthetic reproducible benchmark for the threshold-sensitivity audit protocol.

Generates a synthetic candidate-job dataset with a controllable, ground-truth
disparity, scores it with a noisy model, and runs the four key audit operating
points (Hist, matched-rate, top-k, conventional pred>0.5).

Reproduces the qualitative phenomenon documented in the paper: apparent
disparate-impact "improvement" reported at conventional thresholds is largely
mechanical (driven by the gap between threshold and base rate), while the
matched-rate paired bootstrap reveals the underlying ground-truth gap.

No external data is required. Single file. Numpy + scipy + pandas only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------- data generation ----------


@dataclass
class GenConfig:
    n_jobs: int = 2_500
    candidates_per_job_lambda: float = 22.0  # mean candidates per requisition
    base_hire_rate: float = 0.076  # match observed deployment
    minority_share: float = 0.32
    # ground-truth qualification advantage for majority (log-odds)
    qual_majority_advantage: float = 0.30
    # historical decision-maker ADDITIONAL advantage for majority (log-odds)
    historical_bias: float = 0.55
    # model attenuates historical bias to this fraction (0 = perfectly debiased)
    model_bias_residual: float = 0.30
    # gaussian noise on model logits
    model_noise_sigma: float = 1.2
    seed: int = 0


def generate(cfg: GenConfig) -> pd.DataFrame:
    """Generate synthetic candidate-job pairs.

    Columns: index, job_id, profile_id, group ('Majority'/'Minority'),
    pred (model score), score (historical hire indicator).
    """
    rng = np.random.default_rng(cfg.seed)
    sizes = rng.poisson(cfg.candidates_per_job_lambda, size=cfg.n_jobs).clip(min=1)
    n = sizes.sum()

    job_id = np.repeat(np.arange(cfg.n_jobs), sizes)
    profile_id = np.arange(n)

    # Group membership
    is_minority = rng.random(n) < cfg.minority_share
    group = np.where(is_minority, "Minority", "Majority")

    # Latent qualification: gaussian + group offset
    qual = rng.normal(0.0, 1.0, size=n)
    qual += np.where(is_minority, 0.0, cfg.qual_majority_advantage)

    # Historical hire decision: logistic on qual + extra bias
    h_logit = qual + np.where(is_minority, 0.0, cfg.historical_bias)
    # Calibrate intercept to hit base hire rate
    target = cfg.base_hire_rate
    intercept = _calibrate_intercept(h_logit, target)
    h_p = 1.0 / (1.0 + np.exp(-(h_logit + intercept)))
    score = (rng.random(n) < h_p).astype(np.float32)

    # Model score: predicts qualification with attenuated bias + noise
    m_logit = qual + np.where(is_minority, 0.0, cfg.historical_bias * cfg.model_bias_residual)
    m_logit += rng.normal(0.0, cfg.model_noise_sigma, size=n)
    pred = 1.0 / (1.0 + np.exp(-m_logit))

    return pd.DataFrame(
        {
            "index": np.arange(n),
            "job_id": [f"syn:{j}" for j in job_id],
            "profile_id": [f"prof:{p}" for p in profile_id],
            "group": group,
            "pred": pred.astype(np.float32),
            "score": score,
        }
    )


def _calibrate_intercept(logit: np.ndarray, target: float, tol: float = 1e-3) -> float:
    """Bisection: find intercept so sigmoid(logit + b) has mean = target."""
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(logit + mid)))))
        if rate < target:
            lo = mid
        else:
            hi = mid
        if abs(rate - target) < tol:
            return mid
    return mid


# ---------- audit operating points ----------


def disparate_impact(selected: np.ndarray, group: np.ndarray, ref: str) -> float:
    """Equal DI: min selection rate / max selection rate (we use ref vs other)."""
    sel_ref = selected[group == ref].mean() if (group == ref).any() else 0.0
    sel_oth = selected[group != ref].mean() if (group != ref).any() else 0.0
    if max(sel_ref, sel_oth) == 0:
        return 1.0
    return min(sel_ref, sel_oth) / max(sel_ref, sel_oth)


def select_global_threshold(pred: np.ndarray, target_rate: float) -> np.ndarray:
    cutoff = np.quantile(pred, 1.0 - target_rate)
    return (pred >= cutoff).astype(np.int8)


def select_topk_per_job(df: pd.DataFrame, k: int) -> np.ndarray:
    """For each job, mark top-k by `pred` as selected.

    Robust to df being a bootstrapped slice (does not assume aligned index).
    """
    n = len(df)
    sel = np.zeros(n, dtype=np.int8)
    pred = df["pred"].to_numpy()
    job_id = df["job_id"].to_numpy()
    # Build groups by positional index in the local frame
    groups: dict[str, list[int]] = {}
    for i, j in enumerate(job_id):
        groups.setdefault(j, []).append(i)
    for idx_list in groups.values():
        idx = np.asarray(idx_list)
        if len(idx) <= k:
            sel[idx] = 1
            continue
        sub_pred = pred[idx]
        top = idx[np.argsort(-sub_pred)[:k]]
        sel[top] = 1
    return sel


def select_pred05(pred: np.ndarray) -> np.ndarray:
    return (pred > 0.5).astype(np.int8)


# ---------- paired cluster bootstrap ----------


def paired_bootstrap_di_delta(
    df: pd.DataFrame,
    selectorA,
    selectorB,
    ref: str = "Majority",
    B: int = 200,
    seed: int = 0,
) -> dict:
    """Paired cluster bootstrap on Delta DI = DI(B) - DI(A), clustering by job_id.

    selectorA and selectorB are callables df -> np.ndarray of {0,1} selections.
    """
    rng = np.random.default_rng(seed)
    jobs = df["job_id"].unique()
    job_to_idx = {j: np.array(g) for j, g in df.groupby("job_id").indices.items()}

    sel_A_full = selectorA(df)
    sel_B_full = selectorB(df)
    di_A_full = disparate_impact(sel_A_full, df["group"].values, ref)
    di_B_full = disparate_impact(sel_B_full, df["group"].values, ref)
    delta_full = di_B_full - di_A_full

    deltas = np.empty(B)
    for b in range(B):
        boot_jobs = rng.choice(jobs, size=len(jobs), replace=True)
        rows = np.concatenate([job_to_idx[j] for j in boot_jobs])
        sub = df.iloc[rows].copy()
        sA = selectorA(sub)
        sB = selectorB(sub)
        diA = disparate_impact(sA, sub["group"].values, ref)
        diB = disparate_impact(sB, sub["group"].values, ref)
        deltas[b] = diB - diA

    return dict(
        delta_pt=delta_full,
        delta_lo=float(np.quantile(deltas, 0.025)),
        delta_hi=float(np.quantile(deltas, 0.975)),
        p_pos=float(np.mean(deltas > 0)),
        di_A=di_A_full,
        di_B=di_B_full,
    )


# ---------- end-to-end demo ----------


def run_audit(cfg: GenConfig | None = None, B: int = 200) -> pd.DataFrame:
    cfg = cfg or GenConfig()
    df = generate(cfg)

    # Operating-point selectors
    target = float(df["score"].mean())  # matched to observed hire rate

    sel_hist = lambda d: d["score"].astype(np.int8).values
    sel_match = lambda d: select_global_threshold(d["pred"].values, target)
    sel_top5 = lambda d: select_topk_per_job(d, k=5)
    sel_p05 = lambda d: select_pred05(d["pred"].values)

    # Run paired bootstrap for each operating point vs Hist
    rows = []
    for name, selB in [("Matched", sel_match), ("TopK=5", sel_top5), ("pred>0.5", sel_p05)]:
        out = paired_bootstrap_di_delta(df, sel_hist, selB, B=B)
        rows.append(
            dict(
                operating_point=name,
                di_hist=round(out["di_A"], 3),
                di_model=round(out["di_B"], 3),
                delta=round(out["delta_pt"], 3),
                ci_lo=round(out["delta_lo"], 3),
                ci_hi=round(out["delta_hi"], 3),
                p_pos=round(out["p_pos"], 3),
            )
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    np.random.seed(0)
    print("=== Default config: ground-truth bias 0.55 logit, model attenuates to 30% ===")
    print(run_audit().to_string(index=False))

    print("\n=== Stress: model perfectly debiased (residual=0) ===")
    print(run_audit(GenConfig(model_bias_residual=0.0)).to_string(index=False))

    print("\n=== Null: model is uninformative noise (huge sigma) ===")
    print(run_audit(GenConfig(model_noise_sigma=10.0)).to_string(index=False))
