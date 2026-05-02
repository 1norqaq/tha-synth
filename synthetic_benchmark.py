"""
Synthetic reproducible benchmark for the Threshold-Honest Audit (THA) protocol.

Generates a synthetic candidate-job dataset with a controllable, ground-truth
disparity, scores it with a noisy model, and runs all five THA components (R1-R5):

    R1: matched-rate paired cluster-bootstrap         (paired_bootstrap_di_delta)
    R2: threshold sweep + per-requisition top-k       (run_audit)
    R3: two-way intersectional cell DI                 (intersectional_cell_di)
    R4: null-model sanity check (uniform + shuffled)   (null_model_audit)
    R5: demographic-inference noise sensitivity        (noise_sensitivity_sweep)

Reproduces the qualitative phenomenon documented in the paper: apparent
disparate-impact "improvement" reported at conventional thresholds is largely
mechanical (driven by the gap between threshold and base rate), while the
matched-rate paired bootstrap reveals the underlying ground-truth gap.

No external data is required. Single file. Numpy + pandas only.
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
    # second axis for intersectional analysis (R3): 4 age bands with given shares
    age_band_shares: tuple = (0.18, 0.42, 0.28, 0.12)  # 18-25, 26-35, 36-45, 46+
    # additional disadvantage for the youngest band (log-odds, on top of group).
    # Default 0.0 so the age axis is purely a labelling axis for R3 cell DI;
    # set >0 to make age a substantive second protected attribute.
    age_youngest_disadvantage: float = 0.0
    seed: int = 0


def generate(cfg: GenConfig) -> pd.DataFrame:
    """Generate synthetic candidate-job pairs.

    Columns: index, job_id, profile_id, group ('Majority'/'Minority'),
    age_band ('18-25'/'26-35'/'36-45'/'46+'),
    pred (model score), score (historical hire indicator).
    """
    rng = np.random.default_rng(cfg.seed)
    sizes = rng.poisson(cfg.candidates_per_job_lambda, size=cfg.n_jobs).clip(min=1)
    n = sizes.sum()

    job_id = np.repeat(np.arange(cfg.n_jobs), sizes)
    profile_id = np.arange(n)

    # Group membership (primary protected axis)
    is_minority = rng.random(n) < cfg.minority_share
    group = np.where(is_minority, "Minority", "Majority")

    # Latent qualification: gaussian + group offset
    qual = rng.normal(0.0, 1.0, size=n)
    qual += np.where(is_minority, 0.0, cfg.qual_majority_advantage)

    # Historical hire decision: logistic on qual + group bias
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

    # Age band (second protected axis, for R3 intersectional analysis).
    # Drawn from a SEPARATE RNG stream so it does not perturb the qualification
    # / score / pred sequence above. This preserves Table 8 reproducibility.
    rng_age = np.random.default_rng(cfg.seed + 1)
    age_labels = np.array(["18-25", "26-35", "36-45", "46+"])
    age_idx = rng_age.choice(len(age_labels), size=n, p=np.asarray(cfg.age_band_shares))
    age_band = age_labels[age_idx]
    if cfg.age_youngest_disadvantage > 0:
        # Optional: re-roll score with age penalty applied
        is_youngest = age_idx == 0
        h_logit2 = h_logit + np.where(is_youngest, -cfg.age_youngest_disadvantage, 0.0)
        intercept2 = _calibrate_intercept(h_logit2, target)
        h_p2 = 1.0 / (1.0 + np.exp(-(h_logit2 + intercept2)))
        score = (rng_age.random(n) < h_p2).astype(np.float32)

    return pd.DataFrame(
        {
            "index": np.arange(n),
            "job_id": [f"syn:{j}" for j in job_id],
            "profile_id": [f"prof:{p}" for p in profile_id],
            "group": group,
            "age_band": age_band,
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
    B: int = 500,
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


# ---------- R3: two-way intersectional cell DI ----------


def intersectional_cell_di(
    df: pd.DataFrame,
    selector,
    axis_a: str = "group",
    axis_b: str = "age_band",
    ref_a: str = "Majority",
    ref_b: str = "26-35",
) -> pd.DataFrame:
    """Two-way cell DI relative to (ref_a, ref_b) reference cell.

    For each (axis_a, axis_b) cell, compute selection rate / reference cell rate.
    Returns a long-format DataFrame: cell_a, cell_b, n, sel_rate, di_vs_ref.
    """
    sel = selector(df)
    s_ref = sel[(df[axis_a].values == ref_a) & (df[axis_b].values == ref_b)]
    rate_ref = float(s_ref.mean()) if len(s_ref) else 0.0
    rows = []
    a_vals = sorted(df[axis_a].unique())
    b_vals = sorted(df[axis_b].unique())
    for a in a_vals:
        for b in b_vals:
            mask = (df[axis_a].values == a) & (df[axis_b].values == b)
            if not mask.any():
                continue
            sub = sel[mask]
            rate = float(sub.mean())
            di = (rate / rate_ref) if rate_ref > 0 else float("nan")
            rows.append(
                dict(
                    cell_a=a, cell_b=b, n=int(mask.sum()),
                    sel_rate=round(rate, 4),
                    di_vs_ref=round(di, 3),
                )
            )
    return pd.DataFrame(rows)


# ---------- R4: null-model sanity check ----------


def null_model_audit(
    df: pd.DataFrame, ref: str = "Majority", seed: int = 0
) -> pd.DataFrame:
    """Replace `pred` with two null scorers and re-run the pred>0.5 audit.

    Both null scorers carry no protected-attribute signal; any apparent fairness
    they produce at pred>0.5 is purely a consequence of permissive thresholding.

    Returns a DataFrame with one row per scorer (model, uniform, shuffled),
    reporting Equal DI at pred>0.5.
    """
    rng = np.random.default_rng(seed)
    pred_real = df["pred"].values
    pred_uniform = rng.random(len(df))
    pred_shuffled = rng.permutation(pred_real)

    rows = []
    for name, p in [
        ("Model PRED", pred_real),
        ("Uniform null", pred_uniform),
        ("Shuffled-pred null", pred_shuffled),
    ]:
        sel = (p > 0.5).astype(np.int8)
        di = disparate_impact(sel, df["group"].values, ref)
        sel_rate_total = float(sel.mean())
        rows.append(
            dict(
                scorer=name,
                sel_rate=round(sel_rate_total, 3),
                equal_di=round(di, 3),
            )
        )
    return pd.DataFrame(rows)


# ---------- R5: demographic-inference noise sensitivity ----------


def noise_sensitivity_sweep(
    df: pd.DataFrame,
    selector,
    eta_grid: tuple = (0.0, 0.05, 0.10, 0.20, 0.30),
    ref: str = "Majority",
    B: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Inject independent label-flip noise on the protected attribute and
    re-run the matched-rate paired bootstrap at each noise level.

    With probability eta, replace each candidate's group label with a draw
    from the other groups on that axis (here: flip Majority<->Minority).
    Returns a DataFrame: eta, mean_delta, ci_lo, ci_hi, p_pos.
    """
    rng = np.random.default_rng(seed)
    other = {"Majority": "Minority", "Minority": "Majority"}
    sel_hist = lambda d: d["score"].astype(np.int8).values
    rows = []
    for eta in eta_grid:
        df_noisy = df.copy()
        if eta > 0:
            n = len(df_noisy)
            flip_mask = rng.random(n) < eta
            grp = df_noisy["group"].values.copy()
            for i in np.where(flip_mask)[0]:
                grp[i] = other.get(grp[i], grp[i])
            df_noisy["group"] = grp
        out = paired_bootstrap_di_delta(df_noisy, sel_hist, selector, ref=ref, B=B,
                                        seed=int(rng.integers(0, 1 << 31)))
        rows.append(
            dict(
                eta=eta,
                mean_delta=round(out["delta_pt"], 3),
                ci_lo=round(out["delta_lo"], 3),
                ci_hi=round(out["delta_hi"], 3),
                p_pos=round(out["p_pos"], 3),
            )
        )
    return pd.DataFrame(rows)


# ---------- end-to-end demo ----------


def run_audit(cfg: GenConfig | None = None, B: int = 500) -> pd.DataFrame:
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
    cfg = GenConfig()
    df = generate(cfg)

    # ===== R1 + R2: matched-rate, top-k, pred>0.5 across three regimes =====
    print("=" * 72)
    print("R1+R2: matched-rate, per-requisition top-5, and pred>0.5")
    print("    paired cluster-bootstrap deltas (vs historical)")
    print("=" * 72)
    print("\n--- Realistic: model retains 30% of historical bias ---")
    print(run_audit().to_string(index=False))
    print("\n--- Debiased: model fully removes historical bias ---")
    print(run_audit(GenConfig(model_bias_residual=0.0)).to_string(index=False))
    print("\n--- Random: model is uninformative noise ---")
    print(run_audit(GenConfig(model_noise_sigma=10.0)).to_string(index=False))

    # ===== R3: two-way intersectional cell DI =====
    print("\n" + "=" * 72)
    print("R3: Group x Age-band cell DI (ref = Majority x 26-35)")
    print("=" * 72)
    target = float(df["score"].mean())
    sel_match = lambda d: select_global_threshold(d["pred"].values, target)
    inter = intersectional_cell_di(df, sel_match, ref_a="Majority", ref_b="26-35")
    print(inter.to_string(index=False))

    # ===== R4: null-model sanity check =====
    print("\n" + "=" * 72)
    print("R4: Null-model sanity check at pred>0.5")
    print("    (uniform-random and shuffled-pred scorers carry no signal)")
    print("=" * 72)
    print(null_model_audit(df).to_string(index=False))

    # ===== R5: demographic-inference noise sensitivity =====
    print("\n" + "=" * 72)
    print("R5: Noise sensitivity of matched-rate paired Delta")
    print("    (independent label flips with probability eta)")
    print("=" * 72)
    sel_match = lambda d: select_global_threshold(d["pred"].values, target)
    print(noise_sensitivity_sweep(df, sel_match, B=100).to_string(index=False))
