"""
Synthetic reproducible benchmark for the Threshold-Honest Audit (THA) protocol.

THA-Synth generates synthetic candidate-job pairs with controllable ground-truth
and historical-decision disparity, then runs the operating-point checks used in
Threshold-Honest Audit:

    R1: matched-rate paired cluster-bootstrap         (run_audit / fast_paired_delta)
    R2: threshold sweep + per-requisition top-k       (run_audit)
    R3: two-way intersectional cell DI                (intersectional_cell_di)
    R4: null-model sanity check                       (null_model_audit)
    R5: demographic-inference noise sensitivity       (noise_sensitivity_sweep)

Important scope note: the synthetic benchmark intentionally uses a simplified
binary protected attribute and pooled DI. The production case study in the paper
uses a richer three-layer aggregation over requisitions and occupational
categories; THA-Synth is a lightweight stress test for operating-point artefacts,
not a replica of any employer's data.

No external data is required. Dependencies: numpy and pandas only.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenConfig:
    """Configuration for the synthetic candidate-job generator."""

    n_jobs: int = 2_500
    candidates_per_job_lambda: float = 22.0
    base_hire_rate: float = 0.076
    minority_share: float = 0.32
    qual_majority_advantage: float = 0.30
    historical_bias: float = 0.55
    model_bias_residual: float = 0.30
    model_noise_sigma: float = 1.20
    # "signal" means pred is based on qualification plus residual group bias.
    # "random" means pred is independent of qualification, group and history.
    model_mode: str = "signal"
    age_band_shares: tuple[float, float, float, float] = (0.18, 0.42, 0.28, 0.12)
    age_youngest_disadvantage: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class AuditConfig:
    """Configuration for audit execution."""

    B: int = 50
    seed: int = 0
    top_k: int = 5
    ref_group: str = "Majority"


def generate(cfg: GenConfig) -> pd.DataFrame:
    """Generate synthetic candidate-job pairs.

    Returned columns: job_id, profile_id, group, age_band, pred, score.

    `score` is the historical binary decision. `pred` is the model score.
    In `model_mode="random"`, `pred` is U[0,1] independent noise. This is a
    strict null regime: the model contains no qualification information.
    """
    if cfg.model_mode not in {"signal", "random"}:
        raise ValueError("model_mode must be 'signal' or 'random'")

    rng = np.random.default_rng(cfg.seed)
    sizes = rng.poisson(cfg.candidates_per_job_lambda, size=cfg.n_jobs).clip(min=1)
    n = int(sizes.sum())

    job_numeric = np.repeat(np.arange(cfg.n_jobs), sizes)
    job_id = np.array([f"syn:{j}" for j in job_numeric], dtype=object)
    profile_id = np.array([f"prof:{p}" for p in range(n)], dtype=object)

    is_minority = rng.random(n) < cfg.minority_share
    group = np.where(is_minority, "Minority", "Majority")

    qual = rng.normal(0.0, 1.0, size=n)
    qual += np.where(is_minority, 0.0, cfg.qual_majority_advantage)

    h_logit = qual + np.where(is_minority, 0.0, cfg.historical_bias)
    intercept = _calibrate_intercept(h_logit, cfg.base_hire_rate)
    h_prob = _sigmoid(h_logit + intercept)
    score = (rng.random(n) < h_prob).astype(np.int8)

    if cfg.model_mode == "random":
        # Strict null model: independent of qualification, group and score.
        pred = rng.random(n)
    else:
        m_logit = qual + np.where(
            is_minority, 0.0, cfg.historical_bias * cfg.model_bias_residual
        )
        m_logit += rng.normal(0.0, cfg.model_noise_sigma, size=n)
        pred = _sigmoid(m_logit)

    # Separate RNG stream so age labels do not perturb the qualification/model RNG.
    rng_age = np.random.default_rng(cfg.seed + 1)
    age_labels = np.array(["18-25", "26-35", "36-45", "46+"])
    age_idx = rng_age.choice(len(age_labels), size=n, p=np.asarray(cfg.age_band_shares))
    age_band = age_labels[age_idx]

    if cfg.age_youngest_disadvantage > 0:
        is_youngest = age_idx == 0
        h_logit2 = h_logit + np.where(is_youngest, -cfg.age_youngest_disadvantage, 0.0)
        intercept2 = _calibrate_intercept(h_logit2, cfg.base_hire_rate)
        h_prob2 = _sigmoid(h_logit2 + intercept2)
        score = (rng_age.random(n) < h_prob2).astype(np.int8)

    return pd.DataFrame(
        {
            "job_id": job_id,
            "profile_id": profile_id,
            "group": group,
            "age_band": age_band,
            "pred": pred.astype(np.float32),
            "score": score.astype(np.int8),
        }
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logit: np.ndarray, target: float, tol: float = 1e-3) -> float:
    """Bisection: find intercept so mean(sigmoid(logit + b)) ~= target."""
    lo, hi = -10.0, 10.0
    mid = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(_sigmoid(logit + mid)))
        if rate < target:
            lo = mid
        else:
            hi = mid
        if abs(rate - target) < tol:
            break
    return mid


# ---------------------------------------------------------------------------
# Audit metrics and selectors
# ---------------------------------------------------------------------------


def pooled_disparate_impact(selected: np.ndarray, group: np.ndarray, ref: str) -> float:
    """Pooled DI = min group selection rate / max group selection rate."""
    return weighted_pooled_disparate_impact(selected, group, np.ones(len(selected)), ref)


# Backward-compatible alias used by earlier scripts.
disparate_impact = pooled_disparate_impact


def weighted_pooled_disparate_impact(
    selected: np.ndarray, group: np.ndarray, weights: np.ndarray, ref: str
) -> float:
    """Weighted pooled DI for a binary ref-vs-other grouping.

    This wrapper accepts string labels. The fast bootstrap path converts labels
    to 0/1 codes once and then calls `_weighted_di_codes` for speed.
    """
    group_code = (np.asarray(group) == ref).astype(np.int8)
    return _weighted_di_codes(np.asarray(selected, dtype=float), group_code, np.asarray(weights, dtype=float))


def _weighted_di_codes(selected: np.ndarray, group_code: np.ndarray, weights: np.ndarray) -> float:
    """Fast weighted pooled DI using group_code 1=reference, 0=other."""
    totals = np.bincount(group_code, weights=weights, minlength=2)
    selected_totals = np.bincount(group_code, weights=weights * selected, minlength=2)
    if totals[0] == 0 or totals[1] == 0:
        return 1.0
    rates = selected_totals / totals
    denom = max(float(rates[0]), float(rates[1]))
    if denom == 0:
        return 1.0
    return min(float(rates[0]), float(rates[1])) / denom


def select_global_threshold(pred: np.ndarray, target_rate: float) -> np.ndarray:
    if not 0 < target_rate < 1:
        raise ValueError("target_rate must be in (0, 1)")
    cutoff = np.quantile(pred, 1.0 - target_rate)
    return (pred >= cutoff).astype(np.int8)


def select_global_threshold_weighted(
    pred: np.ndarray, target_rate: float, weights: np.ndarray, order_desc: np.ndarray | None = None
) -> np.ndarray:
    """Select the weighted top target_rate fraction by score."""
    if not 0 < target_rate < 1:
        raise ValueError("target_rate must be in (0, 1)")
    weights = np.asarray(weights, dtype=float)
    if order_desc is None:
        order_desc = np.argsort(-pred)
    w_sorted = weights[order_desc]
    cum = np.cumsum(w_sorted)
    target_count = target_rate * weights.sum()
    cutoff_pos = int(np.searchsorted(cum, target_count, side="left"))
    cutoff_pos = min(max(cutoff_pos, 0), len(order_desc) - 1)
    cutoff = pred[order_desc[cutoff_pos]]
    return (pred >= cutoff).astype(np.int8)


def select_topk_per_job(df: pd.DataFrame, k: int, job_col: str = "job_id") -> np.ndarray:
    """For each requisition, mark top-k candidates by `pred` as selected.

    In the fast bootstrap, duplicated requisitions are represented by cluster
    weights; selecting top-k once and weighting the selected rows is equivalent
    to creating a fresh copy of the requisition and selecting top-k in each copy.
    """
    n = len(df)
    if k <= 0:
        return np.zeros(n, dtype=np.int8)

    sel = np.zeros(n, dtype=np.int8)
    pred = df["pred"].to_numpy()
    for indices in df.groupby(job_col, sort=False).indices.values():
        idx = np.fromiter(indices, dtype=np.int64)
        if len(idx) <= k:
            sel[idx] = 1
        else:
            top_local = idx[np.argpartition(-pred[idx], kth=k - 1)[:k]]
            sel[top_local] = 1
    return sel


def select_pred05(pred: np.ndarray) -> np.ndarray:
    return (pred > 0.5).astype(np.int8)


# ---------------------------------------------------------------------------
# Fast paired cluster bootstrap
# ---------------------------------------------------------------------------


def _cluster_codes(df: pd.DataFrame, cluster_col: str = "job_id") -> tuple[np.ndarray, int]:
    codes, _ = pd.factorize(df[cluster_col], sort=False)
    return codes.astype(np.int64), int(codes.max() + 1)


def _di_from_counts(sel_ref: float, tot_ref: float, sel_oth: float, tot_oth: float) -> float:
    if tot_ref <= 0 or tot_oth <= 0:
        return 1.0
    rate_ref = sel_ref / tot_ref
    rate_oth = sel_oth / tot_oth
    denom = max(rate_ref, rate_oth)
    if denom == 0:
        return 1.0
    return min(rate_ref, rate_oth) / denom


def fast_paired_delta(
    df: pd.DataFrame,
    operating_point: str,
    ref: str = "Majority",
    B: int = 500,
    seed: int = 0,
    top_k: int = 5,
    cluster_col: str = "job_id",
) -> dict[str, float]:
    """Fast paired cluster bootstrap for THA-Synth operating points.

    Supported operating_point values: "matched", "topk", "pred05".
    The matched threshold is recomputed inside each bootstrap replicate using
    that replicate's weighted historical selection rate.

    Implementation detail: duplicate bootstrap requisitions are represented by
    cluster weights. For per-requisition top-k, this is equivalent to creating a
    fresh copy of each sampled requisition and applying top-k inside each copy;
    it avoids the bug where duplicate copies would be merged under the original
    job_id.
    """
    if operating_point not in {"matched", "topk", "pred05"}:
        raise ValueError("operating_point must be one of: matched, topk, pred05")
    if B <= 0:
        raise ValueError("B must be positive")

    rng = np.random.default_rng(seed)
    group = df["group"].to_numpy()
    pred = df["pred"].to_numpy()
    hist = df["score"].to_numpy(dtype=np.float64)
    cluster, n_clusters = _cluster_codes(df, cluster_col)

    ref_row = (group == ref).astype(np.float64)
    oth_row = 1.0 - ref_row

    # Per-cluster totals let each bootstrap replicate be evaluated with a few
    # length-n_clusters dot products rather than materialising a resampled DataFrame.
    tot_ref_c = np.bincount(cluster, weights=ref_row, minlength=n_clusters)
    tot_oth_c = np.bincount(cluster, weights=oth_row, minlength=n_clusters)
    hist_ref_c = np.bincount(cluster, weights=hist * ref_row, minlength=n_clusters)
    hist_oth_c = np.bincount(cluster, weights=hist * oth_row, minlength=n_clusters)

    # Full-sample historical DI.
    tot_ref = float(tot_ref_c.sum())
    tot_oth = float(tot_oth_c.sum())
    hist_ref = float(hist_ref_c.sum())
    hist_oth = float(hist_oth_c.sum())
    di_hist_full = _di_from_counts(hist_ref, tot_ref, hist_oth, tot_oth)

    # Full-sample model DI and fixed per-cluster selected counts where possible.
    model_ref_c = model_oth_c = None
    if operating_point == "matched":
        target_full = float(hist.mean())
        selected_full = select_global_threshold(pred, target_full).astype(np.float64)
        model_ref = float(np.dot(selected_full, ref_row))
        model_oth = float(np.dot(selected_full, oth_row))
        # Pre-sort once for weighted threshold selection in bootstrap replicates.
        order_desc = np.argsort(-pred)
        cluster_sorted = cluster[order_desc]
        ref_sorted = ref_row[order_desc]
        oth_sorted = oth_row[order_desc]
    elif operating_point == "topk":
        selected_full = select_topk_per_job(df, k=top_k, job_col=cluster_col).astype(np.float64)
        model_ref_c = np.bincount(cluster, weights=selected_full * ref_row, minlength=n_clusters)
        model_oth_c = np.bincount(cluster, weights=selected_full * oth_row, minlength=n_clusters)
        model_ref = float(model_ref_c.sum())
        model_oth = float(model_oth_c.sum())
        order_desc = cluster_sorted = ref_sorted = oth_sorted = None
    else:
        selected_full = select_pred05(pred).astype(np.float64)
        model_ref_c = np.bincount(cluster, weights=selected_full * ref_row, minlength=n_clusters)
        model_oth_c = np.bincount(cluster, weights=selected_full * oth_row, minlength=n_clusters)
        model_ref = float(model_ref_c.sum())
        model_oth = float(model_oth_c.sum())
        order_desc = cluster_sorted = ref_sorted = oth_sorted = None

    di_model_full = _di_from_counts(model_ref, tot_ref, model_oth, tot_oth)
    delta_full = di_model_full - di_hist_full

    deltas = np.empty(B, dtype=float)
    for b in range(B):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        cw = np.bincount(sampled, minlength=n_clusters).astype(np.float64)

        b_tot_ref = float(np.dot(cw, tot_ref_c))
        b_tot_oth = float(np.dot(cw, tot_oth_c))
        b_hist_ref = float(np.dot(cw, hist_ref_c))
        b_hist_oth = float(np.dot(cw, hist_oth_c))
        di_hist = _di_from_counts(b_hist_ref, b_tot_ref, b_hist_oth, b_tot_oth)

        if operating_point == "matched":
            total_weight = b_tot_ref + b_tot_oth
            hist_selected_weight = b_hist_ref + b_hist_oth
            target_b = hist_selected_weight / total_weight if total_weight else 0.0
            # Weighted top target_b fraction among predictions. Continuous pred
            # makes ties negligible in this synthetic setting.
            sorted_weights = cw[cluster_sorted]
            cum_total = np.cumsum(sorted_weights)
            target_count = target_b * total_weight
            pos = int(np.searchsorted(cum_total, target_count, side="left"))
            pos = min(max(pos, 0), len(sorted_weights) - 1)
            cum_ref = np.cumsum(sorted_weights * ref_sorted)
            cum_oth = np.cumsum(sorted_weights * oth_sorted)
            b_model_ref = float(cum_ref[pos])
            b_model_oth = float(cum_oth[pos])
        else:
            b_model_ref = float(np.dot(cw, model_ref_c))
            b_model_oth = float(np.dot(cw, model_oth_c))

        di_model = _di_from_counts(b_model_ref, b_tot_ref, b_model_oth, b_tot_oth)
        deltas[b] = di_model - di_hist

    return {
        "delta_pt": float(delta_full),
        "delta_lo": float(np.quantile(deltas, 0.025)),
        "delta_hi": float(np.quantile(deltas, 0.975)),
        "p_pos": float(np.mean(deltas > 0)),
        "di_A": float(di_hist_full),
        "di_B": float(di_model_full),
        "B": int(B),
    }


# Optional generic wrapper for small custom experiments.
Selector = Callable[[pd.DataFrame], np.ndarray]


def paired_bootstrap_di_delta(
    df: pd.DataFrame,
    selector_a: Selector,
    selector_b: Selector,
    ref: str = "Majority",
    B: int = 100,
    seed: int = 0,
    cluster_col: str = "job_id",
) -> dict[str, float]:
    """Generic, slower paired cluster bootstrap for custom selectors.

    For built-in THA-Synth operating points, prefer `fast_paired_delta`.
    This generic version correctly gives duplicated clusters fresh job IDs, so
    top-k selectors do not merge repeated bootstrap copies.
    """
    rng = np.random.default_rng(seed)
    jobs = df[cluster_col].unique()
    job_to_idx = {j: np.asarray(g, dtype=np.int64) for j, g in df.groupby(cluster_col).indices.items()}

    sel_a_full = selector_a(df)
    sel_b_full = selector_b(df)
    di_a_full = pooled_disparate_impact(sel_a_full, df["group"].values, ref)
    di_b_full = pooled_disparate_impact(sel_b_full, df["group"].values, ref)
    deltas = np.empty(B, dtype=np.float64)

    for b in range(B):
        boot_jobs = rng.choice(jobs, size=len(jobs), replace=True)
        pieces = []
        for copy_id, j in enumerate(boot_jobs):
            tmp = df.iloc[job_to_idx[j]].copy()
            tmp[cluster_col] = f"{j}__boot{copy_id}"
            pieces.append(tmp)
        sub = pd.concat(pieces, ignore_index=True)
        sel_a = selector_a(sub)
        sel_b = selector_b(sub)
        deltas[b] = pooled_disparate_impact(sel_b, sub["group"].values, ref) - pooled_disparate_impact(
            sel_a, sub["group"].values, ref
        )

    return {
        "delta_pt": float(di_b_full - di_a_full),
        "delta_lo": float(np.quantile(deltas, 0.025)),
        "delta_hi": float(np.quantile(deltas, 0.975)),
        "p_pos": float(np.mean(deltas > 0)),
        "di_A": float(di_a_full),
        "di_B": float(di_b_full),
        "B": int(B),
    }


# ---------------------------------------------------------------------------
# R3: two-way intersectional cell DI
# ---------------------------------------------------------------------------


def intersectional_cell_di(
    df: pd.DataFrame,
    selector: Selector,
    axis_a: str = "group",
    axis_b: str = "age_band",
    ref_a: str = "Majority",
    ref_b: str = "26-35",
) -> pd.DataFrame:
    """Two-way cell DI relative to the (ref_a, ref_b) reference cell."""
    sel = selector(df)
    a = df[axis_a].values
    b = df[axis_b].values
    ref_mask = (a == ref_a) & (b == ref_b)
    rate_ref = float(sel[ref_mask].mean()) if ref_mask.any() else 0.0

    rows = []
    for aval in sorted(df[axis_a].unique()):
        for bval in sorted(df[axis_b].unique()):
            mask = (a == aval) & (b == bval)
            if not mask.any():
                continue
            rate = float(sel[mask].mean())
            rows.append(
                {
                    "cell_a": aval,
                    "cell_b": bval,
                    "n": int(mask.sum()),
                    "sel_rate": round(rate, 4),
                    "di_vs_ref": round(rate / rate_ref, 3) if rate_ref > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R4: null-model sanity check
# ---------------------------------------------------------------------------


def null_model_audit(df: pd.DataFrame, ref: str = "Majority", seed: int = 0) -> pd.DataFrame:
    """Replace `pred` with null scorers and rerun the pred>0.5 audit."""
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
        sel = select_pred05(p)
        rows.append(
            {
                "scorer": name,
                "sel_rate": round(float(sel.mean()), 3),
                "pooled_di": round(pooled_disparate_impact(sel, df["group"].values, ref), 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R5: demographic-inference noise sensitivity
# ---------------------------------------------------------------------------


def noise_sensitivity_sweep(
    df: pd.DataFrame,
    eta_grid: Iterable[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
    ref: str = "Majority",
    B: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Inject independent group-label flips and rerun matched bootstrap."""
    rng = np.random.default_rng(seed)
    other = {"Majority": "Minority", "Minority": "Majority"}
    rows = []

    for eta in eta_grid:
        df_noisy = df.copy()
        if eta > 0:
            flip_mask = rng.random(len(df_noisy)) < eta
            grp = df_noisy["group"].to_numpy(copy=True)
            for i in np.where(flip_mask)[0]:
                grp[i] = other.get(grp[i], grp[i])
            df_noisy["group"] = grp
        out = fast_paired_delta(
            df_noisy,
            "matched",
            ref=ref,
            B=B,
            seed=int(rng.integers(0, 1 << 31)),
        )
        rows.append(
            {
                "eta": eta,
                "delta": round(out["delta_pt"], 3),
                "ci_lo": round(out["delta_lo"], 3),
                "ci_hi": round(out["delta_hi"], 3),
                "p_pos": round(out["p_pos"], 3),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# End-to-end audit helpers
# ---------------------------------------------------------------------------


def _selectors_for(df: pd.DataFrame, top_k: int) -> tuple[Selector, Selector, Selector, Selector]:
    target = float(df["score"].mean())
    sel_hist = lambda d: d["score"].to_numpy(dtype=np.int8)
    sel_match = lambda d: select_global_threshold(d["pred"].values, target)
    sel_topk = lambda d: select_topk_per_job(d, k=top_k)
    sel_p05 = lambda d: select_pred05(d["pred"].values)
    return sel_hist, sel_match, sel_topk, sel_p05


def run_audit(cfg: GenConfig | None = None, audit: AuditConfig | None = None) -> pd.DataFrame:
    """Run matched, top-k and pred>0.5 audits vs historical selection."""
    cfg = cfg or GenConfig()
    audit = audit or AuditConfig()
    df = generate(cfg)
    rows = []
    for label, op in [("Matched", "matched"), (f"TopK={audit.top_k}", "topk"), ("pred>0.5", "pred05")]:
        out = fast_paired_delta(
            df,
            op,
            ref=audit.ref_group,
            B=audit.B,
            seed=audit.seed,
            top_k=audit.top_k,
        )
        rows.append(
            {
                "operating_point": label,
                "di_hist": round(out["di_A"], 3),
                "di_model": round(out["di_B"], 3),
                "delta": round(out["delta_pt"], 3),
                "ci_lo": round(out["delta_lo"], 3),
                "ci_hi": round(out["delta_hi"], 3),
                "p_pos": round(out["p_pos"], 3),
                "B": audit.B,
            }
        )
    return pd.DataFrame(rows)


def run_table8(B: int = 50, seed: int = 0) -> pd.DataFrame:
    """Run the three Table-8-style regimes and return one tidy table."""
    audit = AuditConfig(B=B, seed=seed)
    regimes = {
        "Realistic": GenConfig(seed=seed, model_bias_residual=0.30, model_noise_sigma=1.20, model_mode="signal"),
        "Debiased": GenConfig(seed=seed, model_bias_residual=0.00, model_noise_sigma=1.20, model_mode="signal"),
        "Random": GenConfig(seed=seed, model_mode="random"),
    }
    frames = []
    for name, cfg in regimes.items():
        table = run_audit(cfg, audit)
        table.insert(0, "regime", name)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def demo(B: int = 20, seed: int = 0) -> None:
    """Fast demo covering R1-R5. Suitable for artifact smoke checks."""
    start = time.time()
    cfg = GenConfig(seed=seed)
    df = generate(cfg)

    print("=" * 76)
    print(f"R1+R2: Table-8-style audit, quick mode B={B}")
    print("=" * 76)
    print(run_table8(B=B, seed=seed).to_string(index=False))

    print("\n" + "=" * 76)
    print("R3: group x age-band cell DI at matched rate")
    print("=" * 76)
    _, sel_match, _, _ = _selectors_for(df, top_k=5)
    print(intersectional_cell_di(df, sel_match).to_string(index=False))

    print("\n" + "=" * 76)
    print("R4: null-model sanity check at pred>0.5")
    print("=" * 76)
    print(null_model_audit(df, seed=seed).to_string(index=False))

    print("\n" + "=" * 76)
    print(f"R5: label-flip sensitivity, quick mode B={max(3, B // 2)}")
    print("=" * 76)
    print(noise_sensitivity_sweep(df, B=max(3, B // 2), seed=seed).to_string(index=False))
    print(f"\nCompleted in {time.time() - start:.1f}s")


def save_table8_csv(path: str | Path, B: int = 500, seed: int = 0) -> pd.DataFrame:
    """Run Table-8-style regimes and write CSV."""
    table = run_table8(B=B, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def save_table8_pickle(path: str | Path, B: int = 500, seed: int = 0) -> pd.DataFrame:
    table = run_table8(B=B, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(table, f)
    return table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run THA-Synth synthetic benchmark.")
    parser.add_argument("--table8", action="store_true", help="Run Table-8-style regimes only.")
    parser.add_argument("--B", type=int, default=20, help="Bootstrap replicates. Use 500 for full paper-style run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path for --table8.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.table8:
        tbl = run_table8(B=args.B, seed=args.seed)
        print(tbl.to_string(index=False))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            tbl.to_csv(args.out, index=False)
            print(f"\nWrote {args.out}")
    else:
        # Safe default: quick demo. Full B=500 reproduction is explicit.
        demo(B=args.B, seed=args.seed)
