# THA-Synth

Reference implementation of the **Threshold-Honest Audit (THA)** protocol's
synthetic benchmark, accompanying the NeurIPS 2026 Evaluations & Datasets
Track submission *"Threshold-Honest Auditing of Production Hiring Models"*
(under double-blind review).

THA-Synth is a self-contained synthetic generator that produces
candidate–job pairs with **controllable ground-truth disparity** and
implements all five Threshold-Honest Audit components (R1–R5)
described in the paper:

| Component | Description                                              | Function                       |
|-----------|----------------------------------------------------------|--------------------------------|
| R1        | Matched-rate paired cluster-bootstrap                    | `paired_bootstrap_di_delta`    |
| R2        | Threshold sweep + per-requisition top-*k*                | `run_audit`                    |
| R3        | Two-way intersectional cell DI (group × age band)        | `intersectional_cell_di`       |
| R4        | Null-model sanity check (uniform & shuffled-pred)        | `null_model_audit`             |
| R5        | Demographic-inference noise sensitivity sweep            | `noise_sensitivity_sweep`      |

It is intended as a **reusable test bed** for evaluation-protocol research,
sitting in the same artifact family as recent NeurIPS D&B fairness benchmarks
(BAF 2022, FairJob 2024) but with a different target: stress-testing of
audit protocols rather than mitigation algorithms.

## Quick start

Requires Python 3.9+ and only `numpy` + `pandas`.

```bash
pip install numpy pandas
python3 synthetic_benchmark.py
```

CPU-only. Reproduces Table 8 of the paper (Realistic / Debiased / Random
regimes, B = 500 paired cluster-bootstrap resamples) under one minute on a
standard laptop. The same script also demonstrates R3 (intersectional cell
DI), R4 (null-model sanity check at `pred > 0.5`), and R5 (η-sweep over
label-flip noise on the protected attribute).

## What the script does

The generator parameterises:

- **α** — ground-truth qualification advantage of the majority group
  (latent qualification `q ∼ N(0, 1) + α · 1[majority]`)
- **β** — additional historical-decision bias on top of the qualification gap
  (decision logit `ℓ_h = q + β · 1[majority]`, with intercept calibrated
  by bisection so the empirical hire rate matches a user-specified target)
- **γ ∈ [0, 1]** — fraction of the historical bias retained by the model
  after debiasing (model logit `ℓ_m = q + γβ · 1[majority] + N(0, σ²)`)
- **σ** — noise on the model logits

Three regimes are reported by default:

| Regime    | Description                              | γ   | σ    |
|-----------|------------------------------------------|-----|------|
| Realistic | model retains 30% of historical bias     | 0.3 | 1.2  |
| Debiased  | model fully removes historical bias      | 0.0 | 1.2  |
| Random    | model is pure noise (no qualification info) | n/a | 10.0 |

Across all three regimes, the inflation ordering
`Match Δ < Top-5 Δ < pred > 0.5 Δ` holds. Under the **Random** regime
(model contains zero qualification information) `pred > 0.5` still
produces an apparent paired Δ ≈ +0.49 — a model with no signal at all
appears to be roughly half a DI unit "fairer" than the historical
baseline if reported at the conventional cutoff.

## Default configuration

Matches the case-study scale of the paper:

- ~54,000 candidate–job pairs across 2,500 requisitions
- Poisson-distributed cluster sizes (mean 22 candidates per requisition)
- ~7.6% base hire rate (calibrated by bisection to within 1e-3)
- Minority share ≈ 32%

The base-rate calibration is exact; the cluster-size distribution is
intentionally Poisson rather than empirical, since the case-study
cluster-size distribution is itself an audit-sensitive quantity.

## Files

| File                       | Description                                    |
|----------------------------|------------------------------------------------|
| `synthetic_benchmark.py`   | Generator + audit pipeline (~200 lines)        |
| `synthetic_results.pkl`    | Saved B = 500 results from the default seed    |
| `README.md`                | This file                                      |
| `LICENSE`                  | MIT licence                                    |

## Intended use

Researchers proposing new fairness-audit metrics or alternative reporting
protocols can use THA-Synth as a reference setting to:

1. **Verify that a proposed metric does not inflate under the Random regime.**
   A metric that reports large positive Δ under Random is flagging
   threshold artefact, not model behaviour.
2. **Verify that a proposed metric distinguishes Realistic from Debiased.**
   A metric that reports the same value in both regimes is insensitive to
   the actual debiasing effect.
3. **Sweep (α, β, γ, σ) to characterise the regime in which a protocol is reliable.**
   All four parameters are exposed via the dataclass `GenConfig` at the top
   of the script.

## Limitations

THA-Synth is **not** a model of any specific employer's hiring data. It
controls for the threshold-induced inflation phenomenon but does not
reproduce real production phenomena that the case study captures:

- missingness patterns in protected-attribute inference,
- requisition-specific recruiter heterogeneity,
- correlated cluster structure across recruiters,
- non-Poisson cluster-size distributions.

Conclusions reached on THA-Synth alone should not be generalised to real
production audits without empirical validation. THA-Synth is a **complement**
to case studies, not a substitute.

## Licence

Released under the MIT licence (see `LICENSE`).

## Citation

The paper is currently under double-blind review at NeurIPS 2026. Citation
information will be updated upon acceptance.
