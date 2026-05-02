# THA-Synth

Reference implementation of the **Threshold-Honest Audit (THA)** protocol's
synthetic benchmark, accompanying the NeurIPS 2026 Evaluations & Datasets Track
submission *"Threshold-Honest Auditing of Production Hiring Models"*.

THA-Synth is a self-contained synthetic generator that produces candidate-job
pairs with controllable ground-truth disparity and runs the operating-point
checks used in Threshold-Honest Audit:

| Component | Description | Implementation |
|---|---|---|
| R1 | matched-rate paired cluster bootstrap | `fast_paired_delta`, `run_audit` |
| R2 | matched threshold, per-requisition top-k, `pred > 0.5` | `run_audit`, `run_table8` |
| R3 | two-way intersectional cell DI | `intersectional_cell_di` |
| R4 | null-model sanity check | `null_model_audit` |
| R5 | demographic-inference noise sensitivity | `noise_sensitivity_sweep` |

## Scope note

The production case study in the paper uses a richer audit pipeline, including
multiple protected attributes and a three-layer DI aggregation over requisitions
and occupational categories. THA-Synth intentionally uses a **simplified binary
protected attribute and pooled DI** so that evaluation-protocol artefacts can be
stress-tested in a lightweight, fully releasable setting. It reproduces the
operating-point logic and paired cluster-bootstrap structure, not any specific
employer's data distribution.

## Installation

Requires Python 3.9+.

```bash
python -m pip install -r requirements.txt
```

Dependencies are only `numpy` and `pandas`.

## Quick start

```bash
python quickstart.py
```

This runs a small smoke audit and should finish quickly on a CPU.

## Tests

```bash
python test_smoke.py
```

Expected output:

```text
Smoke tests passed.
```

## Reproduce the Table-8-style benchmark

Fast check:

```bash
python reproduce_table8.py --quick
```

Full B=500 run:

```bash
python reproduce_table8.py --B 500 --out table8_seed0_B500.csv
```

The checked-in file `table8_seed0_B500.csv` was generated with the full command
above. Small numerical differences across NumPy versions are possible, but the
qualitative ordering should remain stable:

```text
Matched Delta < TopK=5 Delta < pred>0.5 Delta
```

for Realistic, Debiased, and Random regimes.

## Regimes

- **Realistic**: model retains 30% of historical group bias.
- **Debiased**: model removes the historical group-bias offset.
- **Random**: model scores are sampled independently from `U[0, 1]`, independent
  of qualification, group, and historical labels.

## Repository contents

```text
README.md
ARTIFACT.md
LICENSE
requirements.txt
quickstart.py
reproduce_table8.py
test_smoke.py
synthetic_benchmark.py
synthetic_results.pkl
table8_seed0_B500.csv
```

## License

MIT License.
