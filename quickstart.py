# Artifact instructions

This artifact accompanies the NeurIPS 2026 E&D submission *"Threshold-Honest
Auditing of Production Hiring Models"*.

## What is included

- Synthetic candidate-job generator with controllable historical and model bias.
- Fast paired cluster-bootstrap implementation for three operating points:
  matched-rate, per-requisition top-k, and `pred > 0.5`.
- Strict random-score regime where predictions are independent of qualification,
  group, and historical labels.
- Null-model and demographic-inference sensitivity demos.
- Smoke test and expected B=500 output.

## Environment

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.12, NumPy, and pandas.

## Commands for reviewers

### 1. Fast smoke run

```bash
python quickstart.py
```

Expected runtime: under 30 seconds on CPU.

### 2. Minimal tests

```bash
python tests/test_smoke.py
```

Expected output:

```text
Smoke tests passed.
```

### 3. Full Table-8-style reproduction

```bash
python reproduce_table8.py --B 500 --out expected_outputs/table8_seed0_B500.csv
```

Expected runtime: roughly seconds to a few tens of seconds, depending on CPU.
The output should show the qualitative ordering:

```text
Matched Delta < TopK=5 Delta < pred>0.5 Delta
```

for Realistic, Debiased, and Random regimes.

## Expected B=500 seed-0 output

The checked-in file `expected_outputs/table8_seed0_B500.csv` was generated with:

```bash
python reproduce_table8.py --B 500 --out expected_outputs/table8_seed0_B500.csv
```

The values may differ slightly across NumPy versions, but should remain close to:

| regime | Matched Δ | TopK=5 Δ | pred>0.5 Δ |
|---|---:|---:|---:|
| Realistic | +0.080 | +0.221 | +0.350 |
| Debiased | +0.211 | +0.330 | +0.409 |
| Random | +0.489 | +0.508 | +0.530 |

## Implementation details

1. The default entry point is now a quick smoke run, not an expensive full run.
2. The Random regime is now strict independent random scoring.
3. The bootstrap no longer merges duplicated top-k requisitions; cluster weights
   represent repeated bootstrap copies.
4. The README explicitly states that THA-Synth uses simplified pooled DI rather
   than the production case study's richer three-layer Equal DI aggregation.
5. Reproducibility files are included: `requirements.txt`, `quickstart.py`,
   `reproduce_table8.py`, `tests/test_smoke.py`, and expected outputs.
