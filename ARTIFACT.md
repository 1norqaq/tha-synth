"""Minimal smoke tests for THA-Synth.

Run from repository root:
    python tests/test_smoke.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from synthetic_benchmark import (  # noqa: E402
    AuditConfig,
    GenConfig,
    generate,
    run_audit,
    run_table8,
    select_topk_per_job,
)


def main() -> None:
    small = GenConfig(n_jobs=40, candidates_per_job_lambda=8, seed=123)
    df = generate(small)
    assert {"job_id", "group", "age_band", "pred", "score"}.issubset(df.columns)
    assert 0 < df["score"].mean() < 0.2

    top = select_topk_per_job(df, k=3)
    assert len(top) == len(df)
    assert top.sum() <= 3 * df["job_id"].nunique()

    out = run_audit(small, AuditConfig(B=3, seed=123))
    assert set(out["operating_point"]) == {"Matched", "TopK=5", "pred>0.5"}

    tbl = run_table8(B=2, seed=123)
    assert set(tbl["regime"]) == {"Realistic", "Debiased", "Random"}
    assert len(tbl) == 9
    print("Smoke tests passed.")


if __name__ == "__main__":
    main()
