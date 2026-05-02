"""Reproduce the THA-Synth Table-8-style synthetic benchmark.

Quick check:
    python reproduce_table8.py --quick

Full paper-style run:
    python reproduce_table8.py --B 500 --out expected_outputs/table8_seed0_B500.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from synthetic_benchmark import run_table8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="Use B=20 for a fast smoke run.")
    p.add_argument("--B", type=int, default=500, help="Bootstrap replicates for full run.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--out", type=str, default="expected_outputs/table8_seed0.csv", help="CSV output path.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    B = 20 if args.quick else args.B
    table = run_table8(B=B, seed=args.seed)
    print(table.to_string(index=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\nWrote {out} with B={B}")
