"""Fast smoke run for THA-Synth.

This script is intended for artifact reviewers: it should finish quickly and
exercise R1-R5 without running the full B=500 reproduction.
"""

from synthetic_benchmark import demo

if __name__ == "__main__":
    demo(B=5, seed=0)
