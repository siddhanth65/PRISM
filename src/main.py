"""
End-to-end PRISM pipeline driver.

Runs all 7 stages sequentially:
    Stage 1: BPR-MF + multi-scale SVD
    Stage 2: User and item profile builder
    Stage 3: MiniLM embeddings and positive prototype
    Stage 4: 36-D feature vector assembly
    Stage 5: Evaluation and alpha grid search (depends on Stage 6 output)
    Stage 6: 5-fold OOF stacking with LightGBM LambdaRank
    Stage 7: Inductive conformal prediction

Usage:
    python src/main.py --data data/ml-100k --out runs/

Each stage caches its artefacts to disk, so reruns will skip completed stages
unless --force is given.
"""

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = [
    ("stage1_bpr_svd",    "BPR-MF and multi-scale SVD"),
    ("stage2_profiles",   "User and item profiles"),
    ("stage3_embeddings", "MiniLM embeddings"),
    ("stage4_features",   "Feature vector assembly"),
    ("stage6_stacking",   "OOF stacking with LightGBM LambdaRank"),
    ("stage5_eval",       "Alpha grid search"),
    ("stage7_conformal",  "Conformal calibration"),
]


def run_stage(stage_module: str, args, force: bool):
    cmd = [sys.executable, "-m", f"src.{stage_module}",
           "--data", str(args.data),
           "--out", str(args.out / stage_module.split("_")[0])]
    print(f"\n{'=' * 64}\nRunning {stage_module}\n{'=' * 64}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml-100k", type=Path)
    parser.add_argument("--out", default="runs", type=Path)
    parser.add_argument("--force", action="store_true",
                        help="Re-run all stages, ignoring cached artefacts")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for stage, desc in STAGES:
        print(f"\n>>> {stage}: {desc}")
        run_stage(stage, args, args.force)

    print("\nPRISM pipeline complete.")
    print(f"Artefacts: {args.out}")
    print("\nNext: open demo/prism_demo.html to interact with the model.")


if __name__ == "__main__":
    main()
