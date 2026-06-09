"""Convenience wrapper: runs train.py in zeroshot mode.

Equivalent to: HP_ZEROSHOT=1 python train.py <name> --split <split>
"""
import os
import sys

os.environ["HP_ZEROSHOT"] = "1"
from rhaister.train import train_and_evaluate

if __name__ == "__main__":
    name = "zeroshot_phase2"
    split = "tahoe/5_holdout"
    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        name = args.pop(0)
    for i, a in enumerate(args):
        if a == "--split" and i + 1 < len(args):
            split = args[i + 1]
    train_and_evaluate(name, split)
