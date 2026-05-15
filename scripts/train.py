"""
Launch RAVE training for hardstyle kicks.

Usage:
    python scripts/train.py
    python scripts/train.py --name my_kick_v1 --steps 800000
    python scripts/train.py --config configs/kick_rave.gin --db data/kicks.mdb
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train RAVE on hardstyle kicks")
    parser.add_argument("--config", default="configs/kick_rave.gin", help="Gin config path")
    parser.add_argument("--db", default="data/kicks.mdb", help="LMDB training database (from preprocess.py)")
    parser.add_argument("--name", default="kick_rave", help="Run name — used as checkpoint subdirectory")
    parser.add_argument("--out", default="outputs", help="Root output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=600000,
                        help="Training steps. 400k–800k recommended for 100–300 sample datasets")
    parser.add_argument("--val-every", type=int, default=10000, help="Validate and log audio every N steps")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (0 for first GPU)")
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db)
    out_path = Path(args.out)

    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run scripts/preprocess.py first.")
        sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "rave", "train",
        "--config", str(config_path),
        "--db_path", str(db_path),
        "--name", args.name,
        "--out_path", str(out_path),
        "--gpu", str(args.gpu),
    ]

    print("Launching RAVE training:")
    print("  " + " ".join(cmd))
    print(f"\nCheckpoints  → {out_path / args.name}")
    print(f"Monitor with → tensorboard --logdir {out_path}\n")

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: 'rave' CLI not found. Install with: pip install acids-rave")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nTraining interrupted. Latest checkpoint in: {out_path / args.name}")


if __name__ == "__main__":
    main()
