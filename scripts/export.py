"""
Export a trained RAVE checkpoint to TorchScript (.ts) for inference.

Must be run after training completes (or from a checkpoint you want to ship).

Usage:
    python scripts/export.py --run outputs/kick_rave
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export RAVE checkpoint to TorchScript")
    parser.add_argument("--run", required=True,
                        help="Training run directory, e.g. outputs/kick_rave")
    parser.add_argument("--streaming", action="store_true",
                        help="Export in streaming (causal) mode for real-time use")
    args = parser.parse_args()

    run_path = Path(args.run)
    if not run_path.exists():
        print(f"Run directory not found: {run_path}")
        sys.exit(1)

    cmd = ["rave", "export", "--run", str(run_path)]
    if args.streaming:
        cmd.append("--streaming")

    print("Exporting model:")
    print("  " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: 'rave' CLI not found. Install with: pip install acids-rave")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Export failed with exit code {e.returncode}")
        sys.exit(1)

    ts_files = sorted(run_path.glob("*.ts"))
    if ts_files:
        model_path = ts_files[-1]
        print(f"\nExported → {model_path}")
        print("\nNext steps:")
        print(f"  Generate kicks : python scripts/generate.py --model {model_path}")
        print(f"  Interpolate    : python scripts/interpolate.py --model {model_path} --kick-a A.wav --kick-b B.wav")


if __name__ == "__main__":
    main()
