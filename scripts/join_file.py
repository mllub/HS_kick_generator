"""
Reassemble chunks produced by split_file.py.

Usage (on the remote):
    python scripts/join_file.py --dir data/processed/kicks.pt_chunks --out data/processed/kicks.pt
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  required=True, help="Directory containing .partXXX files")
    parser.add_argument("--out",  required=True, help="Output file path")
    args = parser.parse_args()

    chunk_dir = Path(args.dir)
    chunks    = sorted(chunk_dir.glob("*.part*"))

    if not chunks:
        print(f"No chunk files found in {chunk_dir}")
        return

    print(f"Found {len(chunks)} chunks — reassembling into {args.out} ...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        for chunk in chunks:
            print(f"  + {chunk.name}  ({chunk.stat().st_size / 1e6:.0f} MB)")
            f.write(chunk.read_bytes())

    print(f"\nDone. {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")

if __name__ == "__main__":
    main()
