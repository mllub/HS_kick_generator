"""
Split a large file into chunks for transfer.

Usage:
    python scripts/split_file.py                         # splits kicks.pt into 500 MB chunks
    python scripts/split_file.py --file data/processed/kicks.pt --chunk-mb 400
"""
import argparse
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",     default="data/processed/kicks.pt")
    parser.add_argument("--chunk-mb", type=int, default=500)
    args = parser.parse_args()

    src        = Path(args.file)
    chunk_size = args.chunk_mb * 1024 * 1024
    data       = src.read_bytes()
    n_chunks   = math.ceil(len(data) / chunk_size)

    print(f"File   : {src}  ({len(data)/1e9:.2f} GB)")
    print(f"Chunks : {n_chunks}  x  {args.chunk_mb} MB")

    out_dir = src.parent / (src.name + "_chunks")
    out_dir.mkdir(exist_ok=True)

    for i in range(n_chunks):
        chunk_path = out_dir / f"{src.name}.part{i:03d}"
        chunk_path.write_bytes(data[i * chunk_size : (i + 1) * chunk_size])
        print(f"  Written {chunk_path.name}  ({chunk_path.stat().st_size / 1e6:.0f} MB)")

    print(f"\nDone. Chunks in: {out_dir}")
    print(f"\nUpload with:")
    for i in range(n_chunks):
        print(f'  scp -P 31523 "{out_dir}/{src.name}.part{i:03d}" '
              f'root@ssh9.vast.ai:/workspace/HS_kick_generator/data/processed/{src.name}_chunks/')
    print(f"\nThen on the remote run:")
    print(f"  python scripts/join_file.py --dir data/processed/{src.name}_chunks --out data/processed/{src.name}")

if __name__ == "__main__":
    main()
