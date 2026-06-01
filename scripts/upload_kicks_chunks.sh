#!/usr/bin/env bash
set -e

REMOTE="root@ssh1.vast.ai"
PORT=19195
REMOTE_DIR="/workspace/HS_kick_generator/data/processed/kicks.pt_chunks"
CHUNKS_DIR="data/processed/kicks.pt_chunks"

# Split into chunks
/c/Users/Menno/anaconda3/python scripts/split_file.py --file data/processed/kicks.pt --chunk-mb 200

# Create remote chunks directory
ssh -p "$PORT" "$REMOTE" "mkdir -p $REMOTE_DIR"

# Upload each chunk
for chunk in "$CHUNKS_DIR"/kicks.pt.part*; do
    echo "Uploading $chunk ..."
    scp -P "$PORT" "$chunk" "$REMOTE:$REMOTE_DIR/"
done

echo ""
echo "All chunks uploaded. On the remote run:"
echo "  python scripts/join_file.py --dir data/processed/kicks.pt_chunks --out data/processed/kicks.pt"
