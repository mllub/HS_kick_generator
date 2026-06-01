#!/usr/bin/env bash
set -e

REMOTE="root@ssh1.vast.ai"
PORT=19195
REMOTE_DIR="/workspace/HS_kick_generator/data/raw"

echo "Uploading raw kick samples to $REMOTE:$REMOTE_DIR ..."
ssh -p "$PORT" "$REMOTE" "mkdir -p $REMOTE_DIR"
scp -P "$PORT" -r "C:/Users/Menno/hardstyle-kick-rave/data/raw/." "$REMOTE:$REMOTE_DIR/"
echo "Done."
