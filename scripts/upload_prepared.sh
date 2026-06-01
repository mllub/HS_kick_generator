#!/usr/bin/env bash
set -e

REMOTE="root@ssh1.vast.ai"
PORT=19195
REMOTE_DIR="/workspace/HS_kick_generator/data/prepared"

echo "Uploading kicks_prepared.npy to $REMOTE:$REMOTE_DIR ..."
ssh -p "$PORT" "$REMOTE" "mkdir -p $REMOTE_DIR"
scp -P "$PORT" \
    "C:/Users/Menno/hardstyle-kick-rave/data/prepared/kicks_prepared.npy" \
    "$REMOTE:$REMOTE_DIR/"
echo "Done."
