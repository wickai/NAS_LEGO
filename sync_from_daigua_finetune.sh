#!/bin/bash

# Remote configuration
REMOTE_USER="wk"
REMOTE_HOST="202.112.47.35"
REMOTE_BASE_DIR="/data/wk/kai/github/wk_new"
REMOTE_SOURCE_DIR="${REMOTE_BASE_DIR}/logs/finetune_grid/"

# Local configuration
LOCAL_DIR="./logs/finetune_grid_20blk/"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Sync from remote to local
# Only syncing .log and .csv files as requested (avoiding potentially large .pth files)
echo "Syncing logs and csv from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SOURCE_DIR} to ${LOCAL_DIR} ..."

rsync -avzP \
    --include="*.log" \
    --include="*.csv" \
    --exclude="*" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SOURCE_DIR}" \
    "$LOCAL_DIR"

if [ $? -eq 0 ]; then
    echo "Sync completed successfully."
else
    echo "Sync failed."
fi
