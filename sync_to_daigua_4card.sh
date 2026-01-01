#!/bin/bash

# Remote configuration
REMOTE_USER="wk"
REMOTE_HOST="202.112.47.35"
REMOTE_DIR="/data/wk/kai/github/wk_new"

if [ -n "$1" ]; then
    # Sync specific file provided as argument
    echo "Syncing specific file '$1' to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
    rsync -avzP "$1" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}
else
    # Default: Sync python, shell and json files
    echo "Syncing Python, Shell and JSON files to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."
    rsync -avzP *.py *.sh *.json ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}
fi

if [ $? -eq 0 ]; then
    echo "Sync completed successfully."
else
    echo "Sync failed."
fi
