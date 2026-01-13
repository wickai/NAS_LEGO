#!/bin/bash

# Remote configuration
REMOTE_USER="weizixiang"
REMOTE_HOST="210.75.240.33"
REMOTE_DIR="/home/weizixiang/dev/wk/github/NAS_LEGO/gridsearch_output"

# Sync gridsearch_output from remote to local
echo "Syncing gridsearch_output from ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} to ./ ..."

rsync -avzP ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR} ./

if [ $? -eq 0 ]; then
    echo "Sync completed successfully."
else
    echo "Sync failed."
fi
