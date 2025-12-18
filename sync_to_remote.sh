#!/bin/bash

# Remote configuration
REMOTE_USER="weizixiang"
REMOTE_HOST="210.75.240.33"
REMOTE_DIR="/home/weizixiang/dev/wk/github/NAS_LEGO"

# Sync python and shell files
echo "Syncing Python and Shell files to ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}..."

scp *.py *.sh ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}

if [ $? -eq 0 ]; then
    echo "Sync completed successfully."
else
    echo "Sync failed."
fi
