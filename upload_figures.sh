#!/bin/bash
# Use rsync to upload figures to mounted dropbox folder for Overleaf
# For Will's machine

overleaf_dir="$DROPBOX_PATH/Apps/Overleaf/[6.5930] Hardware for Deep Learning Project"

function dropbox {
  # start tmux detached, send rclone, and print message
  tmux new -d -s dropbox &&
  tmux send-keys -t dropbox C-z "rclone mount dropbox: $DROPBOX_PATH --vfs-cache-mode writes --allow-non-empty" Enter &&
  echo "Mounted Dropbox to $DROPBOX_PATH"
}

# Check if directory exists, if not mount dropbox
if [ ! -d "$overleaf_dir" ]; then
  echo "Dropbox not mounted, mounting..."
  dropbox
  sleep 3
fi

# Check if directory exists again, if not exit
if [ ! -d "$overleaf_dir" ]; then
  echo "Dropbox mount seems to have failed, exiting..."
  exit 1
else
  rsync -rvzha notebooks/figures/ "$overleaf_dir"/figures/
  echo "Copied notebooks/figures/ to $overleaf_dir/figures/"
fi
