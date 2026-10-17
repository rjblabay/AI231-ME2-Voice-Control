#!/usr/bin/env bash
# Run this on the Mac.  It opens the local dashboard and keeps the Pi assistant
# plus the secure dashboard tunnel alive in this same terminal.
set -euo pipefail
PI_HOST="rblabay1@192.168.18.19"
( sleep 3; open "http://localhost:8000" ) &
exec ssh -o ExitOnForwardFailure=yes -L 8000:127.0.0.1:8000 "$PI_HOST" \
  'cd ~/ME2-Voice-Control && ME2_HEADLESS=1 ./run_astra_assistant.sh'
