#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$HOME/.config/me2/assistant.env"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
pkill -f 'app.raspberry_pi.voice_controller' 2>/dev/null || true
pkill -f 'app.raspberry_pi.dashboard' 2>/dev/null || true
"$ROOT/.venv/bin/python" -m app.raspberry_pi.dashboard &
# The Pi desktop normally owns :0.  Use it even when this launcher is started
# through SSH, so one command starts both the voice service and HDMI dashboard.
PI_DISPLAY="${DISPLAY:-:0}"
if [[ "${ME2_HEADLESS:-0}" != "1" ]] && command -v chromium >/dev/null 2>&1; then
  DISPLAY="$PI_DISPLAY" XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    chromium --kiosk --noerrdialogs http://127.0.0.1:8000 >/tmp/me2-astra-browser.log 2>&1 &
fi
exec "$ROOT/.venv/bin/python" -m app.raspberry_pi.voice_controller --device 1 --mode push --wake-command-delay-sec 0.75 --command-threshold 0.0 --wake-threshold 0.65 --wake-confirmations 3
