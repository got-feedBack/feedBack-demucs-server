#!/bin/bash
# ============================================================
# Slopsmith Demucs Server — Docker Entrypoint
# ============================================================
# Starts the FastAPI server and an optional background auto-update
# daemon that periodically checks the git remote for changes.
#
# Environment variables:
#   PORT                    — Server port (default: 7865)
#   HOST                    — Server bind address (default: 0.0.0.0)
#   AUTO_UPDATE             — Enable auto-update (default: true)
#   UPDATE_TIME             — HH:MM to check for updates (default: 04:00)
#   UPDATE_CHECK_INTERVAL   — Seconds between time checks (default: 3600)
#   SKIP_WARMUP             — Pass --skip-warmup to server (default: false)
#   SLOPSMITH_DEMUCS_MODEL  — Demucs model override
#   SLOPSMITH_API_KEY       — API auth key
# ============================================================

set -euo pipefail

cd /app


# ── Auto-update daemon ──────────────────────────────────────────────────
auto_update_loop() {
    LAST_UPDATE_DATE=""

    while true; do
        if [ "${AUTO_UPDATE:-true}" = "true" ]; then
            CURRENT_TIME=$(date +%H:%M)
            TARGET_TIME="${UPDATE_TIME:-04:00}"

            echo "[updater] $(date): Checking time — current=$CURRENT_TIME target=$TARGET_TIME"

            # Convert times to minutes since midnight for numeric comparison
            CURRENT_MINUTES=$(( $(date +%H) * 60 + $(date +%M) ))
            TARGET_HOUR="${TARGET_TIME%:*}"
            TARGET_MIN="${TARGET_TIME#*:}"
            # Use 10# prefix to avoid octal interpretation of leading zeros
            TARGET_HOUR=$((10#$TARGET_HOUR))
            TARGET_MIN=$((10#$TARGET_MIN))
            TARGET_MINUTES=$(( TARGET_HOUR * 60 + TARGET_MIN ))

            # 30-minute window around the target time
            WINDOW=30
            DIFF=$(( CURRENT_MINUTES - TARGET_MINUTES ))
            ABS_DIFF=$(( DIFF < 0 ? -DIFF : DIFF ))

            TODAY=$(date +%F)
            if [ "$ABS_DIFF" -le "$WINDOW" ] && [ "$LAST_UPDATE_DATE" != "$TODAY" ]; then
                echo "[updater] $(date): Time matches — checking repository for changes..."

                # Only check if we have a remote configured
                if ! git remote get-url origin &>/dev/null; then
                    echo "[updater] No git remote 'origin' configured. Skipping update check."
                else
                    # Fetch remote without merging
                    if ! git fetch origin 2>/tmp/git_fetch_err; then
                        echo "[updater] git fetch failed: $(cat /tmp/git_fetch_err)"
                    else
                        LOCAL=$(git rev-parse HEAD)
                        REMOTE=$(git rev-parse @{upstream} 2>/dev/null || echo "")

                        if [ -z "$REMOTE" ]; then
                            echo "[updater] No upstream branch configured. Skipping."
                        elif [ "$LOCAL" = "$REMOTE" ]; then
                            echo "[updater] Repository is up-to-date."
                        else
                            echo "[updater] Changes detected ($(git rev-list --count HEAD..@{upstream}) new commits). Updating..."
                            if git pull --ff-only; then
                                echo "[updater] Reinstalling dependencies..."
                                pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo "[updater] Warning: pip install failed (non-fatal)"
                                pip install --no-cache-dir demucs --no-deps 2>/dev/null || echo "[updater] Warning: demucs install failed (non-fatal)"
                                echo "[updater] Update complete. Restarting server..."
                                touch /tmp/restart-server
                            else
                                echo "[updater] Fast-forward update failed. Skipping this cycle."
                            fi
                        fi
                    fi
                fi

                LAST_UPDATE_DATE="$TODAY"
            fi
        fi

        sleep "${UPDATE_CHECK_INTERVAL:-3600}"
    done
}
# ── Graceful shutdown ─────────────────────────────────────────────────
cleanup() {
    echo "[entrypoint] SIGTERM received. Stopping server (PID ${SERVER_PID:-unknown})..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGTERM SIGINT


# ── Server restart loop ─────────────────────────────────────────────────
restart_loop() {
    # Build CLI args
    CLI_ARGS=("--port" "${PORT:-7865}" "--host" "${HOST:-0.0.0.0}")
    if [ "${SKIP_WARMUP:-false}" = "true" ]; then
        CLI_ARGS+=("--skip-warmup")
    fi
    if [ -n "${SLOPSMITH_DEMUCS_MODEL:-}" ]; then
        CLI_ARGS+=("--model" "$SLOPSMITH_DEMUCS_MODEL")
    fi
    if [ -n "${SLOPSMITH_API_KEY:-}" ]; then
        CLI_ARGS+=("--api-key" "$SLOPSMITH_API_KEY")
    fi

    while true; do
        echo "[entrypoint] Starting server: python server.py ${CLI_ARGS[*]}" | sed "s/--api-key [^ ]*/--api-key ****/g"
        python server.py "${CLI_ARGS[@]}" &
        SERVER_PID=$!
        echo "[entrypoint] Server PID: $SERVER_PID"

        # Wait for server process or restart signal
        while kill -0 "$SERVER_PID" 2>/dev/null; do
            if [ -f /tmp/restart-server ]; then
                rm -f /tmp/restart-server
                echo "[entrypoint] Restart signal received. Gracefully stopping server (PID $SERVER_PID)..."
                kill -TERM "$SERVER_PID" 2>/dev/null || true
                wait "$SERVER_PID" 2>/dev/null || true
                echo "[entrypoint] Server stopped."
                break
            fi
            sleep 1
        done

        # If auto-update is off, the outer update loop never creates
        # restart-server files, so we break only if the server exits
        # on its own AND no restart is pending.
        if [ ! -f /tmp/restart-server ] && [ "${AUTO_UPDATE:-true}" != "true" ]; then
            echo "[entrypoint] Server exited and auto-update disabled. Exiting."
            break
        fi
        # Crash-loop backoff — prevent thrash if server exits quickly
        sleep 2
    done
}

# ── Start ───────────────────────────────────────────────────────────────
echo "[entrypoint] Starting Slopsmith Demucs Server"
echo "[entrypoint]   Port:           ${PORT:-7865}"
echo "[entrypoint]   Host:           ${HOST:-0.0.0.0}"
echo "[entrypoint]   Auto-update:    ${AUTO_UPDATE:-true}"
echo "[entrypoint]   Update time:    ${UPDATE_TIME:-04:00}"
echo "[entrypoint]   Check interval: ${UPDATE_CHECK_INTERVAL:-3600}s"

# Start auto-update daemon in background (if enabled)
if [ "${AUTO_UPDATE:-true}" = "true" ]; then
    auto_update_loop &
    echo "[entrypoint] Auto-update daemon started (PID $!)"
else
    echo "[entrypoint] Auto-update disabled."
fi

# Start server with restart capability
restart_loop
