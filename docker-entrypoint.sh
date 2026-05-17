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
    while true; do
        sleep "${UPDATE_CHECK_INTERVAL:-3600}"

        if [ "${AUTO_UPDATE:-true}" != "true" ]; then
            continue
        fi

        CURRENT_TIME=$(date +%H:%M)
        TARGET_TIME="${UPDATE_TIME:-04:00}"

        echo "[updater] $(date): Checking time — current=$CURRENT_TIME target=$TARGET_TIME"

        if [ "$CURRENT_TIME" != "$TARGET_TIME" ]; then
            continue
        fi

        echo "[updater] $(date): Checking repository for changes..."

        # Only check if we have a remote configured
        if ! git remote get-url origin &>/dev/null; then
            echo "[updater] No git remote 'origin' configured. Skipping update check."
            continue
        fi

        # Fetch remote without merging
        if ! git fetch origin 2>/tmp/git_fetch_err; then
            echo "[updater] git fetch failed: $(cat /tmp/git_fetch_err)"
            continue
        fi

        LOCAL=$(git rev-parse HEAD)
        REMOTE=$(git rev-parse @{upstream} 2>/dev/null || echo "")

        if [ -z "$REMOTE" ]; then
            echo "[updater] No upstream branch configured. Skipping."
            continue
        fi

        if [ "$LOCAL" = "$REMOTE" ]; then
            echo "[updater] Repository is up-to-date."
            continue
        fi

        echo "[updater] Changes detected ($(git rev-list --count HEAD..@{upstream}) new commits). Pulling..."
        git pull

        echo "[updater] Reinstalling dependencies..."
        pip install --no-cache-dir -r requirements.txt 2>/dev/null || echo "[updater] Warning: pip install failed (non-fatal)"
        pip install --no-cache-dir demucs --no-deps 2>/dev/null || echo "[updater] Warning: demucs install failed (non-fatal)"

        echo "[updater] Update complete. Restarting server..."
        touch /tmp/restart-server
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
        echo "[entrypoint] Starting server: python server.py ${CLI_ARGS[*]}"
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
