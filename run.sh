#!/usr/bin/env bash
# Starts/stops the backend (FastAPI :8000) and frontend (Streamlit :8501)
# as background processes on this server. Mirrors the manual two-terminal
# steps (cd, activate venv, run) but backgrounded with nohup so one SSH
# session can launch both and still get its prompt back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NLPV2_DIR="$SCRIPT_DIR/NLPv2"
VENV_PY="$NLPV2_DIR/venv/bin/python"
VENV_PIP="$NLPV2_DIR/venv/bin/pip"
VENV_STREAMLIT="$NLPV2_DIR/venv/bin/streamlit"

RUN_DIR="$SCRIPT_DIR/.run"
LOG_DIR="$SCRIPT_DIR/logs"
API_PID_FILE="$RUN_DIR/api.pid"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"

mkdir -p "$RUN_DIR" "$LOG_DIR"

is_running() {
    # $1 = pidfile
    [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null
}

start() {
    if [[ ! -x "$VENV_PY" ]]; then
        echo "venv not found at $NLPV2_DIR/venv — create it first: python3 -m venv $NLPV2_DIR/venv" >&2
        exit 1
    fi

    echo "Pulling latest..."
    git -C "$SCRIPT_DIR" pull

    if is_running "$API_PID_FILE"; then
        echo "API already running (pid $(cat "$API_PID_FILE"))"
    else
        echo "Installing backend deps..."
        "$VENV_PIP" install -q -r "$NLPV2_DIR/backend/requirements.txt"

        echo "Starting API..."
        (
            cd "$NLPV2_DIR/backend"
            nohup "$VENV_PY" api.py >> "$LOG_DIR/api.log" 2>&1 &
            echo $! > "$API_PID_FILE"
        )
        sleep 1
        echo "API started (pid $(cat "$API_PID_FILE")), log: $LOG_DIR/api.log"
    fi

    if is_running "$FRONTEND_PID_FILE"; then
        echo "Frontend already running (pid $(cat "$FRONTEND_PID_FILE"))"
    else
        echo "Installing frontend deps..."
        "$VENV_PIP" install -q -r "$NLPV2_DIR/frontend/streamlit/requirements.txt"

        echo "Starting frontend..."
        (
            cd "$NLPV2_DIR/frontend/streamlit"
            RAG_API_URL=http://localhost:8000 \
                nohup "$VENV_STREAMLIT" run app.py --server.port 8501 --server.address 0.0.0.0 \
                >> "$LOG_DIR/frontend.log" 2>&1 &
            echo $! > "$FRONTEND_PID_FILE"
        )
        sleep 1
        echo "Frontend started (pid $(cat "$FRONTEND_PID_FILE")), log: $LOG_DIR/frontend.log"
    fi

    echo
    echo "API:      http://localhost:8000  (docs at /docs)"
    echo "Frontend: http://localhost:8501"
}

stop() {
    for name in api frontend; do
        pidfile="$RUN_DIR/$name.pid"
        if is_running "$pidfile"; then
            pid="$(cat "$pidfile")"
            echo "Stopping $name (pid $pid)..."
            kill "$pid"
            for _ in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
            rm -f "$pidfile"
        else
            echo "$name not running"
            rm -f "$pidfile"
        fi
    done
}

status() {
    for name in api frontend; do
        pidfile="$RUN_DIR/$name.pid"
        if is_running "$pidfile"; then
            echo "$name: running (pid $(cat "$pidfile"))"
        else
            echo "$name: stopped"
        fi
    done
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status) status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}" >&2
        exit 1
        ;;
esac
