#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null)}"
HARDWARE_PYTHON="${HARDWARE_PYTHON:-${CONDA_BASE}/bin/python}"
GEORT_PYTHON="${GEORT_PYTHON:-${CONDA_BASE}/envs/geort/bin/python}"

EX16_SERIAL="${EX16_SERIAL:-3901DF42503059384C2E3120FF0A0F1E}"
GX16_SERIAL="${GX16_SERIAL:-FTAKRP3A}"
CHECKPOINT="${CHECKPOINT:-gx16_2026-07-28_01-29-39_ex16_pinch_01}"
CALIBRATION="${CALIBRATION:-utils/GeoRT/data/human_ex16_pinch_01_ex16_raw.npz}"
VISER_HOST="${VISER_HOST:-127.0.0.1}"
VISER_PORT="${VISER_PORT:-8080}"
EX16_ZMQ_PORT="${EX16_ZMQ_PORT:-5567}"
GX16_COMMAND_PORT="${GX16_COMMAND_PORT:-5556}"
GX16_STATE_PORT="${GX16_STATE_PORT:-5557}"
GEORT_STATUS_PORT="${GEORT_STATUS_PORT:-5569}"
EX16_STATE_HZ="${EX16_STATE_HZ:-100}"
GEORT_UPDATE_HZ="${GEORT_UPDATE_HZ:-10}"
GEORT_SMOOTHING_ALPHA="${GEORT_SMOOTHING_ALPHA:-0.5}"
GX16_CURR_LIMIT="${GX16_CURR_LIMIT:-800}"
GX16_GOAL_CURRENT="${GX16_GOAL_CURRENT:-500}"
GX16_GOAL_PWM="${GX16_GOAL_PWM:-300}"

EX16_ENDPOINT="tcp://127.0.0.1:${EX16_ZMQ_PORT}"
GX16_COMMAND_ENDPOINT="tcp://127.0.0.1:${GX16_COMMAND_PORT}"
GX16_STATE_ENDPOINT="tcp://127.0.0.1:${GX16_STATE_PORT}"
GEORT_STATUS_ENDPOINT="tcp://127.0.0.1:${GEORT_STATUS_PORT}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: experiment/run_ex16_geort_gx16.sh [--dry-run]

Starts and supervises:
  1. EX16 state publisher
  2. GX16 hardware-owner node
  3. GeoRT retargeting, GX16 command bridge, and Viser

The default mode connects real EX16 and GX16 hardware. --dry-run uses virtual
EX16 sliders and a GX16 node that only prints commands.

Common environment overrides:
  EX16_SERIAL, GX16_SERIAL, CHECKPOINT, CALIBRATION, VISER_PORT
  GEORT_UPDATE_HZ, GEORT_SMOOTHING_ALPHA
  GX16_CURR_LIMIT, GX16_GOAL_CURRENT, GX16_GOAL_PWM
  EX16_ZMQ_PORT, GX16_COMMAND_PORT, GX16_STATE_PORT, GEORT_STATUS_PORT
  HARDWARE_PYTHON, GEORT_PYTHON
EOF
}

case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

for python_path in "${HARDWARE_PYTHON}" "${GEORT_PYTHON}"; do
    if [[ ! -x "${python_path}" ]]; then
        echo "Python interpreter is not executable: ${python_path}" >&2
        exit 1
    fi
done
if [[ ! -f "${REPO_ROOT}/${CALIBRATION}" ]]; then
    echo "Calibration file does not exist: ${REPO_ROOT}/${CALIBRATION}" >&2
    exit 1
fi
if [[ ! -d "${REPO_ROOT}/utils/GeoRT/checkpoint/${CHECKPOINT}" ]]; then
    echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

port_is_listening() {
    local port="$1"
    ss -ltnH "( sport = :${port} )" 2>/dev/null | grep -q .
}

tcp_listener_pids() {
    local port="$1"
    local pids
    pids="$(
        ss -ltnpH "( sport = :${port} )" 2>/dev/null \
            | grep -o 'pid=[0-9]\+' \
            | cut -d= -f2 \
            | sort -u || true
    )"
    if [[ -z "${pids}" ]] && command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -nP -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true)"
    fi
    printf '%s\n' "${pids}" | sed '/^$/d'
}

wait_for_port_free() {
    local port="$1"
    local attempts="${2:-50}"
    local attempt
    for ((attempt = 0; attempt < attempts; attempt++)); do
        if ! port_is_listening "${port}"; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

free_tcp_port() {
    local port="$1"
    local pids=()
    if ! port_is_listening "${port}"; then
        return 0
    fi

    mapfile -t pids < <(tcp_listener_pids "${port}")
    if ((${#pids[@]} == 0)); then
        echo "TCP port ${port} is occupied, but no listener PID was visible." >&2
        return 1
    fi

    echo "TCP port ${port} is occupied by PID(s): ${pids[*]}; stopping them..."
    kill -TERM "${pids[@]}" 2>/dev/null || true
    if wait_for_port_free "${port}" 50; then
        return 0
    fi

    mapfile -t pids < <(tcp_listener_pids "${port}")
    if ((${#pids[@]} > 0)); then
        echo "TCP port ${port} is still occupied; force killing PID(s): ${pids[*]}..."
        kill -KILL "${pids[@]}" 2>/dev/null || true
    fi
    if wait_for_port_free "${port}" 30; then
        return 0
    fi

    echo "TCP port ${port} is still occupied after cleanup." >&2
    return 1
}

for port in "${EX16_ZMQ_PORT}" "${GX16_COMMAND_PORT}" "${GX16_STATE_PORT}" "${GEORT_STATUS_PORT}" "${VISER_PORT}"; do
    free_tcp_port "${port}"
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

EX16_PID=""
GX16_PID=""
GEORT_PID=""
CLEANED_UP=0

wait_for_port() {
    local name="$1"
    local port="$2"
    local pid="$3"
    local log="$4"
    local attempts="${5:-200}"
    local attempt
    for ((attempt = 0; attempt < attempts; attempt++)); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited during startup. Last log lines:" >&2
            tail -n 30 "${log}" >&2 || true
            return 1
        fi
        if port_is_listening "${port}"; then
            return 0
        fi
        sleep 0.1
    done
    echo "Timed out waiting for ${name} on TCP port ${port}." >&2
    tail -n 30 "${log}" >&2 || true
    return 1
}

wait_for_pid_exit() {
    local pid="$1"
    local attempt
    [[ -z "${pid}" ]] && return 0
    for ((attempt = 0; attempt < 30; attempt++)); do
        kill -0 "${pid}" 2>/dev/null || return 0
        sleep 0.1
    done
    return 1
}

gx16_request() {
    "${HARDWARE_PYTHON}" "${REPO_ROOT}/demo/demo_gx16_zmq_client.py" \
        --endpoint "${GX16_COMMAND_ENDPOINT}" --timeout-ms 500 "$@"
}

wait_for_gx16_ready() {
    local pid="$1"
    local log="$2"
    local attempt
    for ((attempt = 0; attempt < 20; attempt++)); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "GX16 node exited before responding to ping. Last log lines:" >&2
            tail -n 30 "${log}" >&2 || true
            return 1
        fi
        if gx16_request ping >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    echo "GX16 command endpoint did not respond to ping." >&2
    tail -n 30 "${log}" >&2 || true
    return 1
}

cleanup() {
    local original_status=$?
    if ((CLEANED_UP)); then
        return
    fi
    CLEANED_UP=1
    trap - EXIT INT TERM HUP
    set +e
    echo
    echo "Stopping EX16 → GeoRT → GX16 experiment..."

    # Stop the command producer before touching the hardware owner.
    if [[ -n "${GEORT_PID}" ]] && kill -0 "${GEORT_PID}" 2>/dev/null; then
        kill -TERM "${GEORT_PID}" 2>/dev/null
        wait_for_pid_exit "${GEORT_PID}" || kill -KILL "${GEORT_PID}" 2>/dev/null
        wait "${GEORT_PID}" 2>/dev/null
    fi

    # Best effort, and deliberately before shutting down the GX16 node.
    if [[ -n "${GX16_PID}" ]] && kill -0 "${GX16_PID}" 2>/dev/null; then
        gx16_request torque_off >/dev/null 2>&1 || \
            echo "Warning: GX16 torque_off request failed." >&2
        gx16_request shutdown >/dev/null 2>&1 || true
        wait_for_pid_exit "${GX16_PID}" || kill -TERM "${GX16_PID}" 2>/dev/null
        wait_for_pid_exit "${GX16_PID}" || kill -KILL "${GX16_PID}" 2>/dev/null
        wait "${GX16_PID}" 2>/dev/null
    fi

    if [[ -n "${EX16_PID}" ]] && kill -0 "${EX16_PID}" 2>/dev/null; then
        kill -TERM "${EX16_PID}" 2>/dev/null
        wait_for_pid_exit "${EX16_PID}" || kill -KILL "${EX16_PID}" 2>/dev/null
        wait "${EX16_PID}" 2>/dev/null
    fi

    echo "All experiment processes stopped. Logs: ${LOG_DIR}"
    exit "${original_status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

cd "${REPO_ROOT}"

if ((DRY_RUN)); then
    echo "Starting virtual EX16 publisher..."
    "${HARDWARE_PYTHON}" nodes/fake_ex16_zmq_node.py \
        --state-endpoint "${EX16_ENDPOINT}" --state-hz "${EX16_STATE_HZ}" \
        >"${LOG_DIR}/ex16.log" 2>&1 &
else
    echo "Starting real EX16 publisher (${EX16_SERIAL})..."
    "${HARDWARE_PYTHON}" nodes/ex16_zmq_node.py \
        --serial-number "${EX16_SERIAL}" \
        --state-endpoint "${EX16_ENDPOINT}" --state-hz "${EX16_STATE_HZ}" \
        --control-status-endpoint "${GEORT_STATUS_ENDPOINT}" \
        >"${LOG_DIR}/ex16.log" 2>&1 &
fi
EX16_PID=$!
wait_for_port "EX16 publisher" "${EX16_ZMQ_PORT}" "${EX16_PID}" "${LOG_DIR}/ex16.log"

echo "Starting GX16 node$([[ ${DRY_RUN} -eq 1 ]] && echo ' (dry-run)')..."
GX16_ARGS=(
    --serial-number "${GX16_SERIAL}"
    --cmd-endpoint "${GX16_COMMAND_ENDPOINT}"
    --state-endpoint "${GX16_STATE_ENDPOINT}"
    --curr-limit "${GX16_CURR_LIMIT}"
    --goal-current "${GX16_GOAL_CURRENT}"
    --goal-pwm "${GX16_GOAL_PWM}"
)
if ((DRY_RUN)); then
    GX16_ARGS+=(--dry-run)
fi
"${HARDWARE_PYTHON}" nodes/gx16_zmq_node.py "${GX16_ARGS[@]}" \
    >"${LOG_DIR}/gx16.log" 2>&1 &
GX16_PID=$!
wait_for_port \
    "GX16 command node" "${GX16_COMMAND_PORT}" "${GX16_PID}" \
    "${LOG_DIR}/gx16.log" 600
wait_for_gx16_ready "${GX16_PID}" "${LOG_DIR}/gx16.log"

echo "Starting GeoRT + Viser + GX16 command bridge..."
"${GEORT_PYTHON}" demo/demo_ex16_geort_gx16_viser.py \
    --ckpt-tag "${CHECKPOINT}" \
    --calibration "${CALIBRATION}" \
    --ex16-endpoint "${EX16_ENDPOINT}" \
    --gx16-command-endpoint "${GX16_COMMAND_ENDPOINT}" \
    --enable-gx16-output \
    --no-collision-check \
    --update-hz "${GEORT_UPDATE_HZ}" \
    --smoothing-alpha "${GEORT_SMOOTHING_ALPHA}" \
    --control-status-endpoint "${GEORT_STATUS_ENDPOINT}" \
    --host "${VISER_HOST}" --port-viser "${VISER_PORT}" \
    >"${LOG_DIR}/geort_viser.log" 2>&1 &
GEORT_PID=$!
wait_for_port "GeoRT Viser" "${VISER_PORT}" "${GEORT_PID}" "${LOG_DIR}/geort_viser.log" 600

echo
if ((DRY_RUN)); then
    echo "DRY-RUN experiment is active."
else
    echo "REAL GX16 OUTPUT IS ACTIVE with conservative current/PWM limits."
fi
echo "Viser: http://${VISER_HOST}:${VISER_PORT}"
echo "Logs:  ${LOG_DIR}"
echo "Press Ctrl+C, close the EX16 window, or stop any child to shut down everything."

set +e
wait -n "${EX16_PID}" "${GX16_PID}" "${GEORT_PID}"
CHILD_STATUS=$?
set -e
echo "A child process exited (status ${CHILD_STATUS}); stopping the experiment." >&2
exit "${CHILD_STATUS}"
