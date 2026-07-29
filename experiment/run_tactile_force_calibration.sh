#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

SERIAL_BY_ID_DIR="${SERIAL_BY_ID_DIR:-/dev/serial/by-id}"
TACTILE_SERIAL_ID="${TACTILE_SERIAL_ID:-usb-1a86_USB_Single_Serial_5AA9026150-if00}"
FORCE_SERIAL_ID="${FORCE_SERIAL_ID:-usb-1a86_USB2.0-Ser_-if00-port0}"
MOTOR_SERIAL_ID="${MOTOR_SERIAL_ID:-}"
TACTILE_PORT="${TACTILE_PORT:-}"
FORCE_PORT="${FORCE_PORT:-}"
MOTOR_PORT="${MOTOR_PORT:-}"
FINGER="${FINGER:-thumb}"
DEVICE_ID="${DEVICE_ID:-0x03}"
TACTILE_RATE="${TACTILE_RATE:-30}"
FORCE_BAUDRATE="${FORCE_BAUDRATE:-2400}"
MOTOR_BAUDRATE="${MOTOR_BAUDRATE:-1000000}"
MOTOR_STATE_RATE="${MOTOR_STATE_RATE:-10}"
MOTOR_MAX_STEP="${MOTOR_MAX_STEP:-10}"
CONTROL_RATE="${CONTROL_RATE:-10}"
MAX_TRACKING_ERROR_DEG="${MAX_TRACKING_ERROR_DEG:-2.0}"
MOTOR_ZERO_DEG="${MOTOR_ZERO_DEG:-90 90 90 90}"
TARGET_FORCE_XYZ_N="${TARGET_FORCE_XYZ_N:-0 0 1}"
GAINS_MM_PER_N="${GAINS_MM_PER_N:-0.02 0.02 0.05}"
MOTION_SIGNS="${MOTION_SIGNS:-1 1 1}"
SENSOR_RPY_DEG="${SENSOR_RPY_DEG:-0 0 0}"
CONTACT_OFFSET_M="${CONTACT_OFFSET_M:-0 0 0}"
TACTILE_ENDPOINT="${TACTILE_ENDPOINT:-tcp://127.0.0.1:5561}"
FORCE_ENDPOINT="${FORCE_ENDPOINT:-tcp://127.0.0.1:5577}"
MOTOR_COMMAND_ENDPOINT="${MOTOR_COMMAND_ENDPOINT:-tcp://127.0.0.1:5580}"
MOTOR_STATE_ENDPOINT="${MOTOR_STATE_ENDPOINT:-tcp://127.0.0.1:5581}"
CONTROL_COMMAND_ENDPOINT="${CONTROL_COMMAND_ENDPOINT:-tcp://127.0.0.1:5590}"
CONTROL_STATE_ENDPOINT="${CONTROL_STATE_ENDPOINT:-tcp://127.0.0.1:5591}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/tactile_force_calibration_${RUN_ID}"
mkdir -p "${LOG_DIR}"

TACTILE_PID=""
FORCE_PID=""
MOTOR_PID=""
CONTROL_PID=""
CLEANED_UP=0

usage() {
    cat <<'EOF'
Usage: experiment/run_tactile_force_calibration.sh

Starts and supervises:
  1. Tactile Fx/Fy/Fz ZMQ publisher
  2. Reference force-sensor ZMQ publisher
  3. Robot/exoskeleton finger motor ZMQ node
  4. URDF-Jacobian XYZ admittance controller
  5. Qt calibration and motor-control UI in subscribe-only mode

Environment overrides:
  PYTHON_BIN, TACTILE_SERIAL_ID, FORCE_SERIAL_ID, MOTOR_SERIAL_ID
  TACTILE_PORT, FORCE_PORT, MOTOR_PORT, SERIAL_BY_ID_DIR, FINGER, DEVICE_ID
  TACTILE_RATE, FORCE_BAUDRATE, MOTOR_BAUDRATE, MOTOR_STATE_RATE
  MOTOR_MAX_STEP, TACTILE_ENDPOINT, FORCE_ENDPOINT
  MOTOR_COMMAND_ENDPOINT, MOTOR_STATE_ENDPOINT
  CONTROL_RATE, MAX_TRACKING_ERROR_DEG, MOTOR_ZERO_DEG, TARGET_FORCE_XYZ_N, GAINS_MM_PER_N
  MOTION_SIGNS, SENSOR_RPY_DEG, CONTACT_OFFSET_M
  CONTROL_COMMAND_ENDPOINT, CONTROL_STATE_ENDPOINT
EOF
}

case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter is not executable: ${PYTHON_BIN:-<empty>}" >&2
    exit 1
fi

print_serial_ids() {
    echo "Available serial devices under ${SERIAL_BY_ID_DIR}:" >&2
    local found=0
    local device
    while IFS= read -r device; do
        found=1
        echo "  $(basename -- "${device}") -> $(readlink -f -- "${device}")" >&2
    done < <(find "${SERIAL_BY_ID_DIR}" -maxdepth 1 -type l -print 2>/dev/null | sort)
    if ((found == 0)); then
        echo "  (none)" >&2
    fi
}

resolve_serial_id() {
    local role="$1"
    local serial_id="$2"
    local device
    if [[ "${serial_id}" == /* ]]; then
        device="${serial_id}"
    else
        device="${SERIAL_BY_ID_DIR}/${serial_id}"
    fi
    if [[ ! -e "${device}" ]]; then
        echo "${role} serial ID was not found: ${serial_id}" >&2
        print_serial_ids
        return 1
    fi
    printf '%s\n' "${device}"
}

if [[ -z "${TACTILE_PORT}" ]]; then
    TACTILE_PORT="$(resolve_serial_id "Tactile sensor" "${TACTILE_SERIAL_ID}")"
fi
if [[ -z "${FORCE_PORT}" ]]; then
    FORCE_PORT="$(resolve_serial_id "Reference force sensor" "${FORCE_SERIAL_ID}")"
fi
if [[ -z "${MOTOR_PORT}" && -n "${MOTOR_SERIAL_ID}" ]]; then
    MOTOR_PORT="$(resolve_serial_id "Motor bus" "${MOTOR_SERIAL_ID}")"
fi
if [[ -z "${MOTOR_PORT}" ]]; then
    tactile_real="$(readlink -f -- "${TACTILE_PORT}")"
    force_real="$(readlink -f -- "${FORCE_PORT}")"
    motor_candidates=()
    while IFS= read -r device; do
        device_real="$(readlink -f -- "${device}")"
        if [[ "${device_real}" != "${tactile_real}" && "${device_real}" != "${force_real}" ]]; then
            motor_candidates+=("${device}")
        fi
    done < <(find "${SERIAL_BY_ID_DIR}" -maxdepth 1 -type l -print 2>/dev/null | sort)
    if ((${#motor_candidates[@]} != 1)); then
        echo "Unable to identify the motor bus: expected exactly one unused serial by-id device, found ${#motor_candidates[@]}." >&2
        echo "Reconnect the motor adapter, or set MOTOR_SERIAL_ID to its by-id name." >&2
        print_serial_ids
        exit 1
    fi
    MOTOR_PORT="${motor_candidates[0]}"
    MOTOR_SERIAL_ID="$(basename -- "${MOTOR_PORT}")"
fi

declare -A SERIAL_TARGETS=()
for role_and_port in \
    "tactile:${TACTILE_PORT}" \
    "force:${FORCE_PORT}" \
    "motor:${MOTOR_PORT}"; do
    role="${role_and_port%%:*}"
    device="${role_and_port#*:}"
    if [[ ! -e "${device}" ]]; then
        echo "${role} serial device does not exist: ${device}" >&2
        exit 1
    fi
    device_real="$(readlink -f -- "${device}")"
    if [[ -n "${SERIAL_TARGETS[${device_real}]:-}" ]]; then
        echo "Serial device conflict: ${role} and ${SERIAL_TARGETS[${device_real}]} both resolve to ${device_real}." >&2
        exit 1
    fi
    SERIAL_TARGETS["${device_real}"]="${role}"
done

echo "Resolved serial devices by ID:"
echo "  tactile: ${TACTILE_PORT} -> $(readlink -f -- "${TACTILE_PORT}")"
echo "  force:   ${FORCE_PORT} -> $(readlink -f -- "${FORCE_PORT}")"
echo "  motor:   ${MOTOR_PORT} -> $(readlink -f -- "${MOTOR_PORT}")"

endpoint_port() {
    local endpoint="$1"
    printf '%s\n' "${endpoint##*:}"
}

port_is_listening() {
    local port="$1"
    ss -ltnH "( sport = :${port} )" 2>/dev/null | grep -q .
}

wait_for_node() {
    local name="$1"
    local pid="$2"
    local endpoint="$3"
    local log_file="$4"
    local port
    port="$(endpoint_port "${endpoint}")"
    for _ in $(seq 1 50); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "${name} exited during startup:" >&2
            tail -n 30 "${log_file}" >&2 || true
            return 1
        fi
        if port_is_listening "${port}"; then
            return 0
        fi
        sleep 0.1
    done
    echo "Timed out waiting for ${name} at ${endpoint}." >&2
    tail -n 30 "${log_file}" >&2 || true
    return 1
}

stop_pid() {
    local pid="$1"
    [[ -z "${pid}" ]] && return
    kill -0 "${pid}" 2>/dev/null || return
    kill -TERM "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
        kill -0 "${pid}" 2>/dev/null || return
        sleep 0.1
    done
    kill -KILL "${pid}" 2>/dev/null || true
}

cleanup() {
    local original_status=$?
    if ((CLEANED_UP)); then
        return
    fi
    CLEANED_UP=1
    trap - EXIT INT TERM HUP
    set +e
    stop_pid "${CONTROL_PID}"
    stop_pid "${MOTOR_PID}"
    stop_pid "${FORCE_PID}"
    stop_pid "${TACTILE_PID}"
    [[ -n "${CONTROL_PID}" ]] && wait "${CONTROL_PID}" 2>/dev/null
    [[ -n "${MOTOR_PID}" ]] && wait "${MOTOR_PID}" 2>/dev/null
    [[ -n "${FORCE_PID}" ]] && wait "${FORCE_PID}" 2>/dev/null
    [[ -n "${TACTILE_PID}" ]] && wait "${TACTILE_PID}" 2>/dev/null
    echo "Calibration experiment stopped. Logs: ${LOG_DIR}"
    exit "${original_status}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

for endpoint in \
    "${TACTILE_ENDPOINT}" \
    "${FORCE_ENDPOINT}" \
    "${MOTOR_COMMAND_ENDPOINT}" \
    "${MOTOR_STATE_ENDPOINT}" \
    "${CONTROL_COMMAND_ENDPOINT}" \
    "${CONTROL_STATE_ENDPOINT}"; do
    port="$(endpoint_port "${endpoint}")"
    if port_is_listening "${port}"; then
        echo "TCP port ${port} is already occupied (${endpoint})." >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"

echo "Starting tactile force node..."
"${PYTHON_BIN}" -u utils/tactile/tactile_sum_force_zmq_node.py \
    --port "${TACTILE_PORT}" \
    --finger "${FINGER}" \
    --device-id "${DEVICE_ID}" \
    --publish-rate "${TACTILE_RATE}" \
    --endpoint "${TACTILE_ENDPOINT}" \
    >"${LOG_DIR}/tactile.log" 2>&1 &
TACTILE_PID=$!
wait_for_node "Tactile node" "${TACTILE_PID}" "${TACTILE_ENDPOINT}" "${LOG_DIR}/tactile.log"

echo "Starting reference force node..."
"${PYTHON_BIN}" -u nodes/force_zmq_node.py \
    --port "${FORCE_PORT}" \
    --baudrate "${FORCE_BAUDRATE}" \
    --endpoint "${FORCE_ENDPOINT}" \
    >"${LOG_DIR}/force.log" 2>&1 &
FORCE_PID=$!
wait_for_node "Reference force node" "${FORCE_PID}" "${FORCE_ENDPOINT}" "${LOG_DIR}/force.log"

echo "Starting robot/exoskeleton finger motor node..."
"${PYTHON_BIN}" -u nodes/finger_pair_zmq_node.py \
    --port "${MOTOR_PORT}" \
    --baudrate "${MOTOR_BAUDRATE}" \
    --state-rate "${MOTOR_STATE_RATE}" \
    --max-step-deg "${MOTOR_MAX_STEP}" \
    --command-endpoint "${MOTOR_COMMAND_ENDPOINT}" \
    --state-endpoint "${MOTOR_STATE_ENDPOINT}" \
    >"${LOG_DIR}/motor.log" 2>&1 &
MOTOR_PID=$!
wait_for_node \
    "Finger motor node" \
    "${MOTOR_PID}" \
    "${MOTOR_COMMAND_ENDPOINT}" \
    "${LOG_DIR}/motor.log"

read -r -a MOTOR_ZERO_ARGS <<<"${MOTOR_ZERO_DEG}"
read -r -a TARGET_FORCE_ARGS <<<"${TARGET_FORCE_XYZ_N}"
read -r -a GAIN_ARGS <<<"${GAINS_MM_PER_N}"
read -r -a MOTION_SIGN_ARGS <<<"${MOTION_SIGNS}"
read -r -a SENSOR_RPY_ARGS <<<"${SENSOR_RPY_DEG}"
read -r -a CONTACT_OFFSET_ARGS <<<"${CONTACT_OFFSET_M}"
if ((${#MOTOR_ZERO_ARGS[@]} != 4)); then
    echo "MOTOR_ZERO_DEG must contain four values." >&2
    exit 1
fi
for values_name in TARGET_FORCE_ARGS GAIN_ARGS MOTION_SIGN_ARGS SENSOR_RPY_ARGS CONTACT_OFFSET_ARGS; do
    declare -n values_ref="${values_name}"
    if ((${#values_ref[@]} != 3)); then
        echo "${values_name} must contain three values." >&2
        exit 1
    fi
done
unset -n values_ref

echo "Starting URDF-Jacobian XYZ admittance controller..."
"${PYTHON_BIN}" -u nodes/tactile_admittance_control_node.py \
    --finger "${FINGER}" \
    --tactile-endpoint "${TACTILE_ENDPOINT}" \
    --motor-command-endpoint "${MOTOR_COMMAND_ENDPOINT}" \
    --motor-state-endpoint "${MOTOR_STATE_ENDPOINT}" \
    --command-endpoint "${CONTROL_COMMAND_ENDPOINT}" \
    --state-endpoint "${CONTROL_STATE_ENDPOINT}" \
    --control-rate "${CONTROL_RATE}" \
    --max-tracking-error-deg "${MAX_TRACKING_ERROR_DEG}" \
    --motor-zero-deg "${MOTOR_ZERO_ARGS[@]}" \
    --target-force-xyz-n "${TARGET_FORCE_ARGS[@]}" \
    --gains-mm-per-n "${GAIN_ARGS[@]}" \
    --motion-signs "${MOTION_SIGN_ARGS[@]}" \
    --sensor-rpy-deg "${SENSOR_RPY_ARGS[@]}" \
    --contact-offset-m "${CONTACT_OFFSET_ARGS[@]}" \
    >"${LOG_DIR}/control.log" 2>&1 &
CONTROL_PID=$!
wait_for_node \
    "Admittance control node" \
    "${CONTROL_PID}" \
    "${CONTROL_COMMAND_ENDPOINT}" \
    "${LOG_DIR}/control.log"

echo "Opening calibration UI..."
"${PYTHON_BIN}" demo/demo_tactile_force_calibration.py \
    --no-start-nodes \
    --tactile-endpoint "${TACTILE_ENDPOINT}" \
    --force-endpoint "${FORCE_ENDPOINT}" \
    --motor-command-endpoint "${MOTOR_COMMAND_ENDPOINT}" \
    --motor-state-endpoint "${MOTOR_STATE_ENDPOINT}" \
    --control-command-endpoint "${CONTROL_COMMAND_ENDPOINT}" \
    --control-state-endpoint "${CONTROL_STATE_ENDPOINT}" \
    --control-rate "${CONTROL_RATE}" \
    --max-tracking-error-deg "${MAX_TRACKING_ERROR_DEG}" \
    --motor-zero-deg "${MOTOR_ZERO_ARGS[@]}" \
    --target-force-xyz-n "${TARGET_FORCE_ARGS[@]}" \
    --gains-mm-per-n "${GAIN_ARGS[@]}" \
    --motion-signs "${MOTION_SIGN_ARGS[@]}" \
    --sensor-rpy-deg "${SENSOR_RPY_ARGS[@]}" \
    --contact-offset-m "${CONTACT_OFFSET_ARGS[@]}" \
    --finger "${FINGER}"
