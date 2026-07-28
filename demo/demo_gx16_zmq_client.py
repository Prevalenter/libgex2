import argparse
import json
import sys


DEFAULT_CMD_ENDPOINT = "tcp://127.0.0.1:5556"
DEFAULT_STATE_ENDPOINT = "tcp://127.0.0.1:5557"
STATE_TOPIC = "gx16/state"
JOINT_COUNT = 16
VALID_UNITS = ("urdf_deg", "motor_deg")


def _load_zmq():
    try:
        import zmq
    except ImportError:
        print(
            "pyzmq is required. Install it with: python -m pip install pyzmq",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return zmq


def _print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True))


def _request(endpoint, payload, timeout_ms):
    zmq = _load_zmq()
    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.linger = 0
    socket.rcvtimeo = timeout_ms
    socket.sndtimeo = timeout_ms
    try:
        socket.connect(endpoint)
        socket.send_json(payload)
        return socket.recv_json()
    except zmq.Again as exc:
        raise TimeoutError(f"ZMQ request timed out after {timeout_ms} ms") from exc
    finally:
        socket.close()


def _listen_state(endpoint, timeout_ms):
    zmq = _load_zmq()
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.linger = 0
    socket.rcvtimeo = timeout_ms
    socket.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
    try:
        socket.connect(endpoint)
        while True:
            message = socket.recv_string()
            topic, payload = message.split(" ", 1)
            print(topic)
            _print_json(json.loads(payload))
    except KeyboardInterrupt:
        return
    except zmq.Again as exc:
        raise TimeoutError(f"No state message received after {timeout_ms} ms") from exc
    finally:
        socket.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Command-line client for GX16 ZMQ node.")
    parser.add_argument("--endpoint", default=DEFAULT_CMD_ENDPOINT, help="Command REQ endpoint.")
    parser.add_argument(
        "--state-endpoint",
        default=DEFAULT_STATE_ENDPOINT,
        help="State SUB endpoint, used by listen-state.",
    )
    parser.add_argument("--timeout-ms", type=int, default=2000)

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ping")
    subparsers.add_parser("status")
    subparsers.add_parser("home")
    subparsers.add_parser("torque_on")
    subparsers.add_parser("torque_off")
    subparsers.add_parser("shutdown")

    getjs = subparsers.add_parser("getjs")
    getjs.add_argument("--units", choices=VALID_UNITS, default="urdf_deg")

    setjs = subparsers.add_parser("setjs")
    setjs.add_argument("--positions", nargs=JOINT_COUNT, type=float, required=True)
    setjs.add_argument("--units", choices=VALID_UNITS, default="urdf_deg")

    setj = subparsers.add_parser("setj")
    setj.add_argument("--joint", type=int, required=True)
    setj.add_argument("--position", type=float, required=True)
    setj.add_argument("--units", choices=VALID_UNITS, default="urdf_deg")

    subparsers.add_parser("listen-state")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "listen-state":
        _listen_state(args.state_endpoint, args.timeout_ms)
        return 0

    if args.command in {"ping", "status", "home", "torque_on", "torque_off", "shutdown"}:
        payload = {"cmd": args.command}
    elif args.command == "getjs":
        payload = {"cmd": "getjs", "units": args.units}
    elif args.command == "setjs":
        payload = {"cmd": "setjs", "positions": args.positions, "units": args.units}
    elif args.command == "setj":
        payload = {
            "cmd": "setj",
            "joint": args.joint,
            "position": args.position,
            "units": args.units,
        }
    else:
        raise RuntimeError(f"unhandled command: {args.command}")

    response = _request(args.endpoint, payload, args.timeout_ms)
    _print_json(response)
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
