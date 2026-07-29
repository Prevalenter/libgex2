from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import serial

TX_HEAD = b"\x55\xAA"
RX_HEAD = b"\xAA\x55"
HEAD_LEN = 13

DEFAULT_BAUDRATE = 921600
DEFAULT_DEVICE_ID = 0x03
READ_FUNC = 0x7B
WRITE_FUNC = 0x79

TACTILE_POINTS = 135
TACTILE_AXES = 3
TACTILE_RAW_BYTES = TACTILE_POINTS * TACTILE_AXES
TACTILE_ADDR = 0x040E
SUM_FORCE_ADDR = 0x03F0
MAX_READ_BYTES = 0xFF


def lrc_cal(data: bytes) -> int:
    total = 0
    for value in data:
        total = (total + value) & 0xFF
    return ((~total + 1) & 0xFF)


def hexdump(data: bytes | Iterable[int]) -> str:
    return bytes(data).hex(" ").upper()


@dataclass(frozen=True)
class Frame:
    raw: bytes
    dev_id: int
    func: int
    addr: int
    data_len: int
    payload: bytes


def build_read_request(
    device_id: int,
    func_code: int,
    addr: int,
    nbytes: int,
    reserve: int = 0x00,
) -> bytes:
    func = (func_code | 0x80) & 0xFF
    length_field = 9
    buf = bytearray()
    buf += TX_HEAD
    buf += struct.pack("<H", length_field)
    buf += struct.pack("<B", device_id)
    buf += struct.pack("<B", reserve)
    buf += struct.pack("<B", func)
    buf += struct.pack("<I", addr)
    buf += struct.pack("<H", nbytes)
    buf += bytes([lrc_cal(buf)])
    return bytes(buf)


def build_write_request(
    device_id: int,
    addr: int,
    payload: bytes,
    func_code: int = WRITE_FUNC,
    reserve: int = 0x00,
) -> bytes:
    func = func_code & 0x7F
    length_field = 9 + len(payload)
    buf = bytearray()
    buf += TX_HEAD
    buf += struct.pack("<H", length_field)
    buf += struct.pack("<B", device_id)
    buf += struct.pack("<B", reserve)
    buf += struct.pack("<B", func)
    buf += struct.pack("<I", addr)
    buf += struct.pack("<H", len(payload))
    buf += payload
    buf += bytes([lrc_cal(buf)])
    return bytes(buf)


def parse_frame(frame_raw: bytes) -> Frame:
    if len(frame_raw) < HEAD_LEN + 1:
        raise ValueError("frame is too short")
    if not frame_raw.startswith(RX_HEAD):
        raise ValueError("frame does not start with AA55")

    length_field = frame_raw[2] | (frame_raw[3] << 8)
    total_len = 2 + 2 + length_field + 1
    if len(frame_raw) != total_len:
        raise ValueError(f"expected {total_len} bytes, got {len(frame_raw)}")
    if lrc_cal(frame_raw[:-1]) != frame_raw[-1]:
        raise ValueError("invalid frame LRC")

    return Frame(
        raw=frame_raw,
        dev_id=frame_raw[4],
        func=frame_raw[6],
        addr=struct.unpack_from("<I", frame_raw, 7)[0],
        data_len=struct.unpack_from("<H", frame_raw, 11)[0],
        payload=frame_raw[HEAD_LEN:-1],
    )


def raw_to_force_frame(raw: np.ndarray | bytes, n_points: int = TACTILE_POINTS) -> np.ndarray:
    if isinstance(raw, bytes):
        data = np.frombuffer(raw, dtype=np.uint8)
    else:
        data = np.asarray(raw, dtype=np.uint8)
    expected = n_points * TACTILE_AXES
    if data.size != expected:
        raise ValueError(f"expected {expected} values, got {data.size}")
    return data.reshape(n_points, TACTILE_AXES).astype(np.float32)


class Gen3UARTClient:
    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        device_id: int = DEFAULT_DEVICE_ID,
        timeout: float = 0.05,
        boot_delay: float = 0.8,
        set_dtr_rts_low: bool = True,
        debug: bool = False,
    ):
        self.device_id = device_id
        self.debug = debug
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            rtscts=False,
            dsrdtr=False,
        )
        if set_dtr_rts_low:
            try:
                self.ser.dtr = False
                self.ser.rts = False
            except Exception:
                pass
        time.sleep(boot_delay)
        self.ser.reset_input_buffer()
        self._rx_buf = bytearray()

    def __enter__(self) -> "Gen3UARTClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.ser.close()

    def _build_read_request(self, func_code: int, addr: int, nbytes: int) -> bytes:
        return build_read_request(self.device_id, func_code, addr, nbytes)

    def _read_one_frame(self, timeout_s: float = 2.0) -> Frame:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            chunk = self.ser.read(self.ser.in_waiting or 1)
            if chunk:
                self._rx_buf += chunk

            idx = self._rx_buf.find(RX_HEAD)
            if idx < 0:
                if len(self._rx_buf) > 1:
                    self._rx_buf = self._rx_buf[-1:]
                continue
            if idx > 0:
                self._rx_buf = self._rx_buf[idx:]
            if len(self._rx_buf) < 4:
                continue

            length_field = self._rx_buf[2] | (self._rx_buf[3] << 8)
            total_len = 2 + 2 + length_field + 1
            if len(self._rx_buf) < total_len:
                continue

            frame_raw = bytes(self._rx_buf[:total_len])
            self._rx_buf = self._rx_buf[total_len:]
            try:
                return parse_frame(frame_raw)
            except ValueError:
                continue

        raise TimeoutError("timeout waiting for a valid RX frame")

    def read(self, func_code: int, addr: int, nbytes: int, timeout_s: float = 2.0) -> bytes:
        req = self._build_read_request(func_code, addr, nbytes)
        if self.debug:
            print("TX:", hexdump(req))

        self.ser.reset_input_buffer()
        self._rx_buf.clear()
        self.ser.write(req)
        self.ser.flush()

        expected_func = (func_code | 0x80) & 0xFF
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                frame = self._read_one_frame(max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                break
            if frame.dev_id != self.device_id or frame.func != expected_func:
                continue
            if not frame.payload:
                raise RuntimeError("read payload is missing status byte")

            status = frame.payload[0]
            if status != 0x00:
                raise RuntimeError(f"read failed, status=0x{status:02X}")

            data = frame.payload[1 : 1 + nbytes]
            if len(data) != nbytes:
                raise RuntimeError(f"expected {nbytes} bytes, got {len(data)}")
            return data

        raise TimeoutError(f"timeout waiting for func=0x{expected_func:02X}")

    def write_no_response(self, addr: int, payload: bytes, func_code: int = WRITE_FUNC) -> None:
        req = build_write_request(self.device_id, addr, payload, func_code=func_code)
        if self.debug:
            print("TX:", hexdump(req))
        self.ser.write(req)
        self.ser.flush()


class L3530(Gen3UARTClient):
    def read_tactile_raw(self) -> np.ndarray:
        chunks = []
        offset = 0
        while offset < TACTILE_RAW_BYTES:
            nbytes = min(MAX_READ_BYTES, TACTILE_RAW_BYTES - offset)
            data = self.read(READ_FUNC, TACTILE_ADDR + offset, nbytes, timeout_s=2.0)
            chunks.append(np.frombuffer(data, dtype=np.uint8))
            offset += nbytes
        return np.concatenate(chunks)

    def read_force_frame(self) -> np.ndarray:
        return raw_to_force_frame(self.read_tactile_raw())

    def read_sum_force(self) -> np.ndarray:
        data = self.read(READ_FUNC, SUM_FORCE_ADDR, 0x03, timeout_s=2.0)
        return np.frombuffer(data, dtype=np.uint8)


def scan_device_ids(
    port: str,
    device_ids: Iterable[int] = range(1, 7),
    start_addr: int = TACTILE_ADDR,
    read_len: int = 0x20,
    baudrate: int = DEFAULT_BAUDRATE,
    timeout_s: float = 0.8,
    send_setup: bool = True,
) -> list[tuple[int, bytes | None, str | None]]:
    results = []
    with Gen3UARTClient(port, baudrate=baudrate, device_id=DEFAULT_DEVICE_ID, boot_delay=0.2) as client:
        for device_id in device_ids:
            client.device_id = device_id
            client.ser.reset_input_buffer()
            client._rx_buf.clear()
            try:
                if send_setup:
                    client.write_no_response(0x0000, b"\x02")
                    time.sleep(0.02)
                data = client.read(READ_FUNC, start_addr, read_len, timeout_s=timeout_s)
                results.append((device_id, data, None))
            except Exception as exc:
                results.append((device_id, None, str(exc)))
    return results
