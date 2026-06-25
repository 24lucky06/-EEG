"""
Raw serial byte probe for the custom EEG board.

Use this when probe_serial_board.py reports 0 valid frames. This script does
not assume the A0...C0 19-byte frame format. It only checks whether COM is
emitting bytes and prints short hex previews.
"""

from __future__ import annotations

import argparse
import time

import serial


def parse_bauds(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def probe_once(port: str, baud: int, seconds: float, command: bytes | None) -> int:
    total = 0
    previews = 0
    start = time.monotonic()
    last_report = start

    print(f"\n--- Raw probe {port} @ {baud}, {seconds:g}s ---")
    with serial.Serial(port, baud, bytesize=8, parity="N", stopbits=1, timeout=0.1) as ser:
        ser.reset_input_buffer()
        if command:
            ser.write(command)
            ser.flush()
            print(f"Sent command bytes: {command!r}")
        while time.monotonic() - start < seconds:
            chunk = ser.read(512)
            now = time.monotonic()
            if chunk:
                total += len(chunk)
                if previews < 12:
                    print(f"{len(chunk):04d} bytes: {chunk[:64].hex(' ').upper()}")
                    previews += 1
            if now - last_report >= 1.0:
                elapsed = max(now - start, 1e-9)
                print(f"raw rate: {total / elapsed:.1f} bytes/s, total={total}")
                last_report = now

    elapsed = max(time.monotonic() - start, 1e-9)
    print(f"TOTAL raw bytes: {total} in {elapsed:.1f}s ({total / elapsed:.1f} bytes/s)")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe raw serial bytes without frame assumptions.")
    parser.add_argument("--port", default="COM5")
    parser.add_argument("--bauds", default="230400,115200,460800,921600")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--command",
        default="b",
        help="ASCII command sent before capture. Use empty string to send nothing.",
    )
    args = parser.parse_args()

    command = args.command.encode("ascii") if args.command else None
    for baud in parse_bauds(args.bauds):
        try:
            probe_once(args.port, baud, args.seconds, command)
        except Exception as exc:
            print(f"ERROR on {args.port} @ {baud}: {exc}")


if __name__ == "__main__":
    main()
