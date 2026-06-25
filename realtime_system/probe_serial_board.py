"""
Probe the custom CP210x serial EEG board.

This script does not decode EEG values yet. It only verifies that the board
responds on the serial port and emits 19-byte frames:

    frame[0]  == 0xA0
    frame[-1] == 0xC0

Usage:
    python realtime_system/probe_serial_board.py --port COM5
"""

from __future__ import annotations

import argparse
import time

import serial


FRAME_LEN = 19
FRAME_HEAD = 0xA0
FRAME_TAIL = 0xC0


def iter_frames(ser: serial.Serial):
    buffer = bytearray()

    while True:
        chunk = ser.read(256)
        if not chunk:
            yield None
            continue

        buffer.extend(chunk)

        while True:
            try:
                start = buffer.index(FRAME_HEAD)
            except ValueError:
                buffer.clear()
                break

            if start:
                del buffer[:start]

            if len(buffer) < FRAME_LEN:
                break

            frame = bytes(buffer[:FRAME_LEN])
            del buffer[:FRAME_LEN]

            if frame[-1] == FRAME_TAIL:
                yield frame
            else:
                # Lost alignment. Keep searching from the next byte.
                buffer[:0] = frame[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe CP210x serial EEG board frames.")
    parser.add_argument("--port", default="COM5", help="Serial port, for example COM5.")
    parser.add_argument("--baud", type=int, default=230400, help="Baud rate.")
    parser.add_argument("--seconds", type=float, default=10.0, help="How long to capture.")
    args = parser.parse_args()

    with serial.Serial(args.port, args.baud, bytesize=8, parity="N", stopbits=1, timeout=0.5) as ser:
        ser.reset_input_buffer()
        ser.write(b"b")
        ser.flush()
        print(f"Started capture on {args.port} @ {args.baud}. Sent ASCII 'b'.")

        start = time.monotonic()
        frame_count = 0
        last_report = start

        try:
            for frame in iter_frames(ser):
                now = time.monotonic()
                if now - start >= args.seconds:
                    break

                if frame is None:
                    continue

                frame_count += 1
                print(f"{frame_count:05d}: {frame.hex(' ').upper()}")

                if now - last_report >= 1.0:
                    elapsed = now - start
                    print(f"rate: {frame_count / elapsed:.1f} frames/s")
                    last_report = now
        finally:
            ser.write(b"s")
            ser.flush()
            print("Sent ASCII 's' to stop capture.")

    elapsed = max(time.monotonic() - start, 1e-9)
    print(f"Captured {frame_count} valid frames in {elapsed:.1f}s ({frame_count / elapsed:.1f} frames/s).")


if __name__ == "__main__":
    main()
