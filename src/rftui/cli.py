"""Command-line entry point for RFTUI."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from typing import TextIO

from .app import RFTUIApp
from .model import SweepConfig
from .source import (
    DemoSpectrumSource,
    HackRFSweepSource,
    SourceError,
    ToolNotFoundError,
    list_devices,
)

DEFAULT_START_MHZ = 902.0
DEFAULT_STOP_MHZ = 928.0
DEFAULT_BIN_WIDTH_HZ = 100_000

MIN_FREQUENCY_MHZ = 1.0
MAX_FREQUENCY_MHZ = 6_000.0
MIN_BIN_WIDTH_HZ = 2_445
MAX_BIN_WIDTH_HZ = 5_000_000
SERIAL_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


def build_parser() -> argparse.ArgumentParser:
    """Build the RFTUI argument parser."""

    parser = argparse.ArgumentParser(
        prog="rftui",
        description="Receive-only spectrum and waterfall cockpit. Current backend: HackRF.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run a deterministic simulated spectrum without HackRF hardware",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="list connected HackRF devices and exit",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=DEFAULT_START_MHZ,
        metavar="MHZ",
        help=f"sweep start frequency in MHz (default: {DEFAULT_START_MHZ:g})",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=DEFAULT_STOP_MHZ,
        metavar="MHZ",
        help=f"sweep stop frequency in MHz (default: {DEFAULT_STOP_MHZ:g})",
    )
    parser.add_argument(
        "--serial",
        metavar="SERIAL",
        help="select a connected HackRF by serial number",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        default=DEFAULT_BIN_WIDTH_HZ,
        metavar="HZ",
        help=f"spectrum bin width in Hz (default: {DEFAULT_BIN_WIDTH_HZ})",
    )
    parser.add_argument(
        "--lna-gain",
        type=int,
        default=16,
        metavar="DB",
        help="RF/LNA gain in dB, from 0 to 40 in steps of 8 (default: 16)",
    )
    parser.add_argument(
        "--vga-gain",
        type=int,
        default=20,
        metavar="DB",
        help="baseband/VGA gain in dB, from 0 to 62 in steps of 2 (default: 20)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if not math.isfinite(args.start) or not math.isfinite(args.stop):
        parser.error("--start and --stop must be finite numbers")
    if not MIN_FREQUENCY_MHZ <= args.start <= MAX_FREQUENCY_MHZ:
        parser.error(f"--start must be between {MIN_FREQUENCY_MHZ:g} and {MAX_FREQUENCY_MHZ:g} MHz")
    if not MIN_FREQUENCY_MHZ <= args.stop <= MAX_FREQUENCY_MHZ:
        parser.error(f"--stop must be between {MIN_FREQUENCY_MHZ:g} and {MAX_FREQUENCY_MHZ:g} MHz")
    if args.start >= args.stop:
        parser.error("--start must be lower than --stop")
    if not args.start.is_integer() or not args.stop.is_integer():
        parser.error("--start and --stop must be whole MHz values")
    if not MIN_BIN_WIDTH_HZ <= args.bin_width <= MAX_BIN_WIDTH_HZ:
        parser.error(f"--bin-width must be between {MIN_BIN_WIDTH_HZ} and {MAX_BIN_WIDTH_HZ} Hz")
    if not 0 <= args.lna_gain <= 40 or args.lna_gain % 8:
        parser.error("--lna-gain must be from 0 to 40 dB in steps of 8")
    if not 0 <= args.vga_gain <= 62 or args.vga_gain % 2:
        parser.error("--vga-gain must be from 0 to 62 dB in steps of 2")
    if args.demo and args.serial:
        parser.error("--serial cannot be used with --demo")
    if args.serial is not None and not SERIAL_PATTERN.fullmatch(args.serial):
        parser.error("--serial may contain only letters, numbers, '.', '_', ':', or '-'")

    return args


def _format_device(device: object) -> str:
    """Return a concise, human-readable line for a discovered radio."""

    index = getattr(device, "index", "?")
    board_name = getattr(device, "board_name", None) or "HackRF"
    serial = getattr(device, "serial_number", None) or getattr(device, "serial", None)
    firmware = getattr(device, "firmware_version", None)

    details = [f"serial {serial or 'unknown'}"]
    if firmware:
        details.append(f"firmware {firmware}")
    return f"[{index}] {board_name} ({', '.join(details)})"


def _print_devices(devices: Sequence[object], stream: TextIO) -> None:
    if not devices:
        print("No HackRF devices found.", file=stream)
        return

    print(f"Found {len(devices)} HackRF device{'s' if len(devices) != 1 else ''}:", file=stream)
    for device in devices:
        print(_format_device(device), file=stream)


def _serial_of(device: object) -> str | None:
    serial = getattr(device, "serial_number", None) or getattr(device, "serial", None)
    return str(serial) if serial is not None else None


def _select_device(devices: Sequence[object], serial: str | None) -> object | None:
    if not devices:
        return None
    if serial is None:
        return devices[0]

    wanted = serial.casefold()
    return next(
        (device for device in devices if (_serial_of(device) or "").casefold() == wanted),
        None,
    )


def _missing_tool_message(error: BaseException) -> str:
    detail = str(error).strip()
    suffix = f" ({detail})" if detail and "hackrf" not in detail.casefold() else ""
    return (
        "HackRF host tools were not found. Install them first "
        "(on macOS: `brew install hackrf`)."
        f"{suffix}"
    )


def _no_device_message() -> str:
    return (
        "No HackRF device was found. Check its USB connection and permissions. "
        "If a PortaPack is attached, select HackRF mode on the PortaPack first."
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run RFTUI and return a process exit status."""

    args = parse_args(argv)

    if args.list_devices:
        try:
            devices = list_devices()
        except (ToolNotFoundError, FileNotFoundError) as error:
            print(f"rftui: error: {_missing_tool_message(error)}", file=sys.stderr)
            return 1
        except SourceError as error:
            print(f"rftui: error: could not inspect HackRF devices: {error}", file=sys.stderr)
            return 1

        _print_devices(devices, sys.stdout)
        return 0 if devices else 1

    device = None
    if not args.demo:
        try:
            devices = list_devices()
        except (ToolNotFoundError, FileNotFoundError) as error:
            print(f"rftui: error: {_missing_tool_message(error)}", file=sys.stderr)
            return 1
        except SourceError as error:
            print(f"rftui: error: could not inspect HackRF devices: {error}", file=sys.stderr)
            return 1

        if not devices:
            print(f"rftui: error: {_no_device_message()}", file=sys.stderr)
            return 1
        device = _select_device(devices, args.serial)
        if device is None:
            available = ", ".join(filter(None, (_serial_of(item) for item in devices)))
            suffix = f" Available serials: {available}." if available else ""
            print(
                f"rftui: error: no connected HackRF has serial {args.serial!r}.{suffix}",
                file=sys.stderr,
            )
            return 1

    selected_serial = _serial_of(device) if device is not None else None
    try:
        config = SweepConfig(
            start_hz=round(args.start * 1_000_000),
            stop_hz=round(args.stop * 1_000_000),
            bin_width_hz=args.bin_width,
            lna_gain=args.lna_gain,
            vga_gain=args.vga_gain,
            serial=selected_serial,
        )
    except ValueError as error:
        print(f"rftui: error: invalid sweep configuration: {error}", file=sys.stderr)
        return 2

    source = DemoSpectrumSource(config) if args.demo else HackRFSweepSource(config)
    app = RFTUIApp(source, config, device=device, demo=args.demo)
    try:
        app.run()
    except KeyboardInterrupt:
        return 130
    return 0


__all__ = ["build_parser", "main", "parse_args"]
