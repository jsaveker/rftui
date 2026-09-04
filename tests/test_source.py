from __future__ import annotations

import io
import os
import queue
import subprocess
import threading
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from rftui.model import (
    MAX_FRAME_BINS,
    DeviceInfo,
    SpectrumFrame,
    SweepConfig,
    plan_hackrf_fft,
)
from rftui.source import (
    DemoSpectrumSource,
    DeviceQueryError,
    HackRFSweepSource,
    SourceProcessError,
    SourceStatus,
    SweepParseError,
    ToolNotFoundError,
    build_sweep_argv,
    list_devices,
    parse_hackrf_info,
    parse_sweep_csv_line,
)

SAMPLE_ROW = (
    "2026-09-04, 12:34:56.123456, 902000000, 907000000, "
    "1000000.00, 20, -91.25, -88.00, -44.50, -86.75, -90.00\n"
)


def test_sweep_config_and_command_are_validated_argv() -> None:
    config = SweepConfig(
        start_hz=902_000_000,
        stop_hz=928_000_000,
        bin_width_hz=1_000_000,
        lna_gain=24,
        vga_gain=30,
        amp=True,
        serial="0000000000000000abc123",
    )

    assert build_sweep_argv(config, executable="/opt/HackRF Tools/hackrf_sweep") == [
        "/opt/HackRF Tools/hackrf_sweep",
        "-d",
        "0000000000000000abc123",
        "-f",
        "902:928",
        "-w",
        "1000000",
        "-l",
        "24",
        "-g",
        "30",
        "-a",
        "1",
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"start_hz": 902_500_000}, "whole MHz"),
        ({"stop_hz": 902_000_000}, "greater than"),
        ({"lna_gain": 17}, "8 dB steps"),
        ({"vga_gain": 21}, "2 dB steps"),
        ({"serial": "bad;serial"}, "serial"),
        (
            {"start_hz": 0, "stop_hz": 1_000_000_000, "bin_width_hz": 2_445},
            "memory limit",
        ),
    ],
)
def test_sweep_config_rejects_values_the_tool_cannot_honor(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SweepConfig(**kwargs)  # type: ignore[arg-type]


def test_official_fft_planner_rejects_adjusted_count_overflow() -> None:
    plan = plan_hackrf_fft(2_465)

    assert plan.fft_bin_count == 8_116
    assert plan.effective_bin_width_hz == pytest.approx(20_000_000 / 8_116)
    assert plan.csv_bin_width_hz == 2_464.27
    assert plan.bins_per_segment == 2_029
    assert plan.bin_count_for_span(323_000_000) == 131_074

    with pytest.raises(ValueError, match="131,072-bin memory limit"):
        SweepConfig(
            start_hz=1_000_000,
            stop_hz=324_000_000,
            bin_width_hz=2_465,
        )


def test_parse_real_hackrf_sweep_csv_shape() -> None:
    frame = parse_sweep_csv_line(SAMPLE_ROW)

    assert frame is not None
    assert frame.timestamp == datetime(2026, 9, 4, 12, 34, 56, 123456).astimezone()
    assert frame.frequencies_hz == (
        902_000_000.0,
        903_000_000.0,
        904_000_000.0,
        905_000_000.0,
        906_000_000.0,
    )
    assert frame.powers_db == (-91.25, -88.0, -44.5, -86.75, -90.0)
    assert frame.peak_frequency_hz == 904_000_000.0
    assert frame.peak_power_db == -44.5
    assert frame.bin_width_hz == 1_000_000.0


def test_parser_accepts_bom_and_timezone_and_blank_lines() -> None:
    row = "\ufeff2026-09-04, 12:34:56+00:00, 1000000, 3000000, 1000000, 8, -80, -70"
    frame = parse_sweep_csv_line(row)
    assert frame is not None
    assert frame.timestamp.tzinfo is UTC
    assert parse_sweep_csv_line(" \n") is None


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="platform has no POSIX timezone control")
def test_parser_resolves_local_offset_for_the_rows_season() -> None:
    previous_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/Chicago"
        time.tzset()
        winter = parse_sweep_csv_line(
            "2026-01-15, 12:00:00, 1000000, 3000000, 1000000, 8, -80, -70"
        )
        summer = parse_sweep_csv_line(
            "2026-07-15, 12:00:00, 1000000, 3000000, 1000000, 8, -80, -70"
        )
    finally:
        if previous_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_tz
        time.tzset()

    assert winter is not None and summer is not None
    assert winter.timestamp.utcoffset().total_seconds() == -6 * 60 * 60
    assert summer.timestamp.utcoffset().total_seconds() == -5 * 60 * 60


@pytest.mark.parametrize(
    "line",
    [
        "hackrf_sweep: device not found",
        "2026-09-04, bad-time, 1, 2, 1, 1, -70",
        "2026-09-04, 12:00:00, 2, 1, 1, 1, -70",
        "2026-09-04, 12:00:00, 1, 2, 1, 1, nan",
        "2026-09-04, 12:00:00, 1, 6, 1, 2, -70, -71",
        "2026-09-04, 12:00:00, 1, 2, 1, 1, -70, -71",
    ],
)
def test_parser_rejects_malformed_or_inconsistent_rows(line: str) -> None:
    with pytest.raises(SweepParseError):
        parse_sweep_csv_line(line)


def test_parser_enforces_per_row_bound() -> None:
    with pytest.raises(SweepParseError, match="bin limit"):
        parse_sweep_csv_line(SAMPLE_ROW, max_bins=4)


HACKRF_INFO_OUTPUT = """\
hackrf_info version: 2024.02.1
libhackrf version: 2024.02.1 (0.9)
Found HackRF
Index: 0
Serial number: 0000000000000000aaa
Board ID Number: 2 (HackRF One)
Firmware Version: 2024.02.1 (API:1.08)
Part ID Number: 0xa000cb3c 0x00544755
Found HackRF
Index: 1
Serial number: RunningFromRAM
Board ID Number: 4 (HackRF One r10)
Firmware Version: git-main (API:1.08)
Part ID Number: 0xa000cb3c 0x00644f56
"""


def test_parse_and_list_multiple_devices_without_shell() -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, HACKRF_INFO_OUTPUT, "")

    devices = list_devices(executable="/usr/local/bin/hackrf_info", runner=runner)

    assert devices == [
        DeviceInfo(
            index=0,
            serial_number="0000000000000000aaa",
            board_name="HackRF One",
            board_id=2,
            firmware_version="2024.02.1 (API:1.08)",
            part_id="0xa000cb3c 0x00544755",
        ),
        DeviceInfo(
            index=1,
            serial_number="RunningFromRAM",
            board_name="HackRF One r10",
            board_id=4,
            firmware_version="git-main (API:1.08)",
            part_id="0xa000cb3c 0x00644f56",
        ),
    ]
    assert calls[0][0] == ["/usr/local/bin/hackrf_info"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["timeout"] == 5.0


def test_device_discovery_accepts_complete_device_before_optional_metadata_failure() -> None:
    output = """\
hackrf_info version: 2026.01.3
libhackrf version: 2026.01.3 (0.9.2)
Found HackRF
Index: 0
Serial number: 000000000000000046d067dc231c6047
Board ID Number: 2 (HackRF One)
Firmware Version: v1.5.4 (API:1.06)
Part ID Number: 0xa000cb3c 0x00704755
"""

    def partial_probe(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            1,
            output,
            "hackrf_board_rev_read() failed: Pipe error (-1000)",
        )

    assert list_devices(runner=partial_probe) == [
        DeviceInfo(
            index=0,
            serial_number="000000000000000046d067dc231c6047",
            board_name="HackRF One",
            board_id=2,
            firmware_version="v1.5.4 (API:1.06)",
            part_id="0xa000cb3c 0x00704755",
        )
    ]


def test_device_discovery_distinguishes_no_device_missing_tool_and_failure() -> None:
    def no_device(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, "", "No HackRF boards found.")

    def missing(*_: Any, **__: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("missing")

    def broken(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 2, "", "libusb initialization failed")

    assert list_devices(runner=no_device) == []
    with pytest.raises(ToolNotFoundError, match="install the HackRF host tools"):
        list_devices(runner=missing)
    with pytest.raises(DeviceQueryError, match="libusb initialization failed"):
        list_devices(runner=broken)


def test_incomplete_device_blocks_are_ignored() -> None:
    assert parse_hackrf_info("Found HackRF\nIndex: 0\nBoard ID Number: 2 (HackRF One)\n") == []


class _QueueStream:
    def __init__(self, *lines: str) -> None:
        self._items: queue.Queue[str | None] = queue.Queue()
        self._closed = False
        for line in lines:
            self._items.put(line)

    def readline(self, _size: int = -1) -> str:
        item = self._items.get(timeout=2)
        return "" if item is None else item

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._items.put(None)


class _BlockingProcess:
    def __init__(self, *stdout_lines: str) -> None:
        self.stdout = _QueueStream(*stdout_lines)
        self.stderr = _QueueStream()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()
        self._done.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()
        self._done.set()

    def wait(self, timeout: float | None = None) -> int:
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired("hackrf_sweep", timeout)
        assert self.returncode is not None
        return self.returncode


def _sweep_row(low_hz: int, timestamp: str = "12:34:56.123456") -> str:
    high_hz = low_hz + 5_000_000
    return (
        f"2026-09-04, {timestamp}, {low_hz}, {high_hz}, 1000000.00, 20, -91, -88, -84, -80, -76\n"
    )


def test_live_source_waits_for_true_boundary_with_interleaved_row_order() -> None:
    # INTERLEAVED mode can report a lower edge late in the same sweep. Only a
    # repeat of the configured first edge marks the next sweep.
    process = _BlockingProcess(
        _sweep_row(902_000_000),
        _sweep_row(912_000_000, "12:34:56.223456"),
        _sweep_row(907_000_000, "12:34:56.323456"),
        _sweep_row(917_000_000, "12:34:56.423456"),
        _sweep_row(922_000_000, "12:34:56.523456"),
        _sweep_row(932_000_000, "12:34:56.623456"),
        _sweep_row(927_000_000, "12:34:56.723456"),
        _sweep_row(937_000_000, "12:34:56.823456"),
        _sweep_row(902_000_000, "12:34:57.123456"),
    )
    frames: list[SpectrumFrame] = []
    received = threading.Event()

    def callback(frame: SpectrumFrame) -> None:
        frames.append(frame)
        received.set()

    source = HackRFSweepSource(
        SweepConfig(),
        popen_factory=lambda *_args, **_kwargs: process,
    )
    source.start(callback)

    assert received.wait(2)
    source.stop()
    assert len(frames) == 1
    assert frames[0].bin_count == 26
    assert frames[0].frequencies_hz == tuple(
        float(frequency) for frequency in range(902_000_000, 928_000_000, 1_000_000)
    )


def test_live_source_constructs_command_skips_bad_rows_and_terminates_cleanly() -> None:
    process = _BlockingProcess("not,csv\n", SAMPLE_ROW, SAMPLE_ROW)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen(argv: list[str], **kwargs: Any) -> _BlockingProcess:
        calls.append((argv, kwargs))
        return process

    frames: list[SpectrumFrame] = []
    received = threading.Event()

    def callback(frame: SpectrumFrame) -> None:
        frames.append(frame)
        received.set()

    source = HackRFSweepSource(SweepConfig(), popen_factory=popen)
    source.start(callback)
    source.start(callback)  # idempotent and callback-deduplicated

    assert received.wait(2)
    assert source.malformed_lines == 1
    assert len(frames) == 1
    assert frames[0].bin_count == 5
    assert calls[0][1]["shell"] is False

    source.stop()
    assert process.terminated
    assert not process.killed
    assert source.status is SourceStatus.STOPPED
    assert not source.running


class _FinishedProcess:
    def __init__(self, returncode: int, stderr: str) -> None:
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:  # pragma: no cover - must not be called
        raise AssertionError("finished process should not be terminated")

    def kill(self) -> None:  # pragma: no cover - must not be called
        raise AssertionError("finished process should not be killed")

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class _UnstoppableStream:
    def __init__(self) -> None:
        self.release = threading.Event()

    def readline(self, _size: int = -1) -> str:
        self.release.wait(2)
        return ""

    def close(self) -> None:
        pass


class _UnstoppableProcess:
    def __init__(self) -> None:
        self.stdout = _UnstoppableStream()
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.terminated = 0
        self.killed = 0

    def poll(self) -> int | None:
        if self.stdout.release.is_set():
            self.returncode = -9
        return self.returncode

    def terminate(self) -> None:
        self.terminated += 1

    def kill(self) -> None:
        self.killed += 1

    def wait(self, timeout: float | None = None) -> int:
        if not self.stdout.release.wait(timeout):
            raise subprocess.TimeoutExpired("hackrf_sweep", timeout)
        self.returncode = -9
        return self.returncode


def test_restart_fails_if_old_reader_survives_terminate_and_kill() -> None:
    process = _UnstoppableProcess()
    source = HackRFSweepSource(
        SweepConfig(),
        termination_timeout_s=0.01,
        popen_factory=lambda *_args, **_kwargs: process,
    )
    source.start()
    assert _wait_until(lambda: source.status is SourceStatus.RUNNING)

    try:
        with pytest.raises(SourceProcessError, match="did not stop"):
            source.stop()
        assert source.status is SourceStatus.FAILED
        with pytest.raises(SourceProcessError, match="did not stop"):
            source.restart()
        assert process.terminated >= 1
        assert process.killed >= 1
    finally:
        process.stdout.release.set()


def test_live_source_surfaces_process_failure_with_bounded_stderr() -> None:
    process = _FinishedProcess(1, "hackrf_open() failed: HACKRF_ERROR_BUSY\n")
    source = HackRFSweepSource(SweepConfig(), popen_factory=lambda *_args, **_kwargs: process)

    source.start()
    assert _wait_until(lambda: source.status is SourceStatus.FAILED)

    assert isinstance(source.error, SourceProcessError)
    assert "HACKRF_ERROR_BUSY" in str(source.error)
    assert source.stderr_tail == ("hackrf_open() failed: HACKRF_ERROR_BUSY",)


def test_demo_frames_are_repeatable_bounded_and_match_source_contract() -> None:
    timestamp = datetime(2026, 9, 4, tzinfo=UTC)
    config = SweepConfig(bin_width_hz=1_000_000)
    first = DemoSpectrumSource(config, seed=7)
    second = DemoSpectrumSource(config, seed=7)

    frame_a = first.next_frame(timestamp=timestamp)
    frame_b = second.next_frame(timestamp=timestamp)

    assert frame_a == frame_b
    assert frame_a.bin_count == config.estimated_bin_count
    assert frame_a.bin_count <= MAX_FRAME_BINS
    assert config.start_hz == frame_a.frequencies_hz[0] < frame_a.frequencies_hz[-1]
    assert frame_a.frequencies_hz[-1] < config.stop_hz
    assert frame_a.bin_width_hz == pytest.approx(config.csv_bin_width_hz)
    assert frame_a.floor_power_db < frame_a.peak_power_db


def test_demo_callback_failures_are_isolated_and_reconfigure_restarts() -> None:
    config = SweepConfig()
    source = DemoSpectrumSource(config, interval_s=0.01)
    received: list[SpectrumFrame] = []
    event = threading.Event()

    def broken_callback(_frame: SpectrumFrame) -> None:
        raise RuntimeError("consumer bug")

    def good_callback(frame: SpectrumFrame) -> None:
        received.append(frame)
        event.set()

    source.add_callback(broken_callback)
    source.start(good_callback)
    assert event.wait(1)
    assert source.callback_errors >= 1

    event.clear()
    new_config = SweepConfig(start_hz=433_000_000, stop_hz=435_000_000)
    source.reconfigure(new_config)
    assert event.wait(1)
    source.stop()

    assert source.config == new_config
    assert any(frame.frequencies_hz[0] < 435_000_000 for frame in received)
    assert source.status is SourceStatus.STOPPED


def _wait_until(predicate: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())
