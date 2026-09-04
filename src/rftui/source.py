"""Receive-only HackRF acquisition and deterministic demo sources."""

from __future__ import annotations

import csv
import io
import math
import random
import re
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Protocol, Self, TextIO

from .model import (
    HACKRF_SWEEP_SEGMENT_HZ,
    MAX_FRAME_BINS,
    MAX_FREQUENCY_HZ,
    DeviceInfo,
    SpectrumFrame,
    SweepConfig,
)

MAX_CSV_LINE_CHARS = 1_048_576
MAX_BINS_PER_ROW = 32_768
MAX_DEVICE_OUTPUT_CHARS = 262_144
MAX_DEVICES = 32
MAX_CALLBACKS = 32
STDERR_LINES = 40
STDERR_LINE_CHARS = 2_048

FrameCallback = Callable[[SpectrumFrame], None]


class SourceError(RuntimeError):
    """Base class for source and HackRF tool failures."""


class ToolNotFoundError(SourceError):
    """A required HackRF host tool was not installed or discoverable."""


class DeviceQueryError(SourceError):
    """``hackrf_info`` could not query attached devices."""


class SourceProcessError(SourceError):
    """``hackrf_sweep`` failed after it was launched."""


class SweepParseError(ValueError):
    """A non-empty ``hackrf_sweep`` CSV row was invalid."""


class SourceStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value


class SpectrumSource(Protocol):
    """Small source contract consumed by the TUI."""

    @property
    def config(self) -> SweepConfig: ...

    @property
    def running(self) -> bool: ...

    def start(self, callback: FrameCallback | None = None) -> None: ...

    def stop(self, timeout: float | None = None) -> None: ...

    def restart(self, callback: FrameCallback | None = None) -> None: ...

    def reconfigure(self, config: SweepConfig) -> None: ...


def build_sweep_argv(
    config: SweepConfig,
    *,
    executable: str = "hackrf_sweep",
) -> list[str]:
    """Build a validated argv list; no shell parsing is involved."""

    if not executable or "\x00" in executable:
        raise ValueError("executable must be a non-empty path without NUL bytes")
    argv = [executable]
    if config.serial:
        argv.extend(("-d", config.serial))
    argv.extend(
        (
            "-f",
            f"{config.start_mhz}:{config.stop_mhz}",
            "-w",
            str(config.bin_width_hz),
            "-l",
            str(config.lna_gain),
            "-g",
            str(config.vga_gain),
            "-a",
            "1" if config.amp else "0",
        )
    )
    return argv


def parse_hackrf_info(output: str) -> list[DeviceInfo]:
    """Parse one or more device blocks from ``hackrf_info`` output."""

    if len(output) > MAX_DEVICE_OUTPUT_CHARS:
        raise DeviceQueryError("hackrf_info output exceeded the safety limit")

    devices: list[DeviceInfo] = []
    current: dict[str, str] | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        serial = current.get("Serial number")
        if not serial:
            current = None
            return
        index_text = current.get("Index", str(len(devices)))
        try:
            index = int(index_text)
        except ValueError:
            index = len(devices)
        board_text = current.get("Board ID Number", "")
        board_match = re.fullmatch(r"\s*(\d+)(?:\s*\((.*?)\))?\s*", board_text)
        board_id = int(board_match.group(1)) if board_match else None
        board_name = board_match.group(2).strip() if board_match and board_match.group(2) else None
        devices.append(
            DeviceInfo(
                index=index,
                serial_number=serial.strip(),
                board_name=board_name,
                board_id=board_id,
                firmware_version=current.get("Firmware Version"),
                part_id=current.get("Part ID Number"),
            )
        )
        current = None

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line == "Found HackRF":
            finish()
            if len(devices) >= MAX_DEVICES:
                raise DeviceQueryError(f"hackrf_info reported more than {MAX_DEVICES} devices")
            current = {}
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        current[key.strip()] = value.strip()
    finish()
    return devices


def list_devices(
    *,
    executable: str = "hackrf_info",
    timeout_s: float = 5.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> list[DeviceInfo]:
    """Return attached HackRF devices without opening a receive stream."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if not executable or "\x00" in executable:
        raise ValueError("executable must be a non-empty path without NUL bytes")
    run = runner or subprocess.run
    try:
        result = run(
            [executable],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ToolNotFoundError(
            f"{executable!r} was not found; install the HackRF host tools"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DeviceQueryError(f"{executable!r} did not finish within {timeout_s:g}s") from exc
    except OSError as exc:
        raise DeviceQueryError(f"could not run {executable!r}: {exc}") from exc

    stdout = _coerce_text(result.stdout)
    stderr = _coerce_text(result.stderr)
    combined = "\n".join(part for part in (stdout, stderr) if part)
    devices = parse_hackrf_info(combined)
    if result.returncode == 0:
        return devices
    lowered = combined.lower()
    no_device_markers = (
        "no hackrf boards found",
        "no hackrf devices found",
        "no devices found",
        "hackrf_error_not_found",
    )
    if not devices and any(marker in lowered for marker in no_device_markers):
        return []
    detail = _last_nonempty_line(stderr or stdout) or "unknown error"
    raise DeviceQueryError(f"{executable!r} exited with status {result.returncode}: {detail}")


@dataclass(frozen=True, slots=True)
class _SweepRow:
    timestamp: datetime
    low_hz: int
    high_hz: int
    bin_width_hz: float
    powers_db: tuple[float, ...]

    @property
    def frequencies_hz(self) -> tuple[float, ...]:
        return tuple(
            self.low_hz + index * self.bin_width_hz for index in range(len(self.powers_db))
        )

    def as_frame(self) -> SpectrumFrame:
        return SpectrumFrame(self.timestamp, self.frequencies_hz, self.powers_db)


def parse_sweep_csv_line(
    line: str,
    *,
    max_bins: int = MAX_BINS_PER_ROW,
) -> SpectrumFrame | None:
    """Parse one documented ``hackrf_sweep`` CSV row.

    Blank lines return ``None``. Other malformed rows raise ``SweepParseError``
    so callers can count and skip them without confusing diagnostics for data.
    """

    row = _parse_sweep_row(line, max_bins=max_bins)
    return None if row is None else row.as_frame()


# Short alias retained for callers that do not include the transport in the name.
parse_sweep_line = parse_sweep_csv_line


def _parse_sweep_row(line: str, *, max_bins: int = MAX_BINS_PER_ROW) -> _SweepRow | None:
    if not isinstance(line, str):
        raise SweepParseError("sweep row must be text")
    if not line.strip():
        return None
    if max_bins < 1 or max_bins > MAX_FRAME_BINS:
        raise ValueError(f"max_bins must be between 1 and {MAX_FRAME_BINS}")
    if len(line) > MAX_CSV_LINE_CHARS:
        raise SweepParseError("sweep row exceeded the input size limit")
    try:
        fields = next(csv.reader([line], skipinitialspace=True, strict=True))
    except (csv.Error, StopIteration) as exc:
        raise SweepParseError("invalid CSV") from exc
    if len(fields) < 7:
        raise SweepParseError("expected six metadata fields and at least one power bin")
    if len(fields) - 6 > max_bins:
        raise SweepParseError(f"sweep row exceeded the {max_bins}-bin limit")

    date_text = fields[0].strip().lstrip("\ufeff")
    time_text = fields[1].strip()
    try:
        timestamp = datetime.fromisoformat(f"{date_text}T{time_text}")
    except ValueError as exc:
        raise SweepParseError("invalid sweep timestamp") from exc
    if timestamp.tzinfo is None:
        # A no-argument astimezone() interprets this wall time using the OS
        # rules for the row's date, including historical/seasonal DST.
        timestamp = timestamp.astimezone()

    low_hz = _parse_integral(fields[2], "hz_low")
    high_hz = _parse_integral(fields[3], "hz_high")
    bin_width_hz = _parse_finite_float(fields[4], "hz_bin_width")
    sample_count = _parse_integral(fields[5], "num_samples")
    if low_hz < 0 or high_hz <= low_hz or high_hz > MAX_FREQUENCY_HZ + 20_000_000:
        raise SweepParseError("invalid sweep frequency bounds")
    if bin_width_hz <= 0:
        raise SweepParseError("hz_bin_width must be positive")
    if sample_count <= 0 or sample_count > 10_000_000:
        raise SweepParseError("num_samples is outside the safety limit")

    powers = tuple(_parse_finite_float(value, "power") for value in fields[6:])
    if len(powers) > sample_count:
        raise SweepParseError("power bin count exceeds num_samples")
    covered_hz = len(powers) * bin_width_hz
    span_hz = high_hz - low_hz
    # The tool prints bin width with two decimal places, so allow accumulated
    # formatting error but reject rows whose bins materially under/over-cover.
    tolerance_hz = max(1.0, len(powers) * 0.006, span_hz * 0.00001)
    if abs(covered_hz - span_hz) > tolerance_hz:
        raise SweepParseError("power bins do not match the reported frequency span")
    return _SweepRow(timestamp, low_hz, high_hz, bin_width_hz, powers)


class _SweepAssembler:
    """Collect bounded CSV segments until HackRF reports the first edge again."""

    def __init__(self, config: SweepConfig) -> None:
        self._config = config
        self._points: dict[float, float] = {}
        self._timestamp: datetime | None = None

    def add(self, row: _SweepRow) -> SpectrumFrame | None:
        completed: SpectrumFrame | None = None
        # INTERLEAVED mode does not guarantee monotonically increasing row
        # edges. hackrf_sweep itself counts a new sweep only when the hardware
        # reports the configured first tuning frequency again.
        if self._points and row.low_hz == self._config.start_hz:
            completed = self.flush()
        if self._timestamp is None:
            self._timestamp = row.timestamp
        for frequency, power in zip(row.frequencies_hz, row.powers_db):
            if self._config.start_hz <= frequency < self._config.stop_hz:
                if frequency not in self._points and len(self._points) >= MAX_FRAME_BINS:
                    self.reset()
                    raise SweepParseError("assembled sweep exceeded the frame memory limit")
                self._points[frequency] = power
        return completed

    def flush(self) -> SpectrumFrame | None:
        if not self._points or self._timestamp is None:
            self.reset()
            return None
        points = sorted(self._points.items())
        frame = SpectrumFrame(
            timestamp=self._timestamp,
            frequencies_hz=tuple(frequency for frequency, _ in points),
            powers_db=tuple(power for _, power in points),
        )
        self.reset()
        return frame

    def reset(self) -> None:
        self._points.clear()
        self._timestamp = None


class _CallbackSource:
    def __init__(self, config: SweepConfig) -> None:
        if not isinstance(config, SweepConfig):
            raise TypeError("config must be a SweepConfig")
        self._config = config
        self._callbacks: list[FrameCallback] = []
        self._state_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._status = SourceStatus.STOPPED
        self._error: SourceError | None = None
        self._callback_errors = 0

    @property
    def config(self) -> SweepConfig:
        with self._state_lock:
            return self._config

    @property
    def status(self) -> SourceStatus:
        with self._state_lock:
            return self._status

    @property
    def error(self) -> SourceError | None:
        with self._state_lock:
            return self._error

    @property
    def callback_errors(self) -> int:
        with self._state_lock:
            return self._callback_errors

    def add_callback(self, callback: FrameCallback) -> FrameCallback:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._state_lock:
            if callback in self._callbacks:
                return callback
            if len(self._callbacks) >= MAX_CALLBACKS:
                raise SourceError(f"a source may have at most {MAX_CALLBACKS} callbacks")
            self._callbacks.append(callback)
        return callback

    subscribe = add_callback

    def remove_callback(self, callback: FrameCallback) -> None:
        with self._state_lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    unsubscribe = remove_callback

    def _emit(self, frame: SpectrumFrame) -> None:
        with self._state_lock:
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            try:
                callback(frame)
            except Exception:  # noqa: BLE001 - isolate arbitrary consumer callbacks
                # User callbacks cannot take down or deadlock the radio reader.
                with self._state_lock:
                    self._callback_errors += 1


class HackRFSweepSource(_CallbackSource):
    """Manage one ``hackrf_sweep`` subprocess and publish complete sweeps."""

    def __init__(
        self,
        config: SweepConfig,
        *,
        executable: str = "hackrf_sweep",
        termination_timeout_s: float = 2.0,
        popen_factory: Callable[..., subprocess.Popen[str]] | None = None,
    ) -> None:
        super().__init__(config)
        if termination_timeout_s <= 0:
            raise ValueError("termination_timeout_s must be positive")
        self.executable = executable
        self.termination_timeout_s = termination_timeout_s
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_LINES)
        self._malformed_lines = 0

    @property
    def argv(self) -> list[str]:
        return build_sweep_argv(self.config, executable=self.executable)

    @property
    def running(self) -> bool:
        with self._state_lock:
            return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    @property
    def malformed_lines(self) -> int:
        with self._state_lock:
            return self._malformed_lines

    @property
    def stderr_tail(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(self._stderr_tail)

    def start(self, callback: FrameCallback | None = None) -> None:
        with self._lifecycle_lock:
            if callback is not None:
                self.add_callback(callback)
            with self._state_lock:
                if self._thread is not None and self._thread.is_alive():
                    if not self._stop_event.is_set() and self._status in (
                        SourceStatus.STARTING,
                        SourceStatus.RUNNING,
                    ):
                        return
                    raise SourceError("the previous sweep reader has not stopped")
                self._stop_event.clear()
                self._error = None
                self._stderr_tail.clear()
                self._malformed_lines = 0
                self._status = SourceStatus.STARTING
                config = self._config
                thread = threading.Thread(
                    target=self._run,
                    args=(config,),
                    name="rftui-sweep",
                    daemon=True,
                )
                self._thread = thread
            thread.start()

    def stop(self, timeout: float | None = None) -> None:
        wait_s = self.termination_timeout_s if timeout is None else timeout
        if wait_s < 0:
            raise ValueError("timeout must be non-negative")
        with self._lifecycle_lock:
            self._stop_event.set()
            with self._state_lock:
                process = self._process
                thread = self._thread
                if self._status not in (SourceStatus.STOPPED, SourceStatus.FAILED):
                    self._status = SourceStatus.STOPPING
            _terminate(process)
            if thread is not None and thread is not threading.current_thread():
                thread.join(wait_s)
                if thread.is_alive():
                    _kill(process)
                    thread.join(wait_s)
                if thread.is_alive():
                    failure = SourceProcessError(
                        "sweep reader did not stop after terminate and kill"
                    )
                    with self._state_lock:
                        self._error = failure
                        self._status = SourceStatus.FAILED
                    raise failure
            with self._state_lock:
                if thread is None or not thread.is_alive():
                    self._status = SourceStatus.STOPPED

    def restart(self, callback: FrameCallback | None = None) -> None:
        with self._lifecycle_lock:
            self.stop()
            self.start(callback)

    def reconfigure(self, config: SweepConfig) -> None:
        if not isinstance(config, SweepConfig):
            raise TypeError("config must be a SweepConfig")
        with self._lifecycle_lock:
            was_running = self.running
            if was_running:
                self.stop()
            with self._state_lock:
                self._config = config
            if was_running:
                self.start()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self, config: SweepConfig) -> None:
        process: subprocess.Popen[str] | None = None
        stderr_thread: threading.Thread | None = None
        failed: SourceError | None = None
        assembler = _SweepAssembler(config)
        try:
            popen = self._popen_factory or subprocess.Popen
            try:
                process = popen(
                    build_sweep_argv(config, executable=self.executable),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                )
            except FileNotFoundError as exc:
                raise ToolNotFoundError(
                    f"{self.executable!r} was not found; install the HackRF host tools"
                ) from exc
            except OSError as exc:
                raise SourceProcessError(f"could not start {self.executable!r}: {exc}") from exc
            if process.stdout is None or process.stderr is None:
                raise SourceProcessError("hackrf_sweep did not provide stdout and stderr pipes")
            with self._state_lock:
                self._process = process
                self._status = SourceStatus.RUNNING
            stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(process.stderr,),
                name="rftui-sweep-stderr",
                daemon=True,
            )
            stderr_thread.start()
            if self._stop_event.is_set():
                _terminate(process)

            for raw_line in _iter_bounded_lines(process.stdout, MAX_CSV_LINE_CHARS):
                if self._stop_event.is_set():
                    break
                if raw_line is None:
                    self._increment_malformed()
                    continue
                try:
                    row = _parse_sweep_row(raw_line)
                    if row is None:
                        continue
                    frame = assembler.add(row)
                except SweepParseError:
                    self._increment_malformed()
                    continue
                if frame is not None and not self._stop_event.is_set():
                    self._emit(frame)

            if self._stop_event.is_set():
                _terminate(process)
            try:
                returncode = process.wait(timeout=self.termination_timeout_s)
            except subprocess.TimeoutExpired:
                _kill(process)
                returncode = process.wait(timeout=self.termination_timeout_s)
            if stderr_thread is not None:
                stderr_thread.join(self.termination_timeout_s)
            if not self._stop_event.is_set() and returncode != 0:
                detail = self._stderr_detail()
                failed = SourceProcessError(
                    f"{self.executable!r} exited with status {returncode}"
                    + (f": {detail}" if detail else "")
                )
            elif not self._stop_event.is_set():
                final_frame = assembler.flush()
                if final_frame is not None:
                    self._emit(final_frame)
        except SourceError as exc:
            failed = exc
        except Exception as exc:  # noqa: BLE001 - convert worker failures into source state
            failed = SourceProcessError(f"unexpected sweep reader failure: {exc}")
        finally:
            if process is not None and _poll(process) is None:
                _terminate(process)
                try:
                    process.wait(timeout=self.termination_timeout_s)
                except (subprocess.TimeoutExpired, OSError):
                    _kill(process)
            _close_pipe(process.stdout if process is not None else None)
            _close_pipe(process.stderr if process is not None else None)
            if stderr_thread is not None and stderr_thread is not threading.current_thread():
                stderr_thread.join(min(self.termination_timeout_s, 0.25))
            with self._state_lock:
                if failed is None and self._status is SourceStatus.FAILED:
                    failed = self._error
                self._process = None
                if self._thread is threading.current_thread():
                    self._thread = None
                self._error = failed
                self._status = SourceStatus.FAILED if failed is not None else SourceStatus.STOPPED

    def _drain_stderr(self, stream: TextIO) -> None:
        for line in _iter_bounded_lines(stream, STDERR_LINE_CHARS):
            value = "[oversized diagnostic line]" if line is None else line.strip()
            if value:
                with self._state_lock:
                    self._stderr_tail.append(value)

    def _increment_malformed(self) -> None:
        with self._state_lock:
            self._malformed_lines += 1

    def _stderr_detail(self) -> str:
        with self._state_lock:
            return self._stderr_tail[-1] if self._stderr_tail else ""


class DemoSpectrumSource(_CallbackSource):
    """A deterministic, bounded spectrum generator with the live-source API."""

    def __init__(
        self,
        config: SweepConfig,
        *,
        interval_s: float = 0.12,
        seed: int = 0x4841434B,
    ) -> None:
        super().__init__(config)
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be a finite positive number")
        self.interval_s = float(interval_s)
        self.seed = int(seed)
        self._frame_index = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def running(self) -> bool:
        with self._state_lock:
            return bool(self._thread and self._thread.is_alive() and not self._stop_event.is_set())

    def next_frame(self, *, timestamp: datetime | None = None) -> SpectrumFrame:
        """Return the next repeatable demo frame (useful for tests and snapshots)."""

        with self._state_lock:
            index = self._frame_index
            self._frame_index += 1
            config = self._config
        frequencies = _planned_demo_frequencies(config)
        count = len(frequencies)
        rng = random.Random((self.seed << 32) ^ index)
        phase = index * 0.17
        powers: list[float] = []
        for position in range(count):
            fraction = (position + 0.5) / count
            floor = -91.0 + 1.7 * math.sin(position * 0.43 + phase) + rng.uniform(-1.25, 1.25)
            peak_a = 35.0 * math.exp(
                -0.5 * ((fraction - (0.24 + 0.015 * math.sin(phase))) / 0.025) ** 2
            )
            peak_b = 26.0 * math.exp(-0.5 * ((fraction - 0.58) / 0.045) ** 2)
            peak_c = 19.0 * math.exp(
                -0.5 * ((fraction - (0.81 - 0.01 * math.cos(phase))) / 0.018) ** 2
            )
            powers.append(round(floor + peak_a + peak_b + peak_c, 3))
        return SpectrumFrame(
            timestamp=timestamp or datetime.now(UTC),
            frequencies_hz=frequencies,
            powers_db=tuple(powers),
        )

    def start(self, callback: FrameCallback | None = None) -> None:
        with self._lifecycle_lock:
            if callback is not None:
                self.add_callback(callback)
            with self._state_lock:
                if self._thread is not None and self._thread.is_alive():
                    if not self._stop_event.is_set() and self._status in (
                        SourceStatus.STARTING,
                        SourceStatus.RUNNING,
                    ):
                        return
                    raise SourceError("the previous demo reader has not stopped")
                self._stop_event.clear()
                self._error = None
                self._frame_index = 0
                self._status = SourceStatus.STARTING
                thread = threading.Thread(target=self._run, name="rftui-demo", daemon=True)
                self._thread = thread
            thread.start()

    def stop(self, timeout: float | None = None) -> None:
        wait_s = max(1.0, self.interval_s * 2) if timeout is None else timeout
        if wait_s < 0:
            raise ValueError("timeout must be non-negative")
        with self._lifecycle_lock:
            self._stop_event.set()
            with self._state_lock:
                thread = self._thread
                if self._status not in (SourceStatus.STOPPED, SourceStatus.FAILED):
                    self._status = SourceStatus.STOPPING
            if thread is not None and thread is not threading.current_thread():
                thread.join(wait_s)
                if thread.is_alive():
                    failure = SourceError("demo reader did not stop within the timeout")
                    with self._state_lock:
                        self._error = failure
                        self._status = SourceStatus.FAILED
                    raise failure
            with self._state_lock:
                if thread is None or not thread.is_alive():
                    self._status = SourceStatus.STOPPED

    def restart(self, callback: FrameCallback | None = None) -> None:
        with self._lifecycle_lock:
            self.stop()
            self.start(callback)

    def reconfigure(self, config: SweepConfig) -> None:
        if not isinstance(config, SweepConfig):
            raise TypeError("config must be a SweepConfig")
        with self._lifecycle_lock:
            was_running = self.running
            if was_running:
                self.stop()
            with self._state_lock:
                self._config = config
                self._frame_index = 0
            if was_running:
                self.start()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _run(self) -> None:
        with self._state_lock:
            self._status = SourceStatus.RUNNING
        try:
            while not self._stop_event.is_set():
                self._emit(self.next_frame())
                self._stop_event.wait(self.interval_s)
        except Exception as exc:  # noqa: BLE001 - keep worker failures observable
            with self._state_lock:
                self._error = SourceError(f"demo source failed: {exc}")
                self._status = SourceStatus.FAILED
        finally:
            with self._state_lock:
                if self._thread is threading.current_thread():
                    self._thread = None
                if self._status is not SourceStatus.FAILED:
                    self._status = SourceStatus.STOPPED


def _planned_demo_frequencies(config: SweepConfig) -> tuple[float, ...]:
    """Reproduce the live tool's five-MHz, CSV-rounded frequency lattice."""

    plan = config.fft_plan
    frequencies: list[float] = []
    for segment_low_hz in range(
        config.start_hz,
        config.stop_hz,
        HACKRF_SWEEP_SEGMENT_HZ,
    ):
        segment_stop_hz = min(segment_low_hz + HACKRF_SWEEP_SEGMENT_HZ, config.stop_hz)
        for index in range(plan.bins_per_segment):
            frequency = segment_low_hz + index * plan.csv_bin_width_hz
            if frequency >= segment_stop_hz:
                break
            frequencies.append(frequency)
    if len(frequencies) != config.estimated_bin_count:
        raise SourceError("internal HackRF FFT plan count mismatch")
    return tuple(frequencies)


def _parse_integral(value: str, field: str) -> int:
    try:
        number = Decimal(value.strip())
    except InvalidOperation as exc:
        raise SweepParseError(f"invalid {field}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise SweepParseError(f"invalid {field}")
    return int(number)


def _parse_finite_float(value: str, field: str) -> float:
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise SweepParseError(f"invalid {field}") from exc
    if not math.isfinite(number):
        raise SweepParseError(f"non-finite {field}")
    return number


def _iter_bounded_lines(stream: TextIO, limit: int) -> Iterator[str | None]:
    """Read lines without ever retaining more than ``limit + 1`` characters."""

    while True:
        chunk = stream.readline(limit + 1)
        if chunk == "":
            return
        if len(chunk) <= limit and (chunk.endswith("\n") or len(chunk) < limit + 1):
            yield chunk
            continue
        while chunk and not chunk.endswith("\n"):
            chunk = stream.readline(limit + 1)
        yield None


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _last_nonempty_line(value: str) -> str:
    return next((line.strip() for line in reversed(value.splitlines()) if line.strip()), "")


def _terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or _poll(process) is not None:
        return
    try:
        process.terminate()
    except (OSError, ProcessLookupError):
        pass


def _kill(process: subprocess.Popen[str] | None) -> None:
    if process is None or _poll(process) is not None:
        return
    try:
        process.kill()
    except (OSError, ProcessLookupError):
        pass


def _poll(process: subprocess.Popen[str]) -> int | None:
    try:
        return process.poll()
    except OSError:
        return process.returncode


def _close_pipe(pipe: io.TextIOBase | TextIO | None) -> None:
    if pipe is None:
        return
    try:
        pipe.close()
    except OSError:
        pass
