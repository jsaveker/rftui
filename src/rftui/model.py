"""Immutable data models shared by RFTUI sources and views."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from itertools import pairwise
from math import isfinite

MIN_FREQUENCY_HZ = 0
MAX_FREQUENCY_HZ = 7_250_000_000
MIN_BIN_WIDTH_HZ = 2_445
MAX_BIN_WIDTH_HZ = 5_000_000
MAX_FRAME_BINS = 131_072
HACKRF_SAMPLE_RATE_HZ = 20_000_000
HACKRF_SWEEP_SEGMENT_HZ = 5_000_000

_SERIAL_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """A HackRF reported by ``hackrf_info``."""

    index: int
    serial_number: str
    board_name: str | None = None
    board_id: int | None = None
    firmware_version: str | None = None
    part_id: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("device index must be non-negative")
        if not self.serial_number or len(self.serial_number) > 256:
            raise ValueError("device serial number is missing or too long")

    @property
    def serial(self) -> str:
        """Return the serial using the shorter common attribute name."""

        return self.serial_number

    @property
    def label(self) -> str:
        """Return a concise human-readable device label."""

        board = self.board_name or "HackRF"
        return f"{board} ({self.serial_number})"


@dataclass(frozen=True, slots=True)
class HackRFFFTPlan:
    """The FFT geometry ``hackrf_sweep`` derives from its ``-w`` value."""

    requested_bin_width_hz: int
    fft_bin_count: int
    csv_bin_width_centihz: int

    @property
    def effective_bin_width_hz(self) -> float:
        """Return the true mathematical FFT resolution before CSV rounding."""

        return HACKRF_SAMPLE_RATE_HZ / self.fft_bin_count

    @property
    def csv_bin_width_hz(self) -> float:
        """Return the two-decimal resolution emitted in sweep CSV rows."""

        return self.csv_bin_width_centihz / 100

    @property
    def bins_per_segment(self) -> int:
        """Return output bins in each five-MHz CSV segment."""

        return self.fft_bin_count // 4

    def bin_count_for_span(self, span_hz: int) -> int:
        """Count parsed points after cropping a whole-MHz sweep span.

        CSV rows restart their frequency lattice every five MHz. Counting each
        segment also accounts for the tool's two-decimal bin-width formatting.
        """

        if isinstance(span_hz, bool) or not isinstance(span_hz, int):
            raise TypeError("span_hz must be an integer")
        if span_hz <= 0:
            raise ValueError("span_hz must be positive")
        full_segments, remainder_hz = divmod(span_hz, HACKRF_SWEEP_SEGMENT_HZ)
        count = full_segments * self.bins_per_segment
        if remainder_hz:
            numerator_centihz = remainder_hz * 100
            partial = (
                numerator_centihz + self.csv_bin_width_centihz - 1
            ) // self.csv_bin_width_centihz
            count += min(partial, self.bins_per_segment)
        return count


def plan_hackrf_fft(requested_bin_width_hz: int) -> HackRFFFTPlan:
    """Mirror the official ``hackrf_sweep`` FFT-width adjustment exactly."""

    if isinstance(requested_bin_width_hz, bool) or not isinstance(requested_bin_width_hz, int):
        raise TypeError("requested_bin_width_hz must be an integer")
    if not MIN_BIN_WIDTH_HZ <= requested_bin_width_hz <= MAX_BIN_WIDTH_HZ:
        raise ValueError(
            f"requested_bin_width_hz must be between {MIN_BIN_WIDTH_HZ} and {MAX_BIN_WIDTH_HZ}"
        )
    fft_bin_count = HACKRF_SAMPLE_RATE_HZ // requested_bin_width_hz
    while (fft_bin_count + 4) % 8:
        fft_bin_count += 1
    # hackrf_sweep writes the double with %.2f. Fraction keeps tie handling and
    # the safety calculation independent of binary floating-point drift.
    csv_bin_width_centihz = round(Fraction(HACKRF_SAMPLE_RATE_HZ * 100, fft_bin_count))
    return HackRFFFTPlan(
        requested_bin_width_hz=requested_bin_width_hz,
        fft_bin_count=fft_bin_count,
        csv_bin_width_centihz=csv_bin_width_centihz,
    )


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Validated, receive-only configuration for ``hackrf_sweep``.

    ``hackrf_sweep`` accepts frequency bounds in whole MHz, so the model rejects
    sub-MHz endpoints instead of silently tuning a different band.
    """

    start_hz: int = 902_000_000
    stop_hz: int = 928_000_000
    bin_width_hz: int = 100_000
    lna_gain: int = 16
    vga_gain: int = 20
    amp: bool = False
    serial: str | None = None

    def __post_init__(self) -> None:
        for name in ("start_hz", "stop_hz", "bin_width_hz", "lna_gain", "vga_gain"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")

        if not MIN_FREQUENCY_HZ <= self.start_hz < MAX_FREQUENCY_HZ:
            raise ValueError("start_hz must be between 0 Hz and 7.25 GHz")
        if not MIN_FREQUENCY_HZ < self.stop_hz <= MAX_FREQUENCY_HZ:
            raise ValueError("stop_hz must be between 1 Hz and 7.25 GHz")
        if self.stop_hz <= self.start_hz:
            raise ValueError("stop_hz must be greater than start_hz")
        if self.start_hz % 1_000_000 or self.stop_hz % 1_000_000:
            raise ValueError("hackrf_sweep frequency endpoints must be whole MHz")
        if not MIN_BIN_WIDTH_HZ <= self.bin_width_hz <= MAX_BIN_WIDTH_HZ:
            raise ValueError(
                f"bin_width_hz must be between {MIN_BIN_WIDTH_HZ} and {MAX_BIN_WIDTH_HZ}"
            )
        if self.lna_gain not in range(0, 41, 8):
            raise ValueError("lna_gain must be 0-40 dB in 8 dB steps")
        if self.vga_gain not in range(0, 63, 2):
            raise ValueError("vga_gain must be 0-62 dB in 2 dB steps")
        if not isinstance(self.amp, bool):
            raise TypeError("amp must be a boolean")
        if self.serial is not None and not _SERIAL_RE.fullmatch(self.serial):
            raise ValueError("serial must contain only letters, numbers, '.', '_', ':', or '-'")
        if self.estimated_bin_count > MAX_FRAME_BINS:
            raise ValueError(
                f"requested sweep would exceed the {MAX_FRAME_BINS:,}-bin memory limit"
            )

    @property
    def start_mhz(self) -> int:
        return self.start_hz // 1_000_000

    @property
    def stop_mhz(self) -> int:
        return self.stop_hz // 1_000_000

    @property
    def span_hz(self) -> int:
        return self.stop_hz - self.start_hz

    @property
    def estimated_bin_count(self) -> int:
        return self.fft_plan.bin_count_for_span(self.span_hz)

    @property
    def fft_plan(self) -> HackRFFFTPlan:
        return plan_hackrf_fft(self.bin_width_hz)

    @property
    def effective_bin_width_hz(self) -> float:
        return self.fft_plan.effective_bin_width_hz

    @property
    def csv_bin_width_hz(self) -> float:
        return self.fft_plan.csv_bin_width_hz

    @property
    def lna_gain_db(self) -> int:
        return self.lna_gain

    @property
    def vga_gain_db(self) -> int:
        return self.vga_gain

    @property
    def amp_enabled(self) -> bool:
        return self.amp


@dataclass(frozen=True, slots=True)
class SpectrumFrame:
    """One immutable sweep across a frequency range.

    Power values are relative dB produced by ``hackrf_sweep``; they are not
    calibrated dBm measurements.
    """

    timestamp: datetime
    frequencies_hz: tuple[float, ...]
    powers_db: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")

        frequencies = tuple(float(value) for value in self.frequencies_hz)
        powers = tuple(float(value) for value in self.powers_db)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "powers_db", powers)

        if not frequencies:
            raise ValueError("a spectrum frame must contain at least one bin")
        if len(frequencies) != len(powers):
            raise ValueError("frequencies_hz and powers_db must have the same length")
        if len(frequencies) > MAX_FRAME_BINS:
            raise ValueError(f"a spectrum frame may contain at most {MAX_FRAME_BINS:,} bins")
        if any(not isfinite(value) or value < 0 for value in frequencies):
            raise ValueError("frequencies_hz must contain finite non-negative values")
        if any(left >= right for left, right in pairwise(frequencies)):
            raise ValueError("frequencies_hz must be strictly increasing")
        if any(not isfinite(value) for value in powers):
            raise ValueError("powers_db must contain only finite values")

    @property
    def bin_count(self) -> int:
        return len(self.powers_db)

    @property
    def bin_width_hz(self) -> float | None:
        """Derive received resolution; return ``None`` for a one-bin frame."""

        if len(self.frequencies_hz) < 2:
            return None
        return self.frequencies_hz[1] - self.frequencies_hz[0]

    @property
    def peak_power_db(self) -> float:
        return max(self.powers_db)

    @property
    def peak_frequency_hz(self) -> float:
        index = max(range(len(self.powers_db)), key=self.powers_db.__getitem__)
        return self.frequencies_hz[index]

    @property
    def floor_power_db(self) -> float:
        return min(self.powers_db)
