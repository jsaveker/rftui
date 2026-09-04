"""Textual spectrum cockpit for RFTUI.

The UI deliberately talks about *relative dB*. ``hackrf_sweep`` is useful for
finding and comparing energy, but its CSV output is not a calibrated power
measurement.  Keeping that distinction visible is part of the interface, not
just a disclaimer in the README.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Footer, Static

from .model import DeviceInfo, SpectrumFrame, SweepConfig

MIN_FREQUENCY_HZ = 1_000_000
MAX_FREQUENCY_HZ = 6_000_000_000
MIN_SPAN_HZ = 2_000_000
WATERFALL_ROWS = 96
WATERFALL_BINS = 512
DISPLAY_BINS = 4_096
DISPLAY_INTERVAL_S = 1 / 24


def _finite(value: object) -> float | None:
    """Return a finite float, or ``None`` for an unusable sample."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _resample_peak(values: Sequence[float], width: int) -> list[float]:
    """Reduce a row to terminal width while retaining narrow RF peaks."""

    width = max(1, width)
    if not values:
        return []
    result: list[float] = []
    count = len(values)
    for column in range(width):
        start = min(count - 1, int(column * count / width))
        stop = min(count, max(start + 1, int((column + 1) * count / width)))
        bucket = [sample for value in values[start:stop] if (sample := _finite(value)) is not None]
        result.append(max(bucket) if bucket else float("nan"))
    return result


def _frequency(value_hz: float, *, compact: bool = False) -> str:
    """Format a frequency without pretending to more precision than we have."""

    if abs(value_hz) >= 1_000_000_000:
        return f"{value_hz / 1_000_000_000:.4f}{'G' if compact else ' GHz'}"
    if abs(value_hz) >= 1_000_000:
        return f"{value_hz / 1_000_000:.3f}{'M' if compact else ' MHz'}"
    if abs(value_hz) >= 1_000:
        return f"{value_hz / 1_000:.1f}{'k' if compact else ' kHz'}"
    return f"{value_hz:.0f}{'Hz' if compact else ' Hz'}"


def _bin_resolution(value_hz: float) -> str:
    """Format sweep resolution precisely enough to expose FFT quantization."""

    if abs(value_hz) >= 1_000_000:
        return f"{value_hz / 1_000_000:.3f} MHz"
    if abs(value_hz) >= 1_000:
        return f"{value_hz / 1_000:.3f} kHz"
    return f"{value_hz:.2f} Hz"


def _requested_bin_is_materially_different(actual_hz: float, requested_hz: float) -> bool:
    """Return whether the requested ``-w`` is useful alongside actual spacing."""

    return not math.isclose(actual_hz, requested_hz, rel_tol=0.005, abs_tol=1.0)


def _frame_values(frame: SpectrumFrame | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Extract matched, finite-friendly frame arrays."""

    if frame is None:
        return (), ()
    frequencies = tuple(float(value) for value in frame.frequencies_hz)
    powers = tuple(float(value) for value in frame.powers_db)
    length = min(len(frequencies), len(powers))
    return frequencies[:length], powers[:length]


def _downsample_frame(frame: SpectrumFrame, max_bins: int = DISPLAY_BINS) -> SpectrumFrame:
    """Build bounded UI state while preserving each bucket's true peak frequency."""

    count = min(len(frame.frequencies_hz), len(frame.powers_db))
    if count <= max_bins:
        return frame
    frequencies: list[float] = []
    powers: list[float] = []
    for bucket in range(max_bins):
        start = bucket * count // max_bins
        stop = max(start + 1, (bucket + 1) * count // max_bins)
        peak_index = max(range(start, stop), key=frame.powers_db.__getitem__)
        frequencies.append(frame.frequencies_hz[peak_index])
        powers.append(frame.powers_db[peak_index])
    return SpectrumFrame(frame.timestamp, tuple(frequencies), tuple(powers))


def _representative_powers(frame: SpectrumFrame, max_bins: int = DISPLAY_BINS) -> tuple[float, ...]:
    """Take a bounded, evenly stratified level sample without peak-selection bias."""

    powers = frame.powers_db
    count = len(powers)
    if count <= max_bins:
        return powers
    samples: list[float] = []
    for bucket in range(max_bins):
        start = bucket * count // max_bins
        stop = max(start + 1, (bucket + 1) * count // max_bins)
        samples.append(powers[(start + stop - 1) // 2])
    return tuple(samples)


def _relative_floor(values: Sequence[float]) -> float | None:
    """Estimate the lower-third median used as the relative noise floor."""

    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    return statistics.median(finite[: max(1, len(finite) // 3)])


def _level_range(values: Iterable[float]) -> tuple[float, float]:
    """Choose stable-looking plot bounds for uncalibrated relative levels."""

    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return -100.0, -20.0
    low_index = min(len(finite) - 1, int(len(finite) * 0.08))
    high_index = min(len(finite) - 1, int(len(finite) * 0.995))
    low = math.floor(finite[low_index] / 10.0) * 10.0
    high = math.ceil(finite[high_index] / 10.0) * 10.0
    if high - low < 30.0:
        midpoint = (high + low) / 2.0
        low, high = midpoint - 20.0, midpoint + 20.0
    if high - low > 100.0:
        low = high - 100.0
    return low, high


def _peak_rows(frame: SpectrumFrame | None, limit: int = 5) -> list[tuple[float, float]]:
    """Find strong local maxima, keeping adjacent bins from flooding the list."""

    frequencies, powers = _frame_values(frame)
    if not powers:
        return []
    candidates: list[int] = []
    for index, power in enumerate(powers):
        if not math.isfinite(power):
            continue
        left = powers[index - 1] if index else float("-inf")
        right = powers[index + 1] if index + 1 < len(powers) else float("-inf")
        if power >= left and power >= right:
            candidates.append(index)
    candidates.sort(key=lambda index: powers[index], reverse=True)
    separation = max(1, len(powers) // 40)
    selected: list[int] = []
    for index in candidates:
        if all(abs(index - previous) >= separation for previous in selected):
            selected.append(index)
        if len(selected) == limit:
            break
    return [(frequencies[index], powers[index]) for index in selected]


class SpectrumPlot(Static):
    """A compact terminal spectrum trace with peak hold."""

    can_focus = False

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self.frame: SpectrumFrame | None = None
        self.peak_hold: tuple[float, ...] = ()
        self.axis_start_hz: float | None = None
        self.axis_stop_hz: float | None = None

    def set_frame(self, frame: SpectrumFrame, config: SweepConfig) -> None:
        _, powers = _frame_values(frame)
        if len(self.peak_hold) != len(powers):
            self.peak_hold = powers
        else:
            self.peak_hold = tuple(
                max(old, new) if math.isfinite(new) else old
                for old, new in zip(self.peak_hold, powers, strict=True)
            )
        self.frame = frame
        self.axis_start_hz = config.start_hz
        self.axis_stop_hz = config.stop_hz
        self.refresh()

    def clear_peak_hold(self) -> None:
        self.peak_hold = ()
        if self.frame is not None:
            _, powers = _frame_values(self.frame)
            self.peak_hold = powers
        self.refresh()

    def reset(self, config: SweepConfig) -> None:
        self.frame = None
        self.peak_hold = ()
        self.axis_start_hz = config.start_hz
        self.axis_stop_hz = config.stop_hz
        self.refresh()

    def render(self) -> Text:
        width = max(18, self.content_size.width)
        height = max(5, self.content_size.height)
        frequencies, powers = _frame_values(self.frame)
        if not powers:
            waiting = Text("\n" * max(0, height // 2 - 1), justify="center")
            waiting.append("NO SPECTRUM FRAMES YET\n", style="bold #52e0c4")
            waiting.append("waiting for receive-only sweep", style="#52777b")
            return waiting

        label_width = 8
        plot_width = max(8, width - label_width)
        chart_height = max(3, height - 2)
        samples = _resample_peak(powers, plot_width)
        held = _resample_peak(self.peak_hold, plot_width)
        low, high = _level_range((*samples, *held))
        span = max(1.0, high - low)

        def row_for(value: float) -> int | None:
            if not math.isfinite(value):
                return None
            ratio = max(0.0, min(1.0, (value - low) / span))
            return chart_height - 1 - round(ratio * (chart_height - 1))

        grid = [[("·", "#123038") for _ in range(plot_width)] for _ in range(chart_height)]
        for row in range(chart_height):
            if row in {0, chart_height // 2, chart_height - 1}:
                grid[row] = [("┄", "#17434a") for _ in range(plot_width)]
        for column in range(0, plot_width, max(1, plot_width // 8)):
            for row in range(chart_height):
                grid[row][column] = ("┊", "#15373d")

        previous_row: int | None = None
        for column, sample in enumerate(samples):
            current_row = row_for(sample)
            held_row = row_for(held[column]) if column < len(held) else None
            if held_row is not None:
                grid[held_row][column] = ("•", "#bc8b45")
            if current_row is None:
                continue
            if previous_row is not None and abs(previous_row - current_row) > 1:
                for row in range(
                    min(previous_row, current_row) + 1, max(previous_row, current_row)
                ):
                    grid[row][column] = ("│", "#246d68")
            grid[current_row][column] = ("●", "bold #54f1ce")
            previous_row = current_row

        result = Text()
        for row, cells in enumerate(grid):
            level = high - row * span / max(1, chart_height - 1)
            if row in {0, chart_height // 2, chart_height - 1}:
                result.append(f"{level:>5.0f} ┤", style="#709498")
            else:
                result.append("      │", style="#2b555a")
            for character, style in cells:
                result.append(character, style=style)
            if row + 1 < chart_height:
                result.append("\n")

        result.append("\n")
        result.append(" rel dB ", style="bold #091114 on #ddb35b")
        axis_width = max(1, width - 8)
        left = _frequency(self.axis_start_hz or frequencies[0], compact=True)
        middle_hz = (
            (self.axis_start_hz + self.axis_stop_hz) / 2
            if self.axis_start_hz is not None and self.axis_stop_hz is not None
            else frequencies[len(frequencies) // 2]
        )
        middle = _frequency(middle_hz, compact=True)
        right = _frequency(self.axis_stop_hz or frequencies[-1], compact=True)
        gap = max(1, (axis_width - len(left) - len(middle) - len(right)) // 2)
        axis = f"{left}{' ' * gap}{middle}{' ' * gap}{right}"
        result.append(axis[:axis_width].ljust(axis_width), style="#68898e")
        return result


@dataclass(frozen=True, slots=True)
class _WaterfallRow:
    powers: tuple[float, ...]
    low: float
    high: float


class WaterfallPlot(Static):
    """A bounded rolling history rendered as a terminal heat map."""

    can_focus = False
    PALETTE: ClassVar[tuple[str, ...]] = (
        "#071317",
        "#0b2530",
        "#11485a",
        "#176e77",
        "#1c9a8b",
        "#39c79d",
        "#82dda0",
        "#d8d86b",
        "#f3a84f",
        "#fff0bd",
    )
    GLYPHS: ClassVar[str] = " ·ˑ:≡+*#%@"

    def __init__(
        self,
        *,
        max_rows: int = WATERFALL_ROWS,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.history: deque[_WaterfallRow] = deque(maxlen=max_rows)
        self._level_low: float | None = None
        self._level_high: float | None = None

    def push(self, frame: SpectrumFrame) -> None:
        _, powers = _frame_values(frame)
        if powers:
            # A source frame may contain 131k bins. Retaining that 96 times
            # would turn a visual history into an accidental memory sink.
            bounded = tuple(_resample_peak(powers, min(WATERFALL_BINS, len(powers))))
            target_low, target_high = _level_range(bounded)
            if self._level_low is None or self._level_high is None:
                self._level_low, self._level_high = target_low, target_high
            else:
                # Adapt slowly to a changed noise environment. Each row keeps
                # the exact scale it was painted with, so new frames never
                # recolour history that the operator already saw.
                self._level_low = self._level_low * 0.92 + target_low * 0.08
                self._level_high = self._level_high * 0.84 + target_high * 0.16
            self.history.append(_WaterfallRow(bounded, self._level_low, self._level_high))
            self.refresh()

    def clear_history(self) -> None:
        self.history.clear()
        self._level_low = None
        self._level_high = None
        self.refresh()

    def render(self) -> Text:
        width = max(8, self.content_size.width)
        height = max(3, self.content_size.height)
        if not self.history:
            result = Text("\n" * max(0, height // 2 - 1), justify="center")
            result.append("WATERFALL ARMED · HISTORY 0/96", style="bold #2d817d")
            return result

        visible_rows = list(self.history)[-height:]
        result = Text()
        scale_steps = min(len(self.PALETTE), len(self.GLYPHS))
        # Newest data stays at the top, so motion is immediately visible.
        for row_number, row in enumerate(reversed(visible_rows)):
            span = max(1.0, row.high - row.low)
            for value in _resample_peak(row.powers, width):
                ratio = 0.0 if not math.isfinite(value) else (value - row.low) / span
                index = round(max(0.0, min(1.0, ratio)) * (scale_steps - 1))
                result.append(self.GLYPHS[index], style=self.PALETTE[index])
            if row_number + 1 < len(visible_rows):
                result.append("\n")
        return result


class MetricsPanel(Static):
    """Current sweep measurements and strongest relative peaks."""

    can_focus = False

    def __init__(self, *, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        self.frame: SpectrumFrame | None = None
        self.config: SweepConfig | None = None
        self.frames_per_second = 0.0
        self.level_samples: tuple[float, ...] = ()
        self.actual_bin_width_hz: float | None = None

    def set_data(
        self,
        frame: SpectrumFrame,
        config: SweepConfig,
        fps: float,
        level_samples: tuple[float, ...],
        actual_bin_width_hz: float | None = None,
    ) -> None:
        self.frame = frame
        self.config = config
        self.frames_per_second = fps
        self.level_samples = level_samples
        # The frame retained by the UI may be peak-downsampled, so callers pass
        # spacing from the raw source frame. The fallback keeps this method
        # useful for small, unmodified frames and third-party integrations.
        self.actual_bin_width_hz = (
            frame.bin_width_hz if actual_bin_width_hz is None else actual_bin_width_hz
        )
        self.refresh()

    def reset(self, config: SweepConfig) -> None:
        self.frame = None
        self.config = config
        self.frames_per_second = 0.0
        self.level_samples = ()
        self.actual_bin_width_hz = None
        self.refresh()

    def _bin_widths(self) -> tuple[float | None, float | None]:
        """Return displayed CSV spacing and an optional materially different request."""

        config = self.config
        if config is None:
            return None, None
        actual = (
            config.csv_bin_width_hz
            if self.actual_bin_width_hz is None
            else self.actual_bin_width_hz
        )
        requested = float(config.bin_width_hz)
        return (
            actual,
            requested if _requested_bin_is_materially_different(actual, requested) else None,
        )

    def _render_compact(self) -> Text:
        """Render all essential metrics inside the nine-row 80x24 panel."""

        text = Text("RX · RELATIVE dB", style="bold #4de2c0")
        config = self.config
        actual_bin, requested_bin = self._bin_widths()
        if config is not None:
            text.append(
                f"\n{_frequency(config.start_hz, compact=True)}"
                f"–{_frequency(config.stop_hz, compact=True)}",
                style="#b8d6d4",
            )
            text.append(
                f"\nspan {_frequency(config.span_hz, compact=True)}"
                f" · L{config.lna_gain} V{config.vga_gain}",
                style="#698b8d",
            )
        if actual_bin is not None:
            text.append(f"\nbin {_bin_resolution(actual_bin)} CSV", style="#79a09e")
        if requested_bin is not None:
            text.append(f"\nrequested {_bin_resolution(requested_bin)}", style="#698b8d")

        _frequencies, powers = _frame_values(self.frame)
        finite = [value for value in powers if math.isfinite(value)]
        if finite:
            strongest = max(finite)
            noise = _relative_floor(self.level_samples)
            assert noise is not None
            text.append(f"\npeak {strongest:.1f} · floor {noise:.1f}", style="bold #e9d473")
            text.append(
                f"\nΔ {strongest - noise:.1f} · display {self.frames_per_second:.1f} fps",
                style="#79a09e",
            )
        else:
            text.append("\nawaiting samples", style="#557579")

        max_lines = max(1, self.content_size.height)
        for rank, (frequency, power) in enumerate(_peak_rows(self.frame), 1):
            if text.plain.count("\n") + 1 >= max_lines:
                break
            text.append(f"\n{rank:02}  {_frequency(frequency, compact=True):>10}", style="#a6cac6")
            text.append(f" {power:>6.1f}", style="#deb55c")
        return text

    def render(self) -> Text:
        if self.content_size.height <= 12:
            return self._render_compact()

        text = Text()

        def heading(label: str) -> None:
            if text:
                text.append("\n")
            text.append(label.upper(), style="bold #4de2c0")
            text.append("\n")

        heading("receiver")
        config = self.config
        if config is not None:
            actual_bin, requested_bin = self._bin_widths()
            text.append(f"{_frequency(config.start_hz)} →\n", style="#b8d6d4")
            text.append(f"{_frequency(config.stop_hz)}\n", style="#b8d6d4")
            text.append(f"span   {_frequency(config.stop_hz - config.start_hz)}\n", style="#698b8d")
            if actual_bin is not None:
                text.append(f"bin    {_bin_resolution(actual_bin)} CSV\n", style="#698b8d")
            if requested_bin is not None:
                text.append(f"requested {_bin_resolution(requested_bin)}\n", style="#557579")
            text.append(
                f"gain   LNA {config.lna_gain:>2} · VGA {config.vga_gain:>2}", style="#698b8d"
            )

        _frequencies, powers = _frame_values(self.frame)
        finite = [value for value in powers if math.isfinite(value)]
        heading("levels · relative dB")
        if finite:
            strongest = max(finite)
            noise = _relative_floor(self.level_samples)
            assert noise is not None
            text.append(f"peak   {strongest:>7.1f} dB\n", style="bold #e9d473")
            text.append(f"floor  {noise:>7.1f} dB\n", style="#79a09e")
            text.append(f"delta  {strongest - noise:>7.1f} dB\n", style="#79a09e")
            text.append(f"display {self.frames_per_second:>5.1f} fps", style="#79a09e")
        else:
            text.append("awaiting samples", style="#557579")

        heading("strongest peaks")
        peaks = _peak_rows(self.frame)
        if not peaks:
            text.append("—", style="#557579")
        for rank, (frequency, power) in enumerate(peaks, 1):
            text.append(f"{rank:02}  {_frequency(frequency, compact=True):>10}", style="#a6cac6")
            text.append(
                f" {power:>6.1f}\n" if rank < len(peaks) else f" {power:>6.1f}", style="#deb55c"
            )
        return text


class HelpScreen(ModalScreen[None]):
    """Keyboard reference and measurement-safety reminder."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("escape,q,question_mark,f1", "close_help", "close", key_display="esc"),
    ]

    def compose(self) -> ComposeResult:
        body = Text()
        body.append("RF//TUI  ", style="bold #57f0cc")
        body.append("KEYBOARD MAP\n", style="bold #80a8ad")
        body.append("receive-only spectrum cockpit\n\n", style="#65888b")
        controls = (
            ("p", "pause / resume display"),
            ("r", "restart sweep source"),
            ("[  ]", "shift down / up one span"),
            ("−  +", "zoom out / in at centre"),
            ("g  G", "LNA gain down / up"),
            ("v  V", "VGA gain down / up"),
            ("c", "clear peak hold + waterfall"),
            ("?", "toggle this help"),
            ("q", "quit"),
        )
        for keys, action in controls:
            body.append(f"{keys:<7}", style="bold #e0bc61")
            body.append(f"{action}\n", style="#c4dbd9")
        body.append("\nRX ONLY", style="bold #091114 on #54e3c1")
        body.append("  Levels are relative dB, never calibrated dBm.\n", style="#d9c172")
        body.append("Use a suitable antenna and follow local law.", style="#718f91")
        with Container(id="help-dialog"):
            yield Static(body, id="help-copy")

    def action_close_help(self) -> None:
        self.dismiss()


class RFTUIApp(App[None]):
    """Keyboard-first, receive-only spectrum cockpit."""

    CSS_PATH = "styles.tcss"
    TITLE = "RFTUI"
    SUB_TITLE = "receive-only spectrum cockpit"
    ENABLE_COMMAND_PALETTE: ClassVar[bool] = False

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "quit"),
        Binding("question_mark,f1", "help", "help", key_display="?"),
        Binding("p", "toggle_pause", "pause"),
        Binding("r", "restart", "restart"),
        Binding("c", "clear", "clear"),
        Binding("left_square_bracket", "shift_down", "band down", key_display="[", show=False),
        Binding("right_square_bracket", "shift_up", "band up", key_display="]", show=False),
        Binding("minus,underscore", "zoom_out", "zoom out", key_display="−", show=False),
        Binding("plus,equals_sign", "zoom_in", "zoom in", key_display="+", show=False),
        Binding("g", "lna_down", "LNA −", show=False),
        Binding("G", "lna_up", "LNA +", show=False),
        Binding("v", "vga_down", "VGA −", show=False),
        Binding("V", "vga_up", "VGA +", show=False),
    ]

    def __init__(
        self,
        source: Any,
        config: SweepConfig,
        *,
        device: DeviceInfo | None = None,
        demo: bool = False,
    ) -> None:
        super().__init__()
        self.source = source
        self.config = config
        self.device = device
        self.demo = bool(demo or type(source).__name__.startswith("Demo"))
        self.paused = False
        self.receiver_state = "starting"
        self.last_error: str | None = None
        self.last_frame: SpectrumFrame | None = None
        # Source callbacks can arrive much faster than a terminal can paint.
        # Keep only the newest frame so the radio reader never waits for UI
        # work and memory use remains constant under a sustained sweep.
        self._pending_lock = threading.Lock()
        self._pending_frame: SpectrumFrame | None = None
        self._frames_received = 0
        self._frames_rendered = 0
        self._frames_coalesced = 0
        self._last_source_bins = 0
        self._display_suspended = False
        self._operation_generation = 0
        self._source_operation_lock = threading.Lock()
        self._arrival_times: deque[float] = deque(maxlen=120)
        self._ui_thread_id: int | None = None
        self._source_started = False
        self._closing = False

    @property
    def frames_per_second(self) -> float:
        if len(self._arrival_times) < 2:
            return 0.0
        elapsed = self._arrival_times[-1] - self._arrival_times[0]
        return (len(self._arrival_times) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def frames_received(self) -> int:
        with self._pending_lock:
            return self._frames_received

    @property
    def frames_rendered(self) -> int:
        return self._frames_rendered

    @property
    def frames_coalesced(self) -> int:
        with self._pending_lock:
            return self._frames_coalesced

    def compose(self) -> ComposeResult:
        yield Static(id="topbar")
        with Grid(id="workspace"):
            yield SpectrumPlot(id="spectrum", classes="panel")
            yield MetricsPanel(id="metrics", classes="panel")
            yield WaterfallPlot(id="waterfall", classes="panel")
        with Horizontal(id="statusline"):
            yield Static(id="status-state")
            yield Static(id="status-detail")
            yield Static("RX ONLY · RELATIVE dB", id="measurement-badge")
        yield Footer()

    def on_mount(self) -> None:
        self._ui_thread_id = threading.get_ident()
        spectrum = self.query_one("#spectrum", SpectrumPlot)
        spectrum.border_title = "SPECTRUM · TRACE + PEAK HOLD"
        spectrum.reset(self.config)
        metrics = self.query_one("#metrics", MetricsPanel)
        metrics.border_title = "SIGNAL INTELLIGENCE"
        metrics.reset(self.config)
        self.query_one(
            "#waterfall", WaterfallPlot
        ).border_title = f"WATERFALL · NEWEST ↑ · BOUNDED {WATERFALL_ROWS} FRAMES"
        self._refresh_chrome()
        self.set_interval(DISPLAY_INTERVAL_S, self._drain_pending_frame)
        self.set_interval(0.25, self._poll_source)
        self.run_worker(self._start_source, thread=True, name="spectrum-source")

    def on_unmount(self) -> None:
        self._closing = True
        # Source stop is idempotent. Always call it so a start worker that was
        # between entering start() and setting our flag cannot escape teardown.
        with self._source_operation_lock:
            try:
                self.source.stop()
            except Exception as error:  # noqa: BLE001 - teardown must remain best-effort
                self.log.warning(f"Could not stop spectrum source cleanly: {error}")

    # --------------------------------------------------------- source bridge

    def _start_source(self) -> None:
        try:
            with self._source_operation_lock:
                if self._closing:
                    return
                self.source.start(self._receive_frame)
                self._source_started = True
                if self._closing:
                    self.source.stop()
                    return
            self._call_ui(self._mark_source_started)
        except Exception as error:  # noqa: BLE001 - backend errors belong in the cockpit
            self._call_ui(self._set_error, error)

    def _call_ui(self, callback: Any, *args: Any) -> None:
        if self._closing or not self.is_running:
            return
        try:
            if threading.get_ident() == self._ui_thread_id:
                callback(*args)
            else:
                self.call_from_thread(callback, *args)
        except RuntimeError:
            # The callback raced app shutdown.
            pass

    def _receive_frame(self, frame: SpectrumFrame) -> None:
        if self._closing:
            return
        with self._pending_lock:
            self._frames_received += 1
            if self._pending_frame is not None:
                self._frames_coalesced += 1
            self._pending_frame = frame

    def _drain_pending_frame(self) -> None:
        with self._pending_lock:
            frame, self._pending_frame = self._pending_frame, None
        if frame is not None:
            self._accept_frame(frame)

    def _accept_frame(self, frame: SpectrumFrame) -> None:
        if not frame.frequencies_hz or not frame.powers_db:
            return
        if self.paused or self._display_suspended:
            self._refresh_chrome()
            return
        self._last_source_bins = min(len(frame.frequencies_hz), len(frame.powers_db))
        level_samples = _representative_powers(frame)
        display_frame = _downsample_frame(frame)
        self.last_frame = display_frame
        self._frames_rendered += 1
        self._arrival_times.append(time.monotonic())
        self.receiver_state = "demo" if self.demo else "receiving"
        self.last_error = None
        self.query_one("#spectrum", SpectrumPlot).set_frame(display_frame, self.config)
        self.query_one("#waterfall", WaterfallPlot).push(display_frame)
        self.query_one("#metrics", MetricsPanel).set_data(
            display_frame,
            self.config,
            self.frames_per_second,
            level_samples,
            actual_bin_width_hz=frame.bin_width_hz,
        )
        self._refresh_chrome()

    def _mark_source_started(self) -> None:
        if self.last_frame is None and not self._display_suspended:
            self.receiver_state = "demo-ready" if self.demo else "connected"
        self._refresh_chrome()

    def _set_error(self, error: BaseException | str) -> None:
        self._display_suspended = False
        self.last_error = str(error).strip() or type(error).__name__
        self.receiver_state = "error"
        self._refresh_chrome()

    def _poll_source(self) -> None:
        if self._closing or self._display_suspended:
            self._refresh_chrome()
            return
        error = getattr(self.source, "error", None)
        if error and str(error) != self.last_error:
            self._set_error(error)
            return
        if self.last_error:
            self._refresh_chrome()
            return
        status = str(getattr(self.source, "status", "")).lower()
        running = bool(getattr(self.source, "running", False))
        if self._source_started and not running and status in {"failed", "stopped"}:
            if status == "failed":
                self._set_error(error or "spectrum source failed")
                return
            self.receiver_state = "stopped"
        self._refresh_chrome()

    # ------------------------------------------------------------- rendering

    def _state_text(self) -> tuple[str, str]:
        if self.last_error:
            return "● ERROR", "bold #ff6b68"
        if self.receiver_state == "stopped":
            return "○ SOURCE STOPPED", "bold #ff8c68"
        if self._display_suspended:
            operations = {
                "restarting": ("↻ RESTARTING", "bold #e8bd62"),
                "tuning": ("◈ RETUNING", "bold #e8bd62"),
            }
            if self.receiver_state in operations:
                return operations[self.receiver_state]
        if self.paused:
            return "Ⅱ PAUSED", "bold #e8bd62"
        states = {
            "starting": ("◌ STARTING", "bold #759a9c"),
            "connected": ("● CONNECTED", "bold #54e3c1"),
            "receiving": ("● RECEIVING", "bold #54e3c1"),
            "demo-ready": ("◆ DEMO READY", "bold #81b7ff"),
            "demo": ("◆ DEMO FEED", "bold #81b7ff"),
            "restarting": ("↻ RESTARTING", "bold #e8bd62"),
            "tuning": ("◈ RETUNING", "bold #e8bd62"),
        }
        return states.get(self.receiver_state, (self.receiver_state.upper(), "bold #8fb5b6"))

    def _device_name(self) -> str:
        if self.demo:
            return "SYNTHETIC RF SOURCE"
        if self.device is None:
            return "HACKRF"
        board = getattr(self.device, "board_name", None) or "HackRF"
        serial = getattr(self.device, "serial_number", None) or getattr(self.device, "serial", None)
        if serial:
            serial = str(serial)
            return f"{board} · …{serial[-8:]}"
        return str(board)

    def _refresh_chrome(self) -> None:
        if not self.is_mounted:
            return
        state_label, state_style = self._state_text()
        left = Text()
        left.append("RF", style="bold #68f5d2")
        left.append("//", style="bold #37666e")
        left.append("TUI", style="bold #88baff")
        left.append("  RX SPECTRUM COCKPIT", style="#66888d")
        right = Text(state_label, style=state_style)
        width = max(20, self.query_one("#topbar", Static).content_size.width)
        gap = max(2, width - left.cell_len - right.cell_len)
        header = left + Text(" " * gap) + right
        second = Text()
        second.append(self._device_name(), style="#9cb8b8")
        second.append("  │  ", style="#31545a")
        second.append(
            f"{_frequency(self.config.start_hz)} — {_frequency(self.config.stop_hz)}",
            style="bold #d1dedb",
        )
        second.append("  │  POWER: RELATIVE dB", style="bold #d4b75f")
        self.query_one("#topbar", Static).update(Text.assemble(header, "\n", second))

        self.query_one("#status-state", Static).update(Text(state_label, style=state_style))
        detail = self.last_error or self._status_detail()
        self.query_one("#status-detail", Static).update(detail)

    def _status_detail(self) -> str:
        if self.receiver_state == "stopped":
            return "sweep source exited · press r to restart"
        if self.last_frame is None:
            return "awaiting first sweep frame"
        frequencies, _ = _frame_values(self.last_frame)
        timestamp = self.last_frame.timestamp
        if isinstance(timestamp, datetime):
            sampled = timestamp.astimezone().strftime("%H:%M:%S")
        else:
            try:
                sampled = (
                    datetime.fromtimestamp(float(timestamp), tz=UTC)
                    .astimezone()
                    .strftime("%H:%M:%S")
                )
            except (TypeError, ValueError, OSError):
                sampled = "now"
        return (
            f"frame {self.frames_received:,} · shown {self.frames_rendered:,} · "
            f"{self._bin_detail(len(frequencies))} · "
            f"{self.frames_per_second:.1f} display fps · sampled {sampled}"
        )

    def _bin_detail(self, display_bins: int) -> str:
        if self._last_source_bins > display_bins:
            return f"{self._last_source_bins:,}→{display_bins:,} bins"
        return f"{display_bins:,} bins"

    # --------------------------------------------------------------- actions

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused and not self._display_suspended:
            self._arrival_times.clear()
            if self.last_error:
                self.receiver_state = "error"
            elif self.last_frame is None:
                self.receiver_state = "demo-ready" if self.demo else "connected"
            else:
                self.receiver_state = "demo" if self.demo else "receiving"
        self._refresh_chrome()

    def action_clear(self) -> None:
        self.query_one("#spectrum", SpectrumPlot).clear_peak_hold()
        self.query_one("#waterfall", WaterfallPlot).clear_history()
        self.notify("Peak hold and waterfall cleared", title="Display reset", timeout=2)

    def action_restart(self) -> None:
        generation = self._begin_operation("restarting")
        self.run_worker(
            lambda: self._restart_source(generation),
            thread=True,
            name="restart-source",
            exclusive=True,
        )

    def _restart_source(self, generation: int) -> None:
        try:
            with self._source_operation_lock:
                # Re-check only after serialization. Otherwise an older
                # worker may wake and retune the radio after a newer action.
                if self._closing or generation != self._operation_generation:
                    return
                self.source.restart()
                if self._closing:
                    self.source.stop()
                    return
            self._call_ui(self._operation_complete, generation)
        except Exception as error:  # noqa: BLE001
            self._call_ui(self._operation_failed, generation, error)

    def _operation_complete(self, generation: int) -> None:
        if generation != self._operation_generation or self._closing:
            return
        self._clear_pending_frame()
        self._display_suspended = False
        self._arrival_times.clear()
        if self.last_frame is None:
            self.receiver_state = "demo-ready" if self.demo else "connected"
        else:
            self.receiver_state = "demo" if self.demo else "connected"
        self._refresh_chrome()

    def _operation_failed(self, generation: int, error: BaseException) -> None:
        if generation != self._operation_generation or self._closing:
            return
        was_tuning = self.receiver_state == "tuning"
        self._display_suspended = False
        source_config = getattr(self.source, "config", None)
        if was_tuning and isinstance(source_config, SweepConfig):
            self.config = source_config
            self.query_one("#spectrum", SpectrumPlot).reset(source_config)
            self.query_one("#metrics", MetricsPanel).reset(source_config)
        self._set_error(error)

    def _begin_operation(self, state: str) -> int:
        self._operation_generation += 1
        self._display_suspended = True
        self._clear_pending_frame()
        self.receiver_state = state
        self.last_error = None
        self._refresh_chrome()
        return self._operation_generation

    def _clear_pending_frame(self) -> None:
        with self._pending_lock:
            self._pending_frame = None

    def _reconfigure(self, **changes: int) -> None:
        try:
            candidate = replace(self.config, **changes)
        except ValueError as error:
            self.notify(str(error), title="Invalid tuning", severity="warning")
            return
        if candidate == self.config:
            return
        self.config = candidate
        generation = self._begin_operation("tuning")
        # The x axis changed. Retaining an old trace or indexed waterfall row
        # under new frequency labels would be materially misleading.
        self.last_frame = None
        self._last_source_bins = 0
        self._arrival_times.clear()
        self.query_one("#spectrum", SpectrumPlot).reset(candidate)
        self.query_one("#waterfall", WaterfallPlot).clear_history()
        self.query_one("#metrics", MetricsPanel).reset(candidate)
        self._refresh_chrome()
        self.run_worker(
            lambda: self._apply_config(candidate, generation),
            thread=True,
            name="reconfigure-source",
            exclusive=True,
        )

    def _apply_config(self, config: SweepConfig, generation: int) -> None:
        try:
            with self._source_operation_lock:
                if self._closing or generation != self._operation_generation:
                    return
                reconfigure = getattr(self.source, "reconfigure", None)
                if reconfigure is None:
                    # Compatibility fallback for simple spectrum sources used by
                    # tests and future adapters without live retuning.
                    self.source.config = config
                    self.source.restart()
                else:
                    reconfigure(config)
                if self._closing:
                    self.source.stop()
                    return
            self._call_ui(self._operation_complete, generation)
        except Exception as error:  # noqa: BLE001
            self._call_ui(self._operation_failed, generation, error)

    def _shift(self, direction: int) -> None:
        span = self.config.stop_hz - self.config.start_hz
        start = self.config.start_hz + direction * span
        stop = self.config.stop_hz + direction * span
        if start < MIN_FREQUENCY_HZ:
            start, stop = MIN_FREQUENCY_HZ, MIN_FREQUENCY_HZ + span
        if stop > MAX_FREQUENCY_HZ:
            start, stop = MAX_FREQUENCY_HZ - span, MAX_FREQUENCY_HZ
        self._reconfigure(start_hz=round(start), stop_hz=round(stop))

    def _zoom(self, factor: float) -> None:
        center = (self.config.start_hz + self.config.stop_hz) / 2
        current_span = self.config.stop_hz - self.config.start_hz
        target_span = max(
            MIN_SPAN_HZ,
            min(MAX_FREQUENCY_HZ - MIN_FREQUENCY_HZ, current_span * factor),
        )
        # hackrf_sweep accepts whole-MHz endpoints. Quantize both the span and
        # lower edge so odd spans (such as half of the default 26 MHz) remain
        # valid SweepConfig values.
        target_span = max(MIN_SPAN_HZ, round(target_span / 1_000_000) * 1_000_000)
        start = round((center - target_span / 2) / 1_000_000) * 1_000_000
        stop = start + target_span
        if start < MIN_FREQUENCY_HZ:
            start, stop = MIN_FREQUENCY_HZ, MIN_FREQUENCY_HZ + target_span
        if stop > MAX_FREQUENCY_HZ:
            start, stop = MAX_FREQUENCY_HZ - target_span, MAX_FREQUENCY_HZ
        self._reconfigure(start_hz=int(start), stop_hz=int(stop))

    def action_shift_down(self) -> None:
        self._shift(-1)

    def action_shift_up(self) -> None:
        self._shift(1)

    def action_zoom_out(self) -> None:
        self._zoom(2.0)

    def action_zoom_in(self) -> None:
        self._zoom(0.5)

    def action_lna_down(self) -> None:
        self._reconfigure(lna_gain=max(0, self.config.lna_gain - 8))

    def action_lna_up(self) -> None:
        self._reconfigure(lna_gain=min(40, self.config.lna_gain + 8))

    def action_vga_down(self) -> None:
        self._reconfigure(vga_gain=max(0, self.config.vga_gain - 2))

    def action_vga_up(self) -> None:
        self._reconfigure(vga_gain=min(62, self.config.vga_gain + 2))


__all__ = [
    "HelpScreen",
    "MetricsPanel",
    "RFTUIApp",
    "SpectrumPlot",
    "WaterfallPlot",
]
