"""Pilot-level checks for the RFTUI spectrum cockpit."""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable
from datetime import UTC, datetime

import pytest
from textual.pilot import Pilot
from textual.widgets import Static

from rftui.app import (
    DISPLAY_BINS,
    WATERFALL_BINS,
    HelpScreen,
    MetricsPanel,
    RFTUIApp,
    SpectrumPlot,
    WaterfallPlot,
)
from rftui.model import MAX_FRAME_BINS, SpectrumFrame, SweepConfig


def frame(offset: float = 0.0, bins: int = 64) -> SpectrumFrame:
    """Build a deterministic frame with two visible carriers."""

    start = 902_000_000.0
    step = 26_000_000.0 / (bins - 1)
    frequencies = tuple(start + index * step for index in range(bins))
    powers = []
    for index in range(bins):
        baseline = -91.0 + (index % 7) * 0.7 + offset
        carrier_a = 42.0 / (1.0 + ((index - bins * 0.35) / 2.2) ** 2)
        carrier_b = 27.0 / (1.0 + ((index - bins * 0.72) / 3.0) ** 2)
        powers.append(baseline + carrier_a + carrier_b)
    return SpectrumFrame(
        timestamp=datetime.now(UTC),
        frequencies_hz=frequencies,
        powers_db=tuple(powers),
    )


class FakeSource:
    """Small lifecycle-compatible source used to exercise real UI plumbing."""

    def __init__(self, config: SweepConfig, *, fail: str | None = None) -> None:
        self.config = config
        self.running = False
        self.status = "stopped"
        self.error: str | None = None
        self.callback: Callable[[SpectrumFrame], None] | None = None
        self.started = 0
        self.stopped = 0
        self.restart_calls = 0
        self.reconfigure_calls = 0
        self.fail = fail

    def start(self, callback: Callable[[SpectrumFrame], None]) -> None:
        if self.fail:
            self.error = self.fail
            raise RuntimeError(self.fail)
        self.callback = callback
        self.running = True
        self.status = "running"
        self.started += 1
        callback(frame())

    def stop(self) -> None:
        self.running = False
        self.status = "stopped"
        self.stopped += 1

    def restart(self, callback: Callable[[SpectrumFrame], None] | None = None) -> None:
        self.restart_calls += 1
        if callback is not None:
            self.callback = callback
        self.running = True
        self.status = "running"

    def reconfigure(self, config: SweepConfig) -> None:
        self.config = config
        self.reconfigure_calls += 1

    def emit(self, value: SpectrumFrame) -> None:
        assert self.callback is not None
        self.callback(value)


class BlockingSource(FakeSource):
    """Hold the first retune so a newer UI operation can overtake it."""

    def __init__(self, config: SweepConfig) -> None:
        super().__init__(config)
        self.retune_entered = threading.Event()
        self.release_retune = threading.Event()

    def reconfigure(self, config: SweepConfig) -> None:
        if not self.retune_entered.is_set():
            self.retune_entered.set()
            assert self.release_retune.wait(2)
        super().reconfigure(config)


class SlowStartSource(FakeSource):
    """Expose the teardown race while source.start is in progress."""

    def __init__(self, config: SweepConfig) -> None:
        super().__init__(config)
        self.start_entered = threading.Event()
        self.release_start = threading.Event()

    def start(self, callback: Callable[[SpectrumFrame], None]) -> None:
        self.callback = callback
        self.start_entered.set()
        assert self.release_start.wait(2)
        self.running = True
        self.status = "running"
        self.started += 1


class QuietSource(FakeSource):
    """Start normally but wait for the test to provide the first frame."""

    def start(self, callback: Callable[[SpectrumFrame], None]) -> None:
        self.callback = callback
        self.running = True
        self.status = "running"
        self.started += 1


async def wait_for(predicate: Callable[[], bool], pilot: Pilot[None]) -> None:
    """Wait briefly for a Textual worker without baking in a fixed sleep."""

    for _ in range(40):
        if predicate():
            return
        await pilot.pause(0.025)
    raise AssertionError("condition was not reached")


@pytest.mark.asyncio
async def test_demo_boot_renders_honest_spectrum_and_stops_source() -> None:
    config = SweepConfig()
    source = FakeSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(128, 42)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)

        assert source.started == 1
        assert app.receiver_state == "demo"
        assert app.frames_received >= 1
        assert len(app.query_one("#waterfall", WaterfallPlot).history) == 1

        topbar = app.query_one("#topbar", Static).render()
        assert topbar.plain.splitlines()[0].startswith("RF//TUI")
        assert "HACKRF//TUI" not in topbar.plain
        assert "RX SPECTRUM COCKPIT" in topbar.plain
        assert "POWER: RELATIVE dB" in topbar.plain
        assert "DEMO FEED" in topbar.plain

        spectrum = app.query_one("#spectrum", SpectrumPlot).render().plain
        metrics = app.query_one("#metrics", MetricsPanel).render().plain
        assert "rel dB" in spectrum
        assert "LEVELS · RELATIVE DB" in metrics.upper()
        assert "STRONGEST PEAKS" in metrics

    assert source.stopped == 1


@pytest.mark.asyncio
async def test_metrics_reports_csv_then_received_spacing_compactly() -> None:
    config = SweepConfig(bin_width_hz=100_000)
    source = QuietSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(80, 24)) as pilot:
        await wait_for(lambda: source.started == 1, pilot)
        metrics = app.query_one("#metrics", MetricsPanel)

        # Before samples arrive, use hackrf_sweep's two-decimal CSV plan, not
        # the requested -w value that the FFT geometry may adjust.
        awaiting = metrics.render().plain
        assert "bin 98.039 kHz CSV" in awaiting
        assert "requested 100.000 kHz" in awaiting

        spacing_hz = 125_000.0
        received = SpectrumFrame(
            timestamp=datetime.now(UTC),
            frequencies_hz=tuple(902_000_000.0 + index * spacing_hz for index in range(64)),
            powers_db=frame().powers_db,
        )
        source.emit(received)
        await wait_for(lambda: metrics.actual_bin_width_hz == spacing_hz, pilot)
        live = metrics.render().plain
        assert "bin 125.000 kHz CSV" in live
        assert "requested 100.000 kHz" in live
        assert len(live.splitlines()) <= metrics.content_size.height

        # Exact requests need no redundant requested-width row.
        metrics.reset(SweepConfig(bin_width_hz=1_000_000))
        exact = metrics.render().plain
        assert "bin 1.000 MHz CSV" in exact
        assert "requested" not in exact


@pytest.mark.asyncio
async def test_keyboard_controls_pause_clear_restart_retune_and_help() -> None:
    config = SweepConfig()
    source = FakeSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(128, 42)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)
        first = app.last_frame

        await pilot.press("p")
        assert app.paused
        source.emit(frame(offset=3.0))
        await pilot.pause(0.08)
        assert app.last_frame is first
        assert "PAUSED" in app.query_one("#status-state", Static).render().plain

        await pilot.press("p")
        source.emit(frame(offset=4.0))
        await wait_for(lambda: app.last_frame is not first, pilot)

        waterfall = app.query_one("#waterfall", WaterfallPlot)
        assert waterfall.history
        await pilot.press("c")
        assert not waterfall.history

        await pilot.press("r")
        await wait_for(lambda: source.restart_calls == 1, pilot)

        default_span = app.config.stop_hz - app.config.start_hz
        await pilot.press("plus")
        assert app.config.stop_hz - app.config.start_hz < default_span
        assert app.config.start_hz % 1_000_000 == 0
        assert app.config.stop_hz % 1_000_000 == 0
        await wait_for(lambda: source.reconfigure_calls == 1, pilot)

        original_start = app.config.start_hz
        original_span = app.config.stop_hz - app.config.start_hz
        await pilot.press("right_square_bracket")
        assert app.config.start_hz == original_start + original_span
        await wait_for(lambda: source.reconfigure_calls == 2, pilot)

        await pilot.press("G")
        assert app.config.lna_gain == 24
        await wait_for(lambda: source.reconfigure_calls == 3, pilot)

        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)


@pytest.mark.asyncio
async def test_waterfall_history_is_bounded() -> None:
    config = SweepConfig()
    source = FakeSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(110, 36)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)
        waterfall = app.query_one("#waterfall", WaterfallPlot)
        for index in range(120):
            waterfall.push(frame(offset=index / 100))
        assert len(waterfall.history) == 96
        assert waterfall.history.maxlen == 96
        assert max(len(row.powers) for row in waterfall.history) <= WATERFALL_BINS


@pytest.mark.asyncio
async def test_high_rate_source_coalesces_into_one_bounded_pending_frame() -> None:
    config = SweepConfig()
    source = FakeSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(110, 36)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)
        waterfall = app.query_one("#waterfall", WaterfallPlot)
        rows_before = len(waterfall.history)
        received_before = app.frames_received

        newest = frame(offset=0.0)
        for index in range(500):
            newest = frame(offset=index / 10)
            source.emit(newest)

        # The producer callback only swaps one slot; it does not synchronously
        # paint or grow a frame queue.
        assert app.frames_received == received_before + 500
        assert len(waterfall.history) == rows_before
        assert app.frames_coalesced >= 499

        await wait_for(lambda: app.last_frame is newest, pilot)
        assert len(waterfall.history) <= rows_before + 2


@pytest.mark.asyncio
async def test_maximum_frame_is_bounded_before_entering_ui_history() -> None:
    config = SweepConfig(bin_width_hz=2_445)
    source = FakeSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(110, 36)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)
        rendered_before = app.frames_rendered
        maximum = frame(bins=MAX_FRAME_BINS)
        source.emit(maximum)
        await wait_for(lambda: app.frames_rendered > rendered_before, pilot)

        assert app.last_frame is not None
        assert len(app.last_frame.powers_db) == DISPLAY_BINS
        spectrum = app.query_one("#spectrum", SpectrumPlot)
        metrics = app.query_one("#metrics", MetricsPanel)
        waterfall = app.query_one("#waterfall", WaterfallPlot)
        assert spectrum.frame is app.last_frame
        assert metrics.frame is app.last_frame
        assert len(waterfall.history[-1].powers) == WATERFALL_BINS
        raw_sorted = sorted(maximum.powers_db)
        sampled_sorted = sorted(metrics.level_samples)
        raw_floor = statistics.median(raw_sorted[: len(raw_sorted) // 3])
        sampled_floor = statistics.median(sampled_sorted[: len(sampled_sorted) // 3])
        assert sampled_floor == pytest.approx(raw_floor, abs=0.2)


@pytest.mark.asyncio
async def test_newer_retune_wins_and_old_band_frames_are_discarded() -> None:
    config = SweepConfig()
    source = BlockingSource(config)
    app = RFTUIApp(source, config, demo=True)

    async with app.run_test(size=(110, 36)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)
        await pilot.press("plus")
        await wait_for(source.retune_entered.is_set, pilot)

        # This is an old-band frame arriving while the first retune is held.
        source.emit(frame(offset=20))
        await pilot.press("right_square_bracket")
        newest_config = app.config
        source.release_retune.set()

        await wait_for(
            lambda: source.reconfigure_calls == 2 and app.receiver_state == "demo-ready",
            pilot,
        )
        assert source.config == newest_config
        assert app.config == newest_config
        assert app.last_frame is None
        assert not app.query_one("#waterfall", WaterfallPlot).history


@pytest.mark.asyncio
async def test_clean_unexpected_source_stop_is_not_left_green() -> None:
    config = SweepConfig()
    source = FakeSource(config)
    app = RFTUIApp(source, config)

    async with app.run_test(size=(110, 36)) as pilot:
        await wait_for(lambda: app.last_frame is not None, pilot)
        source.running = False
        source.status = "stopped"
        await wait_for(lambda: app.receiver_state == "stopped", pilot)
        assert "SOURCE STOPPED" in app.query_one("#status-state", Static).render().plain
        assert "press r" in app.query_one("#status-detail", Static).render().plain


def test_start_teardown_race_cannot_leave_source_running() -> None:
    config = SweepConfig()
    source = SlowStartSource(config)
    app = RFTUIApp(source, config)
    worker = threading.Thread(target=app._start_source)
    worker.start()
    assert source.start_entered.wait(1)

    release = threading.Timer(0.05, source.release_start.set)
    release.start()
    app.on_unmount()
    worker.join(1)
    release.join(1)

    assert not worker.is_alive()
    assert not source.running
    assert source.stopped >= 1


@pytest.mark.asyncio
async def test_source_failure_is_visible_in_cockpit() -> None:
    config = SweepConfig()
    source = FakeSource(config, fail="USB interface is busy")
    app = RFTUIApp(source, config)

    async with app.run_test(size=(110, 36)) as pilot:
        await wait_for(lambda: app.receiver_state == "error", pilot)
        assert app.last_error == "USB interface is busy"
        assert "ERROR" in app.query_one("#status-state", Static).render().plain
        assert "USB interface is busy" in app.query_one("#status-detail", Static).render().plain
