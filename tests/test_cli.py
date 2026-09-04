from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pytest

from rftui import cli


@dataclass(frozen=True)
class FakeDevice:
    index: int
    serial_number: str
    board_name: str = "HackRF One"
    firmware_version: str = "test-fw"


class FakeApp:
    instances: ClassVar[list[FakeApp]] = []

    def __init__(self, source, config, *, device=None, demo=False):
        self.source = source
        self.config = config
        self.device = device
        self.demo = demo
        self.ran = False
        self.instances.append(self)

    def run(self):
        self.ran = True


@pytest.fixture(autouse=True)
def clear_fake_apps():
    FakeApp.instances.clear()


def test_parse_defaults():
    args = cli.parse_args([])

    assert args.start == 902
    assert args.stop == 928
    assert args.bin_width == 100_000
    assert args.lna_gain == 16
    assert args.vga_gain == 20
    assert args.serial is None


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--start", "928", "--stop", "902"], "--start must be lower"),
        (["--start", "nan"], "must be finite"),
        (["--start", "0"], "--start must be between"),
        (["--stop", "6001"], "--stop must be between"),
        (["--start", "433.05", "--stop", "435"], "must be whole MHz"),
        (["--bin-width", "2444"], "--bin-width must be between"),
        (["--lna-gain", "10"], "--lna-gain must be"),
        (["--vga-gain", "3"], "--vga-gain must be"),
        (["--demo", "--serial", "abc"], "--serial cannot be used"),
        (["--serial", "not a serial"], "--serial may contain only"),
    ],
)
def test_invalid_arguments_exit_with_useful_error(arguments, message, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.parse_args(arguments)

    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_demo_builds_config_and_launches_without_device_probe(monkeypatch):
    class FakeDemoSource:
        def __init__(self, config):
            self.config = config

    def unexpected_probe():
        raise AssertionError("demo mode must not inspect hardware")

    monkeypatch.setattr(cli, "DemoSpectrumSource", FakeDemoSource)
    monkeypatch.setattr(cli, "RFTUIApp", FakeApp)
    monkeypatch.setattr(cli, "list_devices", unexpected_probe)

    result = cli.main(
        [
            "--demo",
            "--start",
            "433",
            "--stop",
            "435",
            "--bin-width",
            "100000",
            "--lna-gain",
            "24",
            "--vga-gain",
            "32",
        ]
    )

    assert result == 0
    app = FakeApp.instances[0]
    assert app.ran
    assert app.demo
    assert app.device is None
    assert app.config.start_hz == 433_000_000
    assert app.config.stop_hz == 435_000_000
    assert app.config.bin_width_hz == 100_000
    assert app.config.lna_gain == 24
    assert app.config.vga_gain == 32


def test_real_mode_selects_serial_and_launches(monkeypatch):
    devices = [FakeDevice(0, "first"), FakeDevice(1, "ABC123")]

    class FakeHardwareSource:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(cli, "list_devices", lambda: devices)
    monkeypatch.setattr(cli, "HackRFSweepSource", FakeHardwareSource)
    monkeypatch.setattr(cli, "RFTUIApp", FakeApp)

    result = cli.main(["--serial", "abc123"])

    assert result == 0
    app = FakeApp.instances[0]
    assert app.ran
    assert app.device is devices[1]
    assert app.config.serial == "ABC123"


def test_real_mode_reports_no_device(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_devices", list)

    assert cli.main([]) == 1
    error = capsys.readouterr().err
    assert "No HackRF device was found" in error
    assert "PortaPack" in error


def test_real_mode_reports_unknown_serial(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_devices", lambda: [FakeDevice(0, "known")])

    assert cli.main(["--serial", "missing"]) == 1
    error = capsys.readouterr().err
    assert "missing" in error
    assert "known" in error


def test_missing_host_tool_has_install_hint(monkeypatch, capsys):
    def missing_tool():
        raise cli.ToolNotFoundError("hackrf_info was not found")

    monkeypatch.setattr(cli, "list_devices", missing_tool)

    assert cli.main(["--list-devices"]) == 1
    error = capsys.readouterr().err
    assert "HackRF host tools were not found" in error
    assert "brew install hackrf" in error


def test_list_devices_prints_identity_without_launching(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_devices", lambda: [FakeDevice(0, "ABC123")])
    monkeypatch.setattr(cli, "RFTUIApp", FakeApp)

    assert cli.main(["--list-devices"]) == 0
    output = capsys.readouterr().out
    assert "Found 1 HackRF device:" in output
    assert "[0] HackRF One" in output
    assert "serial ABC123" in output
    assert "firmware test-fw" in output
    assert FakeApp.instances == []


def test_list_devices_empty_is_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_devices", list)

    assert cli.main(["--list-devices"]) == 1
    assert "No HackRF devices found" in capsys.readouterr().out
