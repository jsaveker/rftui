# RFTUI

RFTUI is a keyboard-first spectrum and waterfall cockpit for software-defined
radios. Its first—and currently only—hardware backend is HackRF through
`hackrf_sweep`.

The current release is receive-only: it contains no transmit command or API. It
keeps raw sample volume outside the TUI and labels power as relative dB rather
than calibrated dBm. A deterministic demo mode is included for development
without hardware.

## Quick start

For the current HackRF backend, install the host tools and `uv` first. On macOS
with Homebrew:

```bash
brew install hackrf uv
```

If a PortaPack is attached, select **HackRF** from its main menu and confirm the
mode before starting the app.

```bash
uv sync --extra dev
uv run rftui
```

The default view covers the 902–928 MHz ISM band and requests 100 kHz bins.
`hackrf_sweep` selects the nearest supported FFT geometry; RFTUI shows the
effective spacing in the receiver panel. Choose another whole-MHz range or bin
width:

```bash
uv run rftui --start 433 --stop 435 --bin-width 50000
```

Run without hardware:

```bash
uv run rftui --demo
```

List connected radios:

```bash
uv run rftui --list-devices
```

If that reports no radio, run `hackrf_info` directly. For a PortaPack, select
**HackRF** mode on the device, use the HackRF data port with a data-capable USB
cable, and reconnect it. RFTUI never flashes or reconfigures device
firmware.

## Controls

| Key | Action |
| --- | --- |
| `q` | Quit |
| `?` | Help |
| `p` | Pause or resume display updates |
| `r` | Restart the sweep process |
| `[` / `]` | Shift the selected band down or up by one span |
| `-` / `+` | Zoom out or in around the band centre |
| `g` / `G` | Decrease or increase LNA gain |
| `v` / `V` | Decrease or increase VGA gain |
| `c` | Clear peak hold and waterfall history |

## Safety and measurement honesty

- RFTUI's current HackRF backend does not transmit.
- Other SDR backends are not implemented yet.
- A HackRF is a half-duplex SDR, not a drop-in Meshtastic or MeshCore companion
  radio. Protocol decoding would require a separate LoRa physical-layer decoder.
- Displayed power is relative. Do not interpret it as calibrated RF power.
- Use an antenna appropriate for the selected band and respect local law.

[HackRF](https://greatscottgadgets.com/hackrf/) is an open-source hardware
project from [Great Scott Gadgets](https://greatscottgadgets.com/). PortaPack
users may also be running the independent
[Mayhem firmware](https://github.com/portapack-mayhem/mayhem-firmware).
RFTUI is not affiliated with or endorsed by either project.
