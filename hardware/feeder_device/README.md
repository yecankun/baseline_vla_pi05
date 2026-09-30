# Replacement Guidewire Feeder Device

This directory contains the cleaned project-facing adapter for the replacement
guidewire feeder. The original received files are intentionally preserved under:

```text
reference_code/feeder_device_raw/
```

Do not edit or import the raw UI directly into data-collection code. Use this
directory for stable wrappers after the real protocol is confirmed.

## Current Network Protocol

The currently reported real device interface is UDP JSON:

```text
target: 192.168.5.5:8888/udp (updated by the operator, 2026-09-30)
payload: {"command":"move","parameters":{"action":"forward","value":1}}
```

Supported `action` values:

```text
forward
backward
turn_left
turn_right
```

Each `forward` or `backward` packet executes one feeder step. `value` is
currently meaningful for `turn_left` and `turn_right`, where it represents the
rotation angle in degrees.

For the 2026-09-30 bench run, the operator reported a fixed 20 mm forward
step; `forward.value` does not adjust that distance. The CLI retains the
historical `192.168.5.13` default, so pass `--host 192.168.5.5` explicitly
for the current device, as in the examples below.

Dry-run one forward packet:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device --host 192.168.5.5 --action forward
```

Actually send one forward packet:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device --host 192.168.5.5 --action forward --execute
```

Actually send one backward packet:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device --host 192.168.5.5 --action backward --execute
```

Rotate left by 15 degrees:

```bash
python -m hardware.feeder_device.probe_udp_feeder_device --host 192.168.5.5 --action turn_left --value 15 --execute
```

## Current Raw Evidence

The older/raw received code was serial-based. Keep it as reference evidence and
as a fallback only if the UDP device path is unavailable.

Raw files inspected:

- `reference_code/feeder_device_raw/int_surg_sys_ctrl_ui.py`
- `reference_code/feeder_device_raw/readme.md`
- `reference_code/feeder_device_raw/1int_surg_sys_ctrl_embed_ver_26041401/USER/main.c`
- `reference_code/feeder_device_raw/1int_surg_sys_ctrl_embed_ver_26041401/readme_260414.md`

Observed serial ports from the raw Python UI:

- STM32 control: `COM7`, `9600` baud.
- Rotation motor control: `COM4`, `115200` baud.

Observed STM32 frame shape:

```text
[command_byte, 0x0D, 0x0A]
```

Important command candidates:

| Meaning | Current firmware evidence | Raw UI/readme evidence | Status |
| --- | --- | --- | --- |
| guidewire forward | `0x00 0D 0A` | `0x00 0D 0A` | likely valid |
| guidewire backward | `0x03 0D 0A` | `0xFF 0D 0A` | must probe |
| lock | no matching current branch found | `0x0F 0D 0A` | unverified |
| unlock | no matching current branch found | `0xF0 0D 0A` | unverified |

The raw UI also builds an 11-byte rotation-motor frame:

```text
0x7B 0x01 0x02 direction 0x20 angle_hi angle_lo 0x00 0x64 bcc 0x7D
```

where `direction=0x01` is clockwise and `direction=0x00` is counterclockwise in
the UI naming.

## Probe Commands

These serial probe commands are legacy/fallback commands for the raw embedded
code path. Prefer the UDP probe above for the current feeder device.

Install the hardware dependency in the real-control environment if needed:

```bash
python -m pip install pyserial
```

Dry-run a single feed command:

```bash
python -m hardware.feeder_device.probe_feeder_device --stm-port COM7 --command feed
```

Actually send one feed primitive:

```bash
python -m hardware.feeder_device.probe_feeder_device --stm-port COM7 --command feed --execute
```

Probe the current-firmware backward command:

```bash
python -m hardware.feeder_device.probe_feeder_device --stm-port COM7 --command retract --execute
```

Probe the legacy/raw-UI backward command:

```bash
python -m hardware.feeder_device.probe_feeder_device --stm-port COM7 --command retract-legacy --execute
```

On Ubuntu, replace `COM7` with the actual device path, for example
`/dev/ttyUSB0` or `/dev/ttyACM0`.

## Integration Rule

The real collection path should depend on a high-level feeder interface:

```text
feed_once()
retract_once()
get_state/log_response()
stop/fault handling, if supported by firmware
```

Do not map policy labels to raw bytes inside training or simulation code. The
controller layer should own this mapping and log executed commands, response
bytes, busy state, and any hardware fault.
