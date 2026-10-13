"""Shared device handling for the OAK capture scripts.

Both OAK-D and OAK-D-Lite are supported, and they do NOT have the same sensors:

  OAK-D       CAM_A IMX378 4056x3040   CAM_B/C OV9282 1280x800 (16:10)
  OAK-D-LITE  CAM_A IMX214 4208x3120   CAM_B/C OV7251  640x480 (4:3)

So the mono resolution cannot be hard-coded. Asking a Lite for 640x400 does work, but
silently crops away vertical field of view; mono_size() picks a native-aspect size instead.
"""
import sys
import time

import depthai as dai

# A board that was just released re-enumerates over USB a second or two later, so a script
# run right after another one can see an empty or stale device list.
RETRY_SECONDS = 20.0


def available(timeout: float = RETRY_SECONDS, samples: int = 3, settle: float = 1.2) -> list:
    """All connected boards, de-duplicated by device id.

    USB enumeration is racy: right after another run, `getAllAvailableDevices()` can return a
    board that is still re-enumerating, or miss one entirely — observed as `oak.py list`
    reporting 1 device while the USB tree showed 2. So sample it a few times and union the
    results instead of trusting a single call.
    """
    deadline = time.monotonic() + timeout
    while True:
        seen = {}
        for i in range(samples):
            for info in dai.Device.getAllAvailableDevices():
                seen.setdefault(info.getDeviceId(), info)
            if i < samples - 1:
                time.sleep(settle / samples)
        if seen or time.monotonic() > deadline:
            return list(seen.values())


def open_device(info, timeout: float = RETRY_SECONDS) -> dai.Device:
    """Open a device, retrying while it is still rebooting from a previous run."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return dai.Device(info)
        except RuntimeError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1.0)


def pick_device(selector: str | None = None) -> dai.Device:
    """Open a device. `selector` matches a substring of the device id or product name.

    With several boards plugged in and no selector, this refuses to guess rather than
    grabbing whichever enumerated first.
    """
    infos = available()
    if not infos:
        sys.exit("No OAK device found. Check the USB3 cable and try replugging.")

    if selector:
        sel = selector.lower()
        matches = [i for i in infos if sel in i.getDeviceId().lower()]
        inexact = None
        if not matches:
            # DeviceInfo.name is the USB port path ("3.1") until the board is booted, so the
            # product name is only reachable by opening each device and asking it. A board
            # still rebooting from a previous run cannot be probed; skip it rather than die.
            print(f"looking for {selector!r} (opening each board to read its name)...", flush=True)
            for info in infos:
                try:
                    with open_device(info) as probe:
                        name = probe.getDeviceName().lower()
                except RuntimeError:
                    continue
                if name == sel:  # exact name wins outright over any substring hit
                    matches, inexact = [info], None
                    break
                if sel in name:
                    matches.append(info)
                    inexact = probe_name = name
        if not matches:
            sys.exit(f"No device matching {selector!r}. Connected: "
                     + ", ".join(i.getDeviceId() for i in infos))
        if len(matches) > 1:
            sys.exit(f"{selector!r} matches {len(matches)} devices; be more specific.")
        if inexact:
            # "oak-d" is a substring of "oak-d-lite", so if the real OAK-D happens to be missing
            # from this enumeration pass, a bare substring match silently opens the OTHER board.
            # Say so loudly; select by device id (--device 14442) to make it unambiguous.
            print(f"note: {selector!r} matched product name {inexact.upper()!r} by substring, "
                  f"not exactly. Use --device {matches[0].getDeviceId()[:5]} to be unambiguous.",
                  flush=True)
        infos = matches
    elif len(infos) > 1:
        ids = ", ".join(i.getDeviceId() for i in infos)
        sys.exit(f"{len(infos)} devices connected ({ids}). Pass --device <id-or-name-substring>.")

    return open_device(infos[0])


def mono_size(device: dai.Device) -> tuple[int, int]:
    """Stereo input size for this board, at the mono sensor's native aspect ratio."""
    for f in device.getConnectedCameraFeatures():
        if f.socket == dai.CameraBoardSocket.CAM_B:
            # OV9282 (1280x800) is halved to keep the stereo engine fast; OV7251 (640x480)
            # is already small enough to use as-is.
            return (f.width // 2, f.height // 2) if f.width > 640 else (f.width, f.height)
    sys.exit("No CAM_B (left mono) on this device — depth is not available.")


def describe(device: dai.Device) -> str:
    return (f"{device.getDeviceName()} {device.getDeviceId()} "
            f"{device.getPlatformAsString()} {device.getUsbSpeed()}")
