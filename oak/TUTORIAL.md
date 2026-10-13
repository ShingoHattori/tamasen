# OAK (Luxonis DepthAI) capture — working notes

Verified on this machine: macOS 24.6.0 (Apple Silicon), `depthai==3.10.0`, **two boards
connected at once over USB3**:

| Board | Device id | Colour (CAM_A) | Mono pair (CAM_B/C) |
|---|---|---|---|
| OAK-D | `14442C10F1B7EED0` | IMX378 4056×3040 | OV9282 1280×800 (16:10) |
| OAK-D-Lite | `1844301091A70A13` | IMX214 4208×3120 | OV7251 **640×480 (4:3)** |

Both are RVC2 and both report `UsbSpeed.SUPER`.

This file exists because several plausible-looking pipelines **hang forever with no error
message**. Read "Traps" before writing any DepthAI code here.

---

## 1. Setup

```bash
uv venv --python 3.11 .venv
uv pip install depthai opencv-python numpy
```

**Use Python 3.11.** The system default here is 3.14, and `depthai` has no wheel for it.
`requirements.txt` and `pyproject.toml` pin the exact versions that were verified.

Sanity check before anything else — it takes a second and tells you the board layout:

```bash
.venv/bin/python -c "
import depthai as dai
for info in dai.Device.getAllAvailableDevices():
    with dai.Device(info) as d:
        print(d.getDeviceName(), d.getDeviceId(), d.getUsbSpeed())
        for f in d.getConnectedCameraFeatures(): print('  ', f.socket, f.sensorName, f.width, f.height)
"
```

If `getUsbSpeed()` says `HIGH` rather than `SUPER` you are on a USB2 cable and will drop frames.

## 2. Run

`oak.py` is the CLI: `list` shows the boards, `serve` streams one to a browser. **There is no
native window** — everything goes over HTTP, which works over SSH and sidesteps macOS screen
permissions entirely.

```bash
.venv/bin/python oak.py list
.venv/bin/python oak.py serve --device 14442 --port 8080
```

`serve` prints the URL to open:

```
==========================================================
  Open this in your browser:

      http://localhost:8080/

  Ctrl-C to stop.
==========================================================
```

Endpoints: `/` (page), `/stream.mjpg` (MJPEG, 30 fps), `/snapshot.jpg` (one frame — handy for
checking the camera from a script with `curl`). Default bind is `127.0.0.1`; `--host 0.0.0.0`
also allows other machines, and the banner then lists the LAN URL too. Other options:
`--layout side|stack`, `--scale`, `--min-mm` / `--max-mm`, `--quality`, `--seconds`.

The batch capture scripts take the same `--device`:

```bash
.venv/bin/python capture_still.py  --device oak-d-lite
.venv/bin/python capture_still.py  --device 14442        # the OAK-D
.venv/bin/python record_video.py 10 --device lite
```

With more than one board plugged in you **must** say which one; the scripts refuse to guess.
`--device` takes a substring of either the device id or the product name. **Prefer the device id**
(`--device 14442`) — see 3.5 for why a product-name substring can bite.

Output lands in `output/<device-name>/`, so the two boards never overwrite each other.

Measured on both boards, 1280×720 colour + 1280×720 aligned depth:

| Board | Video | Valid depth pixels (indoor ceiling scene) |
|---|---|---|
| OAK-D | 150 frames / 5.0 s = **29.9 fps** | 2.9 – 75.7 % across runs |
| OAK-D-Lite | 150 frames / 5.0 s = **29.9 fps** | 31.4 – 32.7 % across runs |

Note how wildly the OAK-D's valid-pixel count swings between runs: that is the scene (see 3.8),
not the board. Do not treat a single measurement as a regression signal.

Both boards can also be opened and streamed **simultaneously** from one process — verified with
two `dai.Pipeline(device)` objects running at once.

---

## 3. Traps

### 3.1 `setDepthAlign()` without `setOutputSize()` hangs the pipeline

**This is the one that will cost you an hour.** There is no exception, no warning, no log line —
`queue.get()` simply blocks forever.

```python
# HANGS: never delivers a single frame
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

# WORKS
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(1280, 720)
```

Isolated on this machine with two scripts differing only in that one call: with `setOutputSize`,
5 colour+depth pairs arrived immediately; without it, the process produced zero frames and was
still blocked in `get()` after 136 s before being killed.

So: **whenever you call `setDepthAlign`, call `setOutputSize` too.** Give it the same dimensions
as the colour output — that is what makes the two images overlay pixel-for-pixel anyway.

(A tempting wrong diagnosis: blocking output queues. That was tested and **is not** the cause —
the default blocking queues complete 30 frames fine. The scripts still pass
`maxSize=4, blocking=False` because dropping stale frames is the right behaviour for capture,
not because it fixes a hang.)

### 3.2 Do not hard-code the mono resolution — the two boards differ

OAK-D's OV9282 is 1280×800 (16:10); OAK-D-Lite's OV7251 is **640×480 (4:3)**. Asking a Lite for
640×400 *appears* to work, but it silently crops away vertical field of view. `oak_common.mono_size()`
reads the native size from `getConnectedCameraFeatures()` and halves it only when it is larger
than 640, giving 640×400 on the OAK-D and 640×480 on the Lite.

The Lite also has **no IMU**, so any `dai.node.IMU` code will work on the OAK-D and fail on the Lite.

### 3.3 A board is unavailable for a second or two after the previous run

Running two scripts back to back against the same board fails with
`RuntimeError: Failed to find device after booting, error message: X_LINK_DEVICE_NOT_FOUND` —
the board is still re-enumerating over USB after the first process released it. This bit the
product-name probe in particular, because that probe opens *every* connected board in turn, so a
script targeting the Lite could trip over an OAK-D that was mid-reboot.

`oak_common` handles it: `_available()` retries the enumeration, `_open()` retries the open for
up to 20 s, and the name probe skips boards it cannot open instead of dying. If you write your
own entry point, do the same — do not assume `dai.Device(info)` succeeds on the first try.

### 3.4 Killing a streaming process crashes the board's firmware

`pkill` on a process that is mid-stream produced:

```
[depthai] [error] Device with id 1844301091A70A13 has crashed. Crash dump logs are stored in:
~/Library/Caches/depthai/crashdumps/...
```

and the next run could not open the board. The cause is the device never being closed: a
generator that holds `with dai.Pipeline(...)` open does **not** run its `__exit__` when the
consumer simply stops iterating or the thread is killed.

Two things fix it, and both are in `oak.py`:

- keep a reference to the generator and call `source.close()` in a `finally`, which runs the
  pipeline's `__exit__` and releases the board;
- install a `SIGTERM` handler that shuts down the same way `Ctrl-C` does, then `join()` the
  capture thread before exiting.

With those in place, `pkill -TERM` mid-stream leaves both boards immediately reusable — verified.
If you do wedge a board, it recovers on its own after a few seconds; re-check with `oak.py list`.

### 3.5 A product-name substring can silently open the *wrong* board

`oak-d` is a substring of `oak-d-lite`. Normally the exact-name rule handles it: probing finds a
board actually named `OAK-D` and that wins. But USB enumeration is racy (3.3), and when the real
OAK-D was momentarily missing from the device list, `--device oak-d` matched the Lite by
substring and streamed it — while the log said `OAK-D-LITE`, which is easy to skim past.

Two mitigations, both in place:

- `available()` samples `getAllAvailableDevices()` several times and unions the results by id,
  instead of trusting one call. This was found the same way: `oak.py list` once reported
  **1 device** while `system_profiler SPUSBDataType` showed **2 Movidius MyriadX**.
- a non-exact name match now prints
  `note: 'lite' matched product name 'OAK-D-LITE' by substring, not exactly.`

**Use the device id in anything scripted or repeatable.** It is unambiguous and skips the probe
(which has to open every board, taking several seconds).

### 3.6 `DeviceInfo.name` is a USB port path, not a product name

Before a board is booted, `DeviceInfo.name` is `"3.1"` / `"1.1"` — the USB topology path.
`getDeviceName()` returns `"OAK-D"` / `"OAK-D-LITE"` but **only after opening the device**.
Selecting a board by product name therefore means opening each candidate and asking it;
`oak_common.pick_device()` does that as a fallback after trying the device id.

### 3.7 `StereoDepth` silently downscales

`PresetMode.DEFAULT` returns **320×200** depth even though the mono inputs are 640×400. If you
assumed the depth map matches your mono resolution, your indexing is wrong. `setOutputSize()`
fixes this too — another reason to always call it.

### 3.8 Aspect ratio mismatches are everywhere on these boards

Mono is 16:10 (OAK-D) or 4:3 (Lite); the colour sensors are 4:3; a 1280×720 request is 16:9.
Nothing lines up by default. Pick one output size and use it for **both** the colour request and
`setOutputSize()`, and let depth alignment handle the rest.

### 3.9 depthai 3.x is not depthai 2.x

Almost every tutorial and Stack Overflow answer you will find is for the 2.x API and will not run.

| 2.x (obsolete) | 3.x (this project) |
|---|---|
| `dai.node.ColorCamera` + `setPreviewSize` | `pipeline.create(dai.node.Camera).build(socket)` |
| `dai.node.MonoCamera` | same `Camera` node, on CAM_B / CAM_C |
| `XLinkOut` + `device.getOutputQueue(name)` | `node.requestOutput(size, type).createOutputQueue()` |
| `dai.Device(pipeline)` | `dai.Pipeline(device)`, then `pipeline.start()` |

Do not port 2.x code by guessing; check `dir()` on the node.

### 3.10 Low depth coverage is usually the scene, not a bug

Pointed at a blank white ceiling, the Lite reported **31.7 %** valid pixels and the OAK-D
**75.7 %**. Stereo needs visual texture to match between the two views; flat painted walls, glass
and glare produce no match and come back as `0`. Before debugging code, point the camera at
something textured. Always mask `depth == 0` out of any statistic, or your mean distance is
meaningless. The Lite's narrower baseline and lower-resolution mono pair make it consistently
worse than the OAK-D on the same scene — that is the hardware, not your code.

### 3.11 Save depth as 16-bit PNG, never JPEG or mp4

`stereo.depth` is **uint16 millimetres**. Write it with `cv2.imwrite(..., depth.astype(np.uint16))`
to PNG. JPEG and mp4 are 8-bit and will destroy the measurements. For video, colourise a *copy*
and keep raw depth separately if you need the numbers.

### 3.12 Colourise video with a fixed range

A per-frame percentile stretch (fine for a single still) makes video strobe as the scene changes.
`record_video.py` pins the map to 300–6000 mm.

### 3.13 Warm up before you show or save anything

The first frames are dark and colour-shifted while auto-exposure and white balance converge.
Every entry point here discards ~30 frames first (`frames()` does it centrally). Skipping this
gives you a black or green-tinted image and a false bug report — it caught out a debug
"grab one frame" path that took frame 1 and produced a badly green picture.

### 3.14 Always use `with dai.Pipeline(...)`, and kill hung processes

Each board is a single-owner USB resource. A crashed or hung Python process keeps it claimed and
the next run fails to open it. If a run hangs: `pkill -f <script>.py`, then re-probe with
`dai.Device.getAllAvailableDevices()`.

---

## 4. Environment / shell gotchas on this machine

- **`timeout` does not exist on macOS.** `timeout 90 python foo.py` fails with
  `command not found`, which looks like a Python error but is not. Use `gtimeout` (coreutils) or
  a shell loop.
- **Do not pipe a probe script through `tail`.** `python -u foo.py 2>&1 | tail -20` buffers in
  `tail` and shows nothing until exit — so a hang looks identical to a silent success. When
  diagnosing, run unpiped with `python -u` and `print(..., flush=True)`.
- **`pgrep -f foo.py` also matches your own watcher shell** if the script text is in its command
  line. Wait on the PID (`kill -0 $PID`) instead.
- DepthAI writes a lot of `[timestamp] [device] [info]` lines to stderr; `| grep -v "^\["` keeps
  the real output readable.

---

## 5. Files

| Path | What |
|---|---|
| `oak.py` | CLI: `list` / `serve` (browser streaming) |
| `oak_common.py` | device selection and per-board mono resolution |
| `capture_still.py` | one colour + one depth still |
| `record_video.py` | colour, depth and side-by-side mp4 |
| `pyproject.toml`, `requirements.txt` | pinned, verified versions |
| `venv.tar.gz` | archived `.venv` (317 MB → 116 MB) |
| `output/oak-d/`, `output/oak-d-lite/` | captures, one directory per board |

### Restoring the archived venv

```bash
tar -xzf venv.tar.gz
.venv/bin/python -c "import depthai; print(depthai.__version__)"
```

A venv contains absolute paths, so the archive only works when unpacked at
`/Users/shatto/works/tamasen/oak/.venv`. **Anywhere else, rebuild instead** — it is faster and
gives the same versions:

```bash
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
```
