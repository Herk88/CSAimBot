# CSAimbot - Aim Bot for FPS Games
## Using Tensorflow Object Detection API
##### Tested on Counter Strike 1.6 (Should work on most FPS games)

Inspired from Sentdex's Work on his "Python Plays GTA V" series
Thanks to,
1) Sentdex (https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ)

2) Daniel Kukiela (https://twitter.com/daniel_kukiela) for keys.py file

3) Tensorflow Object Detection API

## Instructions
1) Proceed this tutorial on the below link and get Tensorflow Object Detection API Working.

    (https://goo.gl/ricx6n)

2) Clone this repository and extract in your desired location.

3) Copy paste the object_detection folder from the Step 1 into the CSAimBot cloned folder.

4) Get the game running in windowed mode on top left corner of the screen

   Use Borderless gaming to remove the title bar of the game window (not necessary,but recommended)
   (http://westechsolutions.net/sites/WindowedBorderlessGaming/download)

5) Command Line Arguments available:
```
    --help  : Displays all the available arguments and their usage.
    --width : Width of the game resolution(default:800)
    --height : Height of the game resolution(default:600)
    --resize : Keep this as low as possible to get better detection of person but decreasing it also reduces the frame rate of what                    bot sees. (default:4)
    --score : Increase as long as the bot detects the person,Decrease if bot can't detect the person.(default:0.40)
    --show : Set to False if you don't want to see the captured screen(default:True)
    --input : (Enter Without Quotes)Choose between "keyboard" and "mouse".(default:keyboard).Choose the --key if chose keyboard)
    --key : (Enter Without Quotes)Choose Anyone from 
    --shoot : (Enter Without Quotes) Shoots at CENTER of the person detected by default(choose between:CENTER,HEAD,NECK)
    --duration : How long to shoot(in seconds),default:0.4 seconds
```
    
6) Run the Run_Me.py file (Make sure CSAimBot directory is the current working directory,if not python will throw error)
    ```
    python Run_Me.py
    ```
    if you want use custom settings from Step 5,Example usage:
    ```
    python Run_Me.py --width 1024 -- height 768 --resize 3 --score 0.50 --show False --input mouse --shoot HEAD --duration 0.2 
    ```

## Controller Prediction Profiles
- Install the controller backends before launching the bot so velocity hints stay live:
  ```bash
  pip install inputs hidapi psutil
  ```
  `inputs` feeds stick deltas, `hidapi` enables high-rate USB devices, and `psutil` lets the runtime pin CPU affinity and priority.
- Pick a connection profile that matches your link budget. Each profile loads tuned Kalman and controller defaults:
  - `--profile usb` (or `--connection usb`) — 500 Hz predictor, tight noise model, 7 ms p95 target.
  - `--profile bluetooth` — 200 Hz predictor, higher measurement variance, 12 ms p95 target.
- Override any part of the stack from the CLI (mirrored in the Tk GUI — blank fields inherit the profile):
  - Timing & controller: `--loop-hz`, `--controller-deadzone`, `--controller-sensitivity`, `--controller-smoothing`, `--controller-polling`.
  - Kalman: `--q-process`, `--r-meas-px`, `--p0-pos`, `--p0-vel`, `--drop-grace-frames`, `--vel-hint-decay`, `--min-prediction-dt`, `--max-prediction-dt`.
  - Logging: `--perf-log perf.csv` writes deterministic timing traces to a custom path (default: timestamped in the CWD).

## Deterministic Timing & Instrumentation
- `timing.FixedTicker` keeps the predictor on a fixed step and guards against drift. The runner pins itself to CPU core 0 and lifts priority when `psutil` is present (Windows → `HIGH_PRIORITY_CLASS`, Linux/Unix → `nice -10`).
- `perfprobe.PerfProbe` emits frame→command spans to CSV every run and prints p95 latency/jitter at shutdown. Use the overlay Lag meter (top left) to confirm the loop stays beneath the profile thresholds (7 ms / 3 ms for USB, 12 ms / 6 ms for BT).
- A rolling uncertainty ellipse is rendered around the aim point whenever detections are active; this visualises `P[:2,:2]` at 2σ so you can see filter confidence breathe in real time.
- Hook the probe up to your release checks: target USB p95 < 7 ms and jitter < 3 ms; Bluetooth p95 < 12 ms and jitter < 6 ms.

## Tuning Cheatsheet
- Heuristic: overshoot or oscillation? Bump `--r-meas-px`. Falling behind on sharp accelerations? Increase `--q-process`.
- Use `autotune.py` to sweep `q`/`r` on recorded traces:
  ```python
  from autotune import sweep
  result = sweep(trace, q_values=np.geomspace(10, 80, 10), r_values=np.geomspace(40, 200, 10), base_params=KalmanParams())
  print(result)
  ```
  Commit the winning pair (and, if needed, `p0_*`) back into the active profile.
- Profiles ship with conservative defaults:

  | Profile | loop_hz | q_process | r_meas_px | p0_pos | p0_vel | drop_grace | vel_hint_decay |
  |---------|--------|-----------|-----------|--------|--------|------------|----------------|
  | USB     | 500    | 20.0      | 64.0      | 5 000  | 200    | 8          | 0.88           |
  | BT      | 200    | 35.0      | 110.0     | 7 000  | 350    | 12         | 0.82           |

  (Both keep the predictor ≥2× the detection FPS for stability.)

## Visualisation & Camera Notes
- The preview now renders: detection boxes, aim point, 2σ uncertainty ellipse, and a live lag meter (green when within spec, red if limits are exceeded).
- HighGUI availability is auto-detected at runtime; if the current OpenCV build lacks GUI support the preview falls back to headless mode. To restore windows, uninstall any headless wheels and install `opencv-contrib-python==4.12.0.84` so `Win32 UI` (or your platform equivalent) reports `YES` inside `cv2.getBuildInformation()`.
- For capture sources, lock exposure (2–8 ms), gain, and disable auto features; prefer 720p@120 or 1080p@60 with short shutter. If your backend supports it, set `CAP_PROP_BUFFERSIZE = 1` and grab on a dedicated thread to minimise buffering.

## Testing & Next Steps
- Run quick in-game smoke tests on both USB and Bluetooth hardware after tweaking parameters — confirm the lag meter stays green and the ellipse collapses quickly after re-acquisition.
- Gap tests: drop measurements for 250 ms and check that predictions coast smoothly (no aim spikes) and relock within ~150 ms once detections resume.
- When ready for heavier optimisation, move vision into a separate process (shared-memory tensor ring buffer) or wire gyro/IMU hints into the measurement vector for high-G targets.
