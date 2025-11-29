__author__ = 'Vijayabhaskar J'

import argparse
import os
import tarfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import sys

from highgui_guard import destroy_all_windows_safe, has_highgui, highgui_probe_reason, imshow_safe, waitkey_safe

try:
    import tensorflow as tf
except ImportError as exc:
    raise SystemExit(
        "TensorFlow is required to run CSAimbot. Install it with `pip install tensorflow`."
    ) from exc

tf1 = tf.compat.v1
tf1.disable_eager_execution()

import keys as k
from controller_input import ControllerInputReader
from grabscreen import grab_screen
from perfprobe import PerfProbe
from predictor import KalmanParams, KalmanPredictor
from timing import FixedTicker


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = f'{MODEL_NAME}.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_URL = DOWNLOAD_BASE + MODEL_FILE

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORK_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
PATH_TO_CKPT = os.path.join(MODEL_PATH, 'frozen_inference_graph.pb')

WINDOW_NAME = 'CSAimbot - Press q to quit bot'
PERSON_CLASS_ID = 1

CONTROLLER_PRESETS: Dict[str, Dict[str, object]] = {
    'usb': {
        'loop_hz': 500,
        'kalman': {
            'q_process': 20.0,
            'r_meas_px': 64.0,
            'p0_pos': 5_000.0,
            'p0_vel': 200.0,
            'drop_grace_frames': 8,
            'vel_hint_decay': 0.88,
            'min_dt': 1.0 / 800.0,
            'max_dt': 1.0 / 200.0,
        },
        'controller': {
            'poll_hz': 1000.0,
            'deadzone': 0.03,
            'sensitivity': 1.0,
            'smoothing': 0.10,
        },
        'targets': {
            'latency_p95_ms': 7.0,
            'jitter_ms': 3.0,
        },
    },
    'bluetooth': {
        'loop_hz': 200,
        'kalman': {
            'q_process': 35.0,
            'r_meas_px': 110.0,
            'p0_pos': 7_000.0,
            'p0_vel': 350.0,
            'drop_grace_frames': 12,
            'vel_hint_decay': 0.82,
            'min_dt': 1.0 / 400.0,
            'max_dt': 1.0 / 80.0,
        },
        'controller': {
            'poll_hz': 125.0,
            'deadzone': 0.05,
            'sensitivity': 1.10,
            'smoothing': 0.25,
        },
        'targets': {
            'latency_p95_ms': 12.0,
            'jitter_ms': 6.0,
        },
    },
}


def ellipse_from_cov(cov: np.ndarray, k: float = 2.0) -> Optional[Tuple[float, float, float]]:
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    if vals[0] <= 0.0 or vals[1] <= 0.0:
        return None
    a = k * np.sqrt(vals[0])
    b = k * np.sqrt(vals[1])
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return a, b, angle


@dataclass
class AimbotConfig:
    width: int = 800
    height: int = 600
    resize_factor: int = 4
    score_threshold: float = 0.40
    show_capture: bool = True
    input_mode: str = "keyboard"
    fire_key: str = "RETURN"
    shoot_target: str = "HUMAN"
    hold_duration: float = 0.4
    controller_connection: str = 'usb'
    loop_hz: Optional[int] = None
    controller_deadzone: Optional[float] = None
    controller_sensitivity: Optional[float] = None
    controller_smoothing: Optional[float] = None
    controller_polling_rate: Optional[float] = None
    kalman_q_process: Optional[float] = None
    kalman_r_measurement: Optional[float] = None
    kalman_p0_pos: Optional[float] = None
    kalman_p0_vel: Optional[float] = None
    kalman_max_dt: Optional[float] = None
    kalman_min_dt: Optional[float] = None
    kalman_drop_grace_frames: Optional[int] = None
    kalman_vel_hint_decay: Optional[float] = None
    perf_log_path: Optional[str] = None


DEFAULT_CONFIG = AimbotConfig()


def _safe_extract(tar: tarfile.TarFile, path: str) -> None:
    abs_path = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not member_path.startswith(abs_path):
            raise RuntimeError('Blocked path traversal attempt while extracting the model archive.')
    tar.extractall(path)


def _ensure_model_files() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(PATH_TO_CKPT):
        return

    tar_path = os.path.join(MODEL_DIR, MODEL_FILE)
    print(f'Downloading TensorFlow model: {MODEL_FILE}')
    try:
        urllib.request.urlretrieve(MODEL_URL, tar_path)
        with tarfile.open(tar_path) as tar:
            _safe_extract(tar, MODEL_DIR)
    except Exception as exc:
        raise RuntimeError(f'Failed to download or extract TensorFlow model ({MODEL_FILE}).') from exc
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)


class CSAimbot:
    def __init__(self, keys_driver: Optional[k.Keys] = None) -> None:
        self.keys = keys_driver or k.Keys({})
        self.detection_graph: Optional[tf1.Graph] = None
        self.session: Optional[tf1.Session] = None
        self._tensor_handles: Dict[str, tf.Tensor] = {}
        self.stop_event = threading.Event()
        self.last_error: Optional[Exception] = None
        self.predictor = KalmanPredictor()
        self.controller_reader = ControllerInputReader()
        self._ticker: Optional[FixedTicker] = None
        self._dt_target: float = 0.005
        self.perf_probe: Optional[PerfProbe] = None
        self._active_controller_profile: Dict[str, float] = {}
        self._active_targets: Dict[str, float] = {}

    def run(self, config: AimbotConfig) -> None:
        self.stop_event.clear()
        self.last_error = None
        try:
            self._validate_config(config)
            _ensure_model_files()
            self._ensure_session()
            self._prepare_runtime(config)
            self._event_loop(config)
        except Exception as exc:
            self.last_error = exc
            raise
        finally:
            if self.perf_probe is not None:
                latency = self.perf_probe.p95_cmd_latency_ms()
                jitter = self.perf_probe.jitter_ms()
                print(f'CSAimbot timing: p95 frame->cmd {latency:.2f}ms, jitter {jitter:.2f}ms (logged to {self.perf_probe.path})')
                self.perf_probe.close()
                self.perf_probe = None
            self.controller_reader.stop()
            destroy_all_windows_safe()

    def stop(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        self.controller_reader.stop()
        if self.perf_probe is not None:
            self.perf_probe.close()
            self.perf_probe = None
        if self.session is not None:
            self.session.close()
            self.session = None
        self.detection_graph = None
        self._tensor_handles.clear()
        destroy_all_windows_safe()

    def _validate_config(self, config: AimbotConfig) -> None:
        if config.width <= 0 or config.height <= 0:
            raise ValueError('Width and height must be positive values.')
        if config.resize_factor <= 0:
            raise ValueError('Resize factor must be a positive value.')
        if not 0 <= config.score_threshold <= 1:
            raise ValueError('Score threshold must be between 0 and 1.')
        if config.hold_duration < 0:
            raise ValueError('Hold duration must be non-negative.')

        config.input_mode = config.input_mode.lower()
        config.shoot_target = config.shoot_target.upper()
        config.fire_key = config.fire_key.upper()

        valid_targets = ('HUMAN', 'HUMAN_BODY', 'HEAD')
        if config.input_mode not in ('keyboard', 'mouse'):
            raise ValueError('Input mode must be either "keyboard" or "mouse".')
        if config.shoot_target not in valid_targets:
            raise ValueError(f'Shoot target must be one of {", ".join(valid_targets)}.')
        if config.input_mode == 'keyboard' and config.fire_key not in self.keys.dk:
            raise ValueError(f'Unsupported fire key "{config.fire_key}". See keys.py for valid options.')

        valid_connections = tuple(CONTROLLER_PRESETS.keys())
        if config.controller_connection not in valid_connections:
            raise ValueError(f'Controller connection must be one of {", ".join(valid_connections)}.')
        if config.loop_hz is not None and config.loop_hz <= 0:
            raise ValueError('Loop frequency must be positive when provided.')
        if config.controller_deadzone is not None and not 0.0 <= config.controller_deadzone < 1.0:
            raise ValueError('Controller deadzone must be between 0 (inclusive) and 1 (exclusive).')
        if config.controller_sensitivity is not None and config.controller_sensitivity <= 0:
            raise ValueError('Controller sensitivity must be a positive value.')
        if config.controller_smoothing is not None and not 0.0 <= config.controller_smoothing <= 1.0:
            raise ValueError('Controller smoothing must be between 0 and 1 when provided.')
        if config.controller_polling_rate is not None and config.controller_polling_rate <= 0:
            raise ValueError('Controller polling rate must be positive when provided.')

        for value, label in (
            (config.kalman_q_process, 'Kalman process noise'),
            (config.kalman_r_measurement, 'Kalman measurement variance'),
            (config.kalman_p0_pos, 'Kalman initial position variance'),
            (config.kalman_p0_vel, 'Kalman initial velocity variance'),
        ):
            if value is not None and value <= 0:
                raise ValueError(f'{label} must be positive when provided.')

        if config.kalman_min_dt is not None and config.kalman_min_dt <= 0:
            raise ValueError('Kalman min dt must be positive when provided.')
        if config.kalman_max_dt is not None and config.kalman_max_dt <= 0:
            raise ValueError('Kalman max dt must be positive when provided.')
        if (
            config.kalman_min_dt is not None
            and config.kalman_max_dt is not None
            and config.kalman_min_dt > config.kalman_max_dt
        ):
            raise ValueError('Kalman min dt cannot exceed max dt.')
        if config.kalman_drop_grace_frames is not None and config.kalman_drop_grace_frames < 0:
            raise ValueError('Kalman drop grace frames must be non-negative when provided.')
        if config.kalman_vel_hint_decay is not None and not 0.0 <= config.kalman_vel_hint_decay <= 1.0:
            raise ValueError('Kalman velocity hint decay must be between 0 and 1 when provided.')

    def _prepare_runtime(self, config: AimbotConfig) -> None:
        if config.show_capture and not has_highgui():
            reason = highgui_probe_reason()
            message = 'OpenCV GUI backend not available; disabling capture preview.'
            if reason:
                message += f' ({reason})'
            print(message + ' Install `opencv-contrib-python` for window display support.')
            config.show_capture = False

        profile = CONTROLLER_PRESETS[config.controller_connection]
        kalman_profile: Dict[str, float] = profile['kalman']  # type: ignore[assignment]
        controller_profile: Dict[str, float] = profile['controller']  # type: ignore[assignment]

        loop_hz = int(config.loop_hz or profile['loop_hz'])  # type: ignore[index]
        self._ticker = FixedTicker(loop_hz)
        self._dt_target = 1.0 / loop_hz

        params = KalmanParams(
            q_process=config.kalman_q_process or kalman_profile['q_process'],
            r_meas_px=config.kalman_r_measurement or kalman_profile['r_meas_px'],
            p0_pos=config.kalman_p0_pos or kalman_profile['p0_pos'],
            p0_vel=config.kalman_p0_vel or kalman_profile['p0_vel'],
            max_dt=config.kalman_max_dt or kalman_profile['max_dt'],
            min_dt=config.kalman_min_dt or kalman_profile['min_dt'],
            drop_grace_frames=config.kalman_drop_grace_frames or int(kalman_profile['drop_grace_frames']),
            vel_hint_decay=config.kalman_vel_hint_decay or kalman_profile['vel_hint_decay'],
        )
        self.predictor.configure(params)

        polling_rate = config.controller_polling_rate or controller_profile['poll_hz']
        deadzone = config.controller_deadzone if config.controller_deadzone is not None else controller_profile['deadzone']
        sensitivity = (
            config.controller_sensitivity if config.controller_sensitivity is not None else controller_profile['sensitivity']
        )
        smoothing = (
            config.controller_smoothing if config.controller_smoothing is not None else controller_profile['smoothing']
        )
        self.controller_reader.configure(
            polling_rate=polling_rate,
            deadzone=deadzone,
            sensitivity=sensitivity,
            smoothing=smoothing,
        )
        self.controller_reader.start()

        if self.perf_probe is not None:
            self.perf_probe.close()
        self.perf_probe = PerfProbe(config.perf_log_path)

        self._active_controller_profile = controller_profile
        self._active_targets = profile.get('targets', {})  # type: ignore[assignment]
        self._lock_runtime(loop_hz)

    def _lock_runtime(self, loop_hz: int) -> None:
        try:
            import psutil  # type: ignore[import-not-found]
            import sys

            process = psutil.Process()
            core_id = 0
            if hasattr(process, 'cpu_affinity'):
                current_affinity = process.cpu_affinity()  # type: ignore[attr-defined]
                if current_affinity != [core_id]:
                    process.cpu_affinity([core_id])  # type: ignore[attr-defined]
            if sys.platform.startswith('win'):
                process.nice(psutil.HIGH_PRIORITY_CLASS)  # type: ignore[attr-defined]
            else:
                try:
                    process.nice(-10)
                except Exception:
                    pass
        except Exception:
            # Best-effort; lack of psutil or permissions should not abort the run.
            pass

    def _ensure_session(self) -> None:
        if self.detection_graph is None:
            self.detection_graph = tf1.Graph()
            with self.detection_graph.as_default():
                graph_def = tf1.GraphDef()
                with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    graph_def.ParseFromString(fid.read())
                    tf1.import_graph_def(graph_def, name='')
        if self.session is None:
            self.session = tf1.Session(graph=self.detection_graph)
        if not self._tensor_handles:
            assert self.detection_graph is not None
            self._tensor_handles = {
                'image_tensor': self.detection_graph.get_tensor_by_name('image_tensor:0'),
                'detection_boxes': self.detection_graph.get_tensor_by_name('detection_boxes:0'),
                'detection_scores': self.detection_graph.get_tensor_by_name('detection_scores:0'),
                'detection_classes': self.detection_graph.get_tensor_by_name('detection_classes:0'),
                'num_detections': self.detection_graph.get_tensor_by_name('num_detections:0'),
            }

    def _event_loop(self, config: AimbotConfig) -> None:
        assert self.session is not None
        ticker = self._ticker or FixedTicker(200)
        frame_width = max(1, int(config.width / config.resize_factor))
        frame_height = max(1, int(config.height / config.resize_factor))
        self.predictor.reset()

        controller_profile = self._active_controller_profile or {}
        poll_hz = float(controller_profile.get('poll_hz', ticker.hz))  # type: ignore[arg-type]

        while not self.stop_event.is_set():
            frame_rgb = grab_screen(region=(0, 0, config.width, config.height))
            frame_rgb = cv2.resize(frame_rgb, (frame_width, frame_height))
            frame_ready_ns = time.perf_counter_ns()
            input_tensor = np.expand_dims(frame_rgb, axis=0)

            detect_start_ns = time.perf_counter_ns()
            boxes, scores, classes, num = self.session.run(
                (
                    self._tensor_handles['detection_boxes'],
                    self._tensor_handles['detection_scores'],
                    self._tensor_handles['detection_classes'],
                    self._tensor_handles['num_detections'],
                ),
                feed_dict={self._tensor_handles['image_tensor']: input_tensor},
            )
            detect_end_ns = time.perf_counter_ns()

            boxes = boxes[0]
            scores = scores[0]
            classes = classes[0].astype(np.int32)
            detection_count = int(num[0])

            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            measurement_px: Optional[np.ndarray] = None
            aim_x_norm = 0.5
            aim_y_norm = 0.5
            had_measurement = False

            for idx in range(detection_count):
                if scores[idx] < config.score_threshold:
                    continue
                if classes[idx] != PERSON_CLASS_ID:
                    continue

                ymin, xmin, ymax, xmax = boxes[idx]
                mid_x = (xmin + xmax) / 2.0
                mid_y = (ymin + ymax) / 2.0
                body_y = ymin + (ymax - ymin) * 0.65
                head_y = ymin + (ymax - ymin) * 0.2

                if config.shoot_target == 'HUMAN':
                    aim_y_norm = mid_y
                elif config.shoot_target == 'HUMAN_BODY':
                    aim_y_norm = body_y
                else:
                    aim_y_norm = head_y

                aim_x_norm = min(max(mid_x, 0.0), 1.0)
                aim_y_norm = min(max(aim_y_norm, 0.0), 1.0)

                if config.show_capture:
                    self._draw_box(display_frame, xmin, ymin, xmax, ymax, scores[idx])

                measurement_px = np.array(
                    [aim_x_norm * config.width, aim_y_norm * config.height],
                    dtype=float,
                )
                had_measurement = True
                break

            raw_hint = self.controller_reader.get_velocity() if self.controller_reader else None
            velocity_hint_px = None
            if raw_hint is not None:
                velocity_hint_px = np.array(
                    [
                        raw_hint[0] * config.width * poll_hz,
                        raw_hint[1] * config.height * poll_hz,
                    ],
                    dtype=float,
                )

            prediction_px = self.predictor.step(measurement_px, self._dt_target, velocity_hint=velocity_hint_px)
            predict_end_ns = time.perf_counter_ns()
            had_hint = velocity_hint_px is not None
            command_end_ns = predict_end_ns
            covariance = self.predictor.covariance()

            if prediction_px is not None:
                aim_x_norm = min(max(float(prediction_px[0] / config.width), 0.0), 1.0)
                aim_y_norm = min(max(float(prediction_px[1] / config.height), 0.0), 1.0)

                if config.show_capture:
                    self._draw_prediction(display_frame, aim_x_norm, aim_y_norm)
                    if covariance is not None:
                        self._draw_uncertainty(display_frame, covariance, aim_x_norm, aim_y_norm, config)

                self._move_crosshair(aim_x_norm, aim_y_norm, config)
                if had_measurement and not self.stop_event.is_set():
                    self._fire(config)
                command_end_ns = time.perf_counter_ns()

            latency_ms = None
            jitter_ms = None
            if self.perf_probe is not None:
                self.perf_probe.log(
                    frame_ready_ns,
                    detect_end_ns,
                    predict_end_ns,
                    command_end_ns,
                    had_measurement,
                    had_hint,
                )
                latency_ms = self.perf_probe.p95_cmd_latency_ms()
                jitter_ms = self.perf_probe.jitter_ms()

            if config.show_capture:
                if latency_ms is not None:
                    self._draw_lag_meter(display_frame, latency_ms, jitter_ms)
                imshow_safe(WINDOW_NAME, display_frame)

            key = waitkey_safe(1)
            if key & 0xFF == ord('q'):
                self.stop()

            if self.stop_event.is_set():
                break

            ticker.sleep_until_next()

    def _draw_box(self, frame: np.ndarray, xmin: float, ymin: float, xmax: float, ymax: float, score: float) -> None:
        height, width, _ = frame.shape
        left = max(0, int(xmin * width))
        top = max(0, int(ymin * height))
        right = min(width - 1, int(xmax * width))
        bottom = min(height - 1, int(ymax * height))
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f'Person {score * 100:.1f}%'
        label_y = top - 10 if top > 15 else top + 20
        cv2.putText(
            frame,
            label,
            (left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    def _draw_prediction(self, frame: np.ndarray, x: float, y: float) -> None:
        height, width, _ = frame.shape
        center_x = int(min(max(x, 0.0), 1.0) * width)
        center_y = int(min(max(y, 0.0), 1.0) * height)
        cv2.circle(frame, (center_x, center_y), 4, (0, 165, 255), -1)

    def _draw_uncertainty(
        self,
        frame: np.ndarray,
        covariance: np.ndarray,
        aim_x_norm: float,
        aim_y_norm: float,
        config: AimbotConfig,
    ) -> None:
        cov_pos = covariance[:2, :2]
        scale_matrix = np.diag([1.0 / max(config.width, 1), 1.0 / max(config.height, 1)])
        cov_norm = scale_matrix @ cov_pos @ scale_matrix
        ellipse = ellipse_from_cov(cov_norm, k=2.0)
        if ellipse is None:
            return
        a_norm, b_norm, angle = ellipse
        height, width, _ = frame.shape
        center = (int(aim_x_norm * width), int(aim_y_norm * height))
        axes = (
            max(1, int(a_norm * width)),
            max(1, int(b_norm * height)),
        )
        cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 215, 0), 1, cv2.LINE_AA)

    def _draw_lag_meter(self, frame: np.ndarray, latency_ms: float, jitter_ms: Optional[float]) -> None:
        targets = self._active_targets or {}
        latency_target = float(targets.get('latency_p95_ms', 10.0))
        jitter_target = float(targets.get('jitter_ms', 4.0))
        jitter_val = jitter_ms if jitter_ms is not None else 0.0
        color = (0, 255, 0)
        if latency_ms > latency_target or jitter_val > jitter_target:
            color = (0, 0, 255)
        text = f'p95 {latency_ms:.1f}ms | jitter {jitter_val:.1f}ms'
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        threshold_text = f'target {latency_target:.1f}/{jitter_target:.1f}ms'
        cv2.putText(frame, threshold_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def _move_crosshair(self, mid_x: float, mid_y: float, config: AimbotConfig) -> None:
        dx = -int((0.5 - mid_x) * config.width)
        dy = -int((0.5 - mid_y) * config.height)
        if dx != 0 or dy != 0:
            self.keys.directMouse(dx, dy)

    def _fire(self, config: AimbotConfig) -> None:
        if config.input_mode == 'keyboard':
            self.keys.directKey(config.fire_key)
            self._responsive_sleep(config.hold_duration)
            self.keys.directKey(config.fire_key, self.keys.key_release)
        else:
            self.keys.directMouse(0, 0, self.keys.mouse_lb_press)
            self._responsive_sleep(config.hold_duration)
            self.keys.directMouse(0, 0, self.keys.mouse_lb_release)

    def _responsive_sleep(self, duration: float) -> None:
        end_time = time.time() + duration
        while time.time() < end_time:
            if self.stop_event.is_set():
                break
            time.sleep(0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CSAimbot controller')
    parser.add_argument('--width', type=int, default=DEFAULT_CONFIG.width, help='Capture width (default: 800).')
    parser.add_argument('--height', type=int, default=DEFAULT_CONFIG.height, help='Capture height (default: 600).')
    parser.add_argument(
        '--resize',
        dest='resize',
        type=int,
        default=DEFAULT_CONFIG.resize_factor,
        help='Downscale factor applied before detection (default: 4).',
    )
    parser.add_argument(
        '--score',
        dest='score',
        type=float,
        default=DEFAULT_CONFIG.score_threshold,
        help='Minimum detection confidence required (default: 0.40).',
    )
    parser.add_argument(
        '--show',
        dest='show',
        action='store_true',
        help='Display a preview window (default).',
    )
    parser.add_argument(
        '--no-show',
        dest='show',
        action='store_false',
        help='Disable the preview window.',
    )
    parser.add_argument(
        '--input',
        dest='input_mode',
        choices=('keyboard', 'mouse'),
        default=DEFAULT_CONFIG.input_mode,
        help='Fire using keyboard or mouse (default: keyboard).',
    )
    parser.add_argument(
        '--key',
        type=str,
        default=DEFAULT_CONFIG.fire_key,
        help='Keyboard key to press when firing (keyboard mode only).',
    )
    parser.add_argument(
        '--shoot',
        choices=('HUMAN', 'HUMAN_BODY', 'HEAD'),
        default=DEFAULT_CONFIG.shoot_target,
        help='Aim at HUMAN center mass, HUMAN_BODY (torso), or HEAD (default: HUMAN).',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=DEFAULT_CONFIG.hold_duration,
        help='How long to hold the fire input in seconds (default: 0.4).',
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch the graphical interface for configuration.',
    )
    parser.add_argument(
        '--connection',
        '--profile',
        dest='controller_connection',
        choices=tuple(CONTROLLER_PRESETS.keys()),
        default=DEFAULT_CONFIG.controller_connection,
        help='Controller connection profile for Kalman/controller tuning (default: usb).',
    )
    parser.add_argument(
        '--loop-hz',
        type=int,
        default=DEFAULT_CONFIG.loop_hz,
        help='Override predictor loop frequency in Hz (default depends on profile).',
    )
    parser.add_argument(
        '--controller-deadzone',
        type=float,
        default=DEFAULT_CONFIG.controller_deadzone,
        help='Deadzone applied to controller stick input (blank → profile default).',
    )
    parser.add_argument(
        '--controller-sensitivity',
        type=float,
        default=DEFAULT_CONFIG.controller_sensitivity,
        help='Scaling applied to controller stick velocity hints (blank → profile default).',
    )
    parser.add_argument(
        '--controller-smoothing',
        type=float,
        default=DEFAULT_CONFIG.controller_smoothing,
        help='Low-pass smoothing applied to controller hints (blank → profile default).',
    )
    parser.add_argument(
        '--controller-polling',
        dest='controller_polling_rate',
        type=float,
        default=None,
        help='Override controller polling rate in Hz.',
    )
    parser.add_argument(
        '--q-process',
        '--process-noise',
        dest='kalman_q_process',
        type=float,
        default=None,
        help='Override Kalman process acceleration noise q (px^2/s^4).',
    )
    parser.add_argument(
        '--r-meas-px',
        '--measurement-noise',
        dest='kalman_r_measurement',
        type=float,
        default=None,
        help='Override Kalman measurement variance r (px^2).',
    )
    parser.add_argument(
        '--p0-pos',
        dest='kalman_p0_pos',
        type=float,
        default=None,
        help='Override Kalman initial position variance (px^2).',
    )
    parser.add_argument(
        '--p0-vel',
        dest='kalman_p0_vel',
        type=float,
        default=None,
        help='Override Kalman initial velocity variance ((px/s)^2).',
    )
    parser.add_argument(
        '--drop-grace-frames',
        dest='kalman_drop_grace_frames',
        type=int,
        default=None,
        help='Override how many frames to coast on prediction before inflating covariance.',
    )
    parser.add_argument(
        '--vel-hint-decay',
        dest='kalman_vel_hint_decay',
        type=float,
        default=None,
        help='Override blend factor for velocity hints (0=trust hint, 1=ignore).',
    )
    parser.add_argument(
        '--max-prediction-dt',
        dest='kalman_max_dt',
        type=float,
        default=None,
        help='Override Kalman maximum delta time in seconds (default depends on connection).',
    )
    parser.add_argument(
        '--min-prediction-dt',
        dest='kalman_min_dt',
        type=float,
        default=None,
        help='Override Kalman minimum delta time in seconds.',
    )
    parser.add_argument(
        '--perf-log',
        dest='perf_log_path',
        type=str,
        default=None,
        help='Path to write PerfProbe CSV (default: auto timestamp in current directory).',
    )
    parser.set_defaults(show=DEFAULT_CONFIG.show_capture)
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> AimbotConfig:
    return AimbotConfig(
        width=args.width,
        height=args.height,
        resize_factor=args.resize,
        score_threshold=args.score,
        show_capture=args.show,
        input_mode=args.input_mode,
        fire_key=args.key,
        shoot_target=args.shoot,
        hold_duration=args.duration,
        controller_connection=args.controller_connection,
        loop_hz=args.loop_hz,
        controller_deadzone=args.controller_deadzone,
        controller_sensitivity=args.controller_sensitivity,
        controller_smoothing=args.controller_smoothing,
        controller_polling_rate=args.controller_polling_rate,
        kalman_q_process=args.kalman_q_process,
        kalman_r_measurement=args.kalman_r_measurement,
        kalman_p0_pos=args.kalman_p0_pos,
        kalman_p0_vel=args.kalman_p0_vel,
        kalman_max_dt=args.kalman_max_dt,
        kalman_min_dt=args.kalman_min_dt,
        kalman_drop_grace_frames=args.kalman_drop_grace_frames,
        kalman_vel_hint_decay=args.kalman_vel_hint_decay,
        perf_log_path=args.perf_log_path,
    )


def launch_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except ImportError as exc:
        raise SystemExit('tkinter is required for GUI mode but could not be imported.') from exc

    class AimbotGUI(tk.Tk):
        def __init__(self, runner: CSAimbot) -> None:
            super().__init__()
            self.runner = runner
            self.worker_thread: Optional[threading.Thread] = None
            self.title('CSAimbot Control Panel')
            self.protocol('WM_DELETE_WINDOW', self.on_close)
            self.resizable(False, False)

            self.status_var = tk.StringVar(value='Idle')
            self.width_var = tk.StringVar(value=str(DEFAULT_CONFIG.width))
            self.height_var = tk.StringVar(value=str(DEFAULT_CONFIG.height))
            self.resize_var = tk.StringVar(value=str(DEFAULT_CONFIG.resize_factor))
            self.score_var = tk.StringVar(value=str(DEFAULT_CONFIG.score_threshold))
            self.show_var = tk.BooleanVar(value=DEFAULT_CONFIG.show_capture)
            self.input_mode_var = tk.StringVar(value=DEFAULT_CONFIG.input_mode)
            self.key_var = tk.StringVar(value=DEFAULT_CONFIG.fire_key)
            self.shoot_var = tk.StringVar(value=DEFAULT_CONFIG.shoot_target)
            self.duration_var = tk.StringVar(value=str(DEFAULT_CONFIG.hold_duration))
            self.connection_var = tk.StringVar(value=DEFAULT_CONFIG.controller_connection)
            self.loop_hz_var = tk.StringVar(value='' if DEFAULT_CONFIG.loop_hz is None else str(DEFAULT_CONFIG.loop_hz))
            self.deadzone_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.controller_deadzone is None else str(DEFAULT_CONFIG.controller_deadzone)
            )
            self.sensitivity_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.controller_sensitivity is None else str(DEFAULT_CONFIG.controller_sensitivity)
            )
            self.smoothing_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.controller_smoothing is None else str(DEFAULT_CONFIG.controller_smoothing)
            )
            self.polling_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.controller_polling_rate is None else str(DEFAULT_CONFIG.controller_polling_rate)
            )
            self.q_process_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_q_process is None else str(DEFAULT_CONFIG.kalman_q_process)
            )
            self.r_meas_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_r_measurement is None else str(DEFAULT_CONFIG.kalman_r_measurement)
            )
            self.p0_pos_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_p0_pos is None else str(DEFAULT_CONFIG.kalman_p0_pos)
            )
            self.p0_vel_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_p0_vel is None else str(DEFAULT_CONFIG.kalman_p0_vel)
            )
            self.drop_grace_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_drop_grace_frames is None else str(DEFAULT_CONFIG.kalman_drop_grace_frames)
            )
            self.vel_hint_decay_var = tk.StringVar(
                value=''
                if DEFAULT_CONFIG.kalman_vel_hint_decay is None
                else str(DEFAULT_CONFIG.kalman_vel_hint_decay)
            )
            self.min_dt_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_min_dt is None else str(DEFAULT_CONFIG.kalman_min_dt)
            )
            self.max_dt_var = tk.StringVar(
                value='' if DEFAULT_CONFIG.kalman_max_dt is None else str(DEFAULT_CONFIG.kalman_max_dt)
            )

            self._build_layout()

        def _build_layout(self) -> None:
            padding = {'padx': 10, 'pady': 5}

            ttk.Label(self, text='Capture width').grid(row=0, column=0, sticky='w', **padding)
            ttk.Entry(self, textvariable=self.width_var, width=12).grid(row=0, column=1, sticky='ew', **padding)

            ttk.Label(self, text='Capture height').grid(row=1, column=0, sticky='w', **padding)
            ttk.Entry(self, textvariable=self.height_var, width=12).grid(row=1, column=1, sticky='ew', **padding)

            ttk.Label(self, text='Resize factor').grid(row=2, column=0, sticky='w', **padding)
            ttk.Entry(self, textvariable=self.resize_var, width=12).grid(row=2, column=1, sticky='ew', **padding)

            ttk.Label(self, text='Score threshold').grid(row=3, column=0, sticky='w', **padding)
            ttk.Entry(self, textvariable=self.score_var, width=12).grid(row=3, column=1, sticky='ew', **padding)

            ttk.Checkbutton(self, text='Show preview window', variable=self.show_var).grid(
                row=4, column=0, columnspan=2, sticky='w', **padding
            )

            ttk.Label(self, text='Input mode').grid(row=5, column=0, sticky='w', **padding)
            self.input_mode_combo = ttk.Combobox(
                self,
                textvariable=self.input_mode_var,
                values=('keyboard', 'mouse'),
                state='readonly',
                width=10,
            )
            self.input_mode_combo.grid(row=5, column=1, sticky='ew', **padding)
            self.input_mode_combo.bind('<<ComboboxSelected>>', self._on_input_mode_changed)

            ttk.Label(self, text='Fire key').grid(row=6, column=0, sticky='w', **padding)
            self.key_entry = ttk.Entry(self, textvariable=self.key_var, width=12)
            self.key_entry.grid(row=6, column=1, sticky='ew', **padding)

            ttk.Label(self, text='Aim for').grid(row=7, column=0, sticky='w', **padding)
            self.shoot_combo = ttk.Combobox(
                self,
                textvariable=self.shoot_var,
                values=('HUMAN', 'HUMAN_BODY', 'HEAD'),
                state='readonly',
                width=10,
            )
            self.shoot_combo.grid(row=7, column=1, sticky='ew', **padding)

            ttk.Label(self, text='Hold duration (s)').grid(row=8, column=0, sticky='w', **padding)
            ttk.Entry(self, textvariable=self.duration_var, width=12).grid(row=8, column=1, sticky='ew', **padding)

            controller_frame = ttk.LabelFrame(self, text='Controller & Prediction')
            controller_frame.grid(row=9, column=0, columnspan=2, sticky='ew', padx=10, pady=(5, 5))
            controller_frame.columnconfigure(1, weight=1)
            subpad = {'padx': 8, 'pady': 3}

            ttk.Label(controller_frame, text='Connection type').grid(row=0, column=0, sticky='w', **subpad)
            self.connection_combo = ttk.Combobox(
                controller_frame,
                textvariable=self.connection_var,
                values=tuple(CONTROLLER_PRESETS.keys()),
                state='readonly',
                width=12,
            )
            self.connection_combo.grid(row=0, column=1, sticky='ew', **subpad)

            ttk.Label(controller_frame, text='Loop Hz (blank auto)').grid(row=1, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.loop_hz_var, width=14).grid(
                row=1, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Deadzone (blank auto)').grid(row=2, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.deadzone_var, width=14).grid(
                row=2, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Sensitivity (blank auto)').grid(row=3, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.sensitivity_var, width=14).grid(
                row=3, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Smoothing (blank auto)').grid(row=4, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.smoothing_var, width=14).grid(
                row=4, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Polling Hz (blank auto)').grid(row=5, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.polling_var, width=14).grid(
                row=5, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='q process (blank auto)').grid(row=6, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.q_process_var, width=14).grid(
                row=6, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='r meas px (blank auto)').grid(row=7, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.r_meas_var, width=14).grid(
                row=7, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='P0 pos (blank auto)').grid(row=8, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.p0_pos_var, width=14).grid(
                row=8, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='P0 vel (blank auto)').grid(row=9, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.p0_vel_var, width=14).grid(
                row=9, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Grace frames (blank auto)').grid(row=10, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.drop_grace_var, width=14).grid(
                row=10, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Vel hint decay (blank auto)').grid(row=11, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.vel_hint_decay_var, width=14).grid(
                row=11, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Min dt (blank auto)').grid(row=12, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.min_dt_var, width=14).grid(
                row=12, column=1, sticky='ew', **subpad
            )

            ttk.Label(controller_frame, text='Max dt (blank auto)').grid(row=13, column=0, sticky='w', **subpad)
            ttk.Entry(controller_frame, textvariable=self.max_dt_var, width=14).grid(
                row=13, column=1, sticky='ew', **subpad
            )

            button_frame = ttk.Frame(self)
            button_frame.grid(row=10, column=0, columnspan=2, pady=(10, 5))
            self.start_button = ttk.Button(button_frame, text='Start', command=self.start_bot)
            self.start_button.grid(row=0, column=0, padx=5)
            self.stop_button = ttk.Button(button_frame, text='Stop', command=self.stop_bot, state=tk.DISABLED)
            self.stop_button.grid(row=0, column=1, padx=5)

            ttk.Label(self, textvariable=self.status_var).grid(row=11, column=0, columnspan=2, pady=(5, 10))

            self.columnconfigure(1, weight=1)
            self._on_input_mode_changed()

        def start_bot(self) -> None:
            if self.worker_thread and self.worker_thread.is_alive():
                return
            try:
                config = self._collect_config()
            except ValueError as exc:
                messagebox.showerror('Invalid configuration', str(exc), parent=self)
                return

            self.runner.stop()
            self.worker_thread = threading.Thread(target=self._run_bot, args=(config,), daemon=True)
            self.worker_thread.start()
            self.status_var.set('Running')
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.after(200, self._poll_worker)

        def _run_bot(self, config: AimbotConfig) -> None:
            try:
                self.runner.run(config)
            except Exception as exc:  # pragma: no cover - errors are surfaced via last_error
                self.runner.last_error = exc

        def _poll_worker(self) -> None:
            if self.worker_thread and self.worker_thread.is_alive():
                self.after(200, self._poll_worker)
                return

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            if self.runner.last_error:
                messagebox.showerror('CSAimbot error', str(self.runner.last_error), parent=self)
                self.runner.last_error = None
                self.status_var.set('Error')
            elif self.status_var.get().startswith('Stopping'):
                self.status_var.set('Stopped')
            else:
                self.status_var.set('Idle')

        def stop_bot(self) -> None:
            if self.worker_thread and self.worker_thread.is_alive():
                self.status_var.set('Stopping...')
                self.runner.stop()

        def on_close(self) -> None:
            self.stop_bot()
            if self.worker_thread and self.worker_thread.is_alive():
                self.after(200, self.on_close)
                return
            self.runner.close()
            self.destroy()

        def _on_input_mode_changed(self, _event: Optional[object] = None) -> None:
            mode = self.input_mode_var.get()
            if mode == 'keyboard':
                self.key_entry.config(state='normal')
            else:
                self.key_entry.config(state='disabled')

        def _collect_config(self) -> AimbotConfig:
            def _float_or_none(value: str) -> Optional[float]:
                text = value.strip()
                if not text:
                    return None
                return float(text)

            def _int_or_none(value: str) -> Optional[int]:
                text = value.strip()
                if not text:
                    return None
                return int(float(text))

            try:
                width = int(self.width_var.get())
                height = int(self.height_var.get())
                resize = int(self.resize_var.get())
                score = float(self.score_var.get())
                duration = float(self.duration_var.get())
                loop_hz = _int_or_none(self.loop_hz_var.get())
                deadzone = _float_or_none(self.deadzone_var.get())
                sensitivity = _float_or_none(self.sensitivity_var.get())
                smoothing = _float_or_none(self.smoothing_var.get())
                polling = _float_or_none(self.polling_var.get())
                q_process = _float_or_none(self.q_process_var.get())
                r_meas = _float_or_none(self.r_meas_var.get())
                p0_pos = _float_or_none(self.p0_pos_var.get())
                p0_vel = _float_or_none(self.p0_vel_var.get())
                drop_grace = _int_or_none(self.drop_grace_var.get())
                vel_hint_decay = _float_or_none(self.vel_hint_decay_var.get())
                min_dt = _float_or_none(self.min_dt_var.get())
                max_dt = _float_or_none(self.max_dt_var.get())
            except ValueError as exc:
                raise ValueError(
                    'Numeric fields must contain valid numbers. Leave fields blank to use profile defaults.'
                ) from exc

            config = AimbotConfig(
                width=width,
                height=height,
                resize_factor=resize,
                score_threshold=score,
                show_capture=self.show_var.get(),
                input_mode=self.input_mode_var.get(),
                fire_key=self.key_var.get(),
                shoot_target=self.shoot_var.get(),
                hold_duration=duration,
                controller_connection=self.connection_var.get(),
                loop_hz=loop_hz,
                controller_deadzone=deadzone,
                controller_sensitivity=sensitivity,
                controller_smoothing=smoothing,
                controller_polling_rate=polling,
                kalman_q_process=q_process,
                kalman_r_measurement=r_meas,
                kalman_p0_pos=p0_pos,
                kalman_p0_vel=p0_vel,
                kalman_drop_grace_frames=drop_grace,
                kalman_vel_hint_decay=vel_hint_decay,
                kalman_min_dt=min_dt,
                kalman_max_dt=max_dt,
            )
            self.runner._validate_config(config)
            return config

    gui = AimbotGUI(CSAimbot())
    gui.mainloop()


def main() -> None:
    args = parse_args()
    if getattr(args, 'gui', False):
        launch_gui()
        return

    config = build_config_from_args(args)
    runner = CSAimbot()

    try:
        runner.run(config)
    except KeyboardInterrupt:
        runner.stop()
    except Exception as exc:
        runner.stop()
        runner.close()
        raise SystemExit(f'CSAimbot failed: {exc}') from exc
    finally:
        runner.close()


if __name__ == '__main__':
    main()
