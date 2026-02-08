#!/usr/bin/env python3
"""
Realtime hand tracking -> Franka arm + gripper retargeting.

Uses MediaPipe for hand detection and tracks wrist position to control the Franka arm.
Gripper width is controlled by thumb-index finger distance.

Usage:
    python show_franka_retargeting.py --arm-id left --use-realsense
    python show_franka_retargeting.py --arm-id right --camera-path /dev/video0

Controls:
    Space: Toggle motion enable/disable
    r: Reset reference position
    t: Toggle rotation tracking on/off
    q: Quit
"""

import argparse
import importlib.util
import os
import sys
import time
import signal
import threading
import types
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

from single_hand_detector import SingleHandDetector
from tactile_sensor import TactileSensor, FingerForceController


# MANO keypoint indices
MANO_WRIST = 0
MANO_THUMB_TIP = 4
MANO_INDEX_TIP = 8


def _add_path(path: str) -> None:
    """Add path to sys.path if not already present."""
    if path and path not in sys.path:
        sys.path.insert(0, path)


def _select_realsense_serial(prefer_model: str = "D435") -> Optional[str]:
    """Select a RealSense camera serial number, preferring the specified model."""
    if not HAS_REALSENSE:
        return None
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
    except Exception:
        return None

    if len(devices) == 0:
        return None

    prefer_model = (prefer_model or "").lower()
    for dev in devices:
        try:
            name = dev.get_info(rs.camera_info.name).lower()
            if prefer_model and prefer_model in name:
                serial = dev.get_info(rs.camera_info.serial_number)
                print(f"Selected RealSense: {name} (serial: {serial})")
                return serial
        except Exception:
            continue

    # Fallback to first device
    try:
        serial = devices[0].get_info(rs.camera_info.serial_number)
        name = devices[0].get_info(rs.camera_info.name)
        print(f"Fallback to first RealSense: {name} (serial: {serial})")
        return serial
    except Exception:
        return None


def _parse_axis_map(axis_map_str: str) -> np.ndarray:
    """Parse axis map string like '1,1,1' or '-1,1,1' to numpy array."""
    parts = [p.strip() for p in axis_map_str.split(",")]
    if len(parts) != 3:
        raise ValueError("axis_map must have 3 comma-separated values, e.g. '1,1,1'")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _parse_coord_order(coord_order: str) -> tuple:
    """Parse coordinate order string like 'xyz', 'zxy', etc.

    Returns tuple of (indices, signs) where:
    - indices: which camera axis maps to each robot axis
    - signs: sign flips for each mapping

    Examples:
        'xyz' -> ([0,1,2], [1,1,1])  # cam_x->robot_x, cam_y->robot_y, cam_z->robot_z
        'zxy' -> ([2,0,1], [1,1,1])  # cam_z->robot_x, cam_x->robot_y, cam_y->robot_z
        '-x,y,z' -> ([0,1,2], [-1,1,1])  # cam_x->robot_x (flipped), etc.
    """
    coord_order = coord_order.lower().strip()

    # Check if using negative signs
    parts = coord_order.split(',')
    if len(parts) == 3:
        # Format: '-x,y,z'
        indices = []
        signs = []
        for part in parts:
            part = part.strip()
            sign = -1 if part.startswith('-') else 1
            axis = part.lstrip('-+')
            if axis == 'x':
                indices.append(0)
            elif axis == 'y':
                indices.append(1)
            elif axis == 'z':
                indices.append(2)
            else:
                raise ValueError(f"Invalid axis: {axis}")
            signs.append(sign)
        return indices, signs
    else:
        # Format: 'xyz', 'zxy', etc.
        if len(coord_order) != 3:
            raise ValueError("coord_order must be 3 characters like 'xyz' or 'zxy'")

        indices = []
        signs = []
        for char in coord_order:
            sign = 1
            if char == '-':
                continue
            if char in ['+', '-']:
                sign = -1 if char == '-' else 1
                continue
            if char == 'x':
                indices.append(0)
            elif char == 'y':
                indices.append(1)
            elif char == 'z':
                indices.append(2)
            else:
                raise ValueError(f"Invalid axis: {char}, must be x, y, or z")
            signs.append(sign)

        if len(indices) != 3:
            raise ValueError(f"coord_order must specify exactly 3 axes, got: {coord_order}")

        return indices, signs


def _apply_coord_transform(camera_coords: np.ndarray, coord_indices: tuple, coord_signs: tuple) -> np.ndarray:
    """Apply coordinate transformation based on coord_order.

    Args:
        camera_coords: [x, y, z] in camera frame
        coord_indices: which camera axis maps to each robot axis
        coord_signs: sign flips for each mapping

    Returns:
        [rx, ry, rz] in robot frame
    """
    indices, signs = coord_indices, coord_signs
    robot_coords = np.array([
        signs[0] * camera_coords[indices[0]],
        signs[1] * camera_coords[indices[1]],
        signs[2] * camera_coords[indices[2]]
    ], dtype=np.float32)
    return robot_coords


def _is_valid_keypoints(joint_pos: Optional[np.ndarray]) -> bool:
    """Check if keypoints are valid."""
    if joint_pos is None:
        return False
    if joint_pos.shape != (21, 3):
        return False
    if not np.isfinite(joint_pos).all():
        return False
    if np.allclose(joint_pos, 0.0):
        return False
    return True


def _compute_gripper_width_norm(joint_pos: np.ndarray, scale: float, offset: float) -> float:
    """
    Compute normalized gripper width from thumb-index finger distance.

    Args:
        joint_pos: 21x3 array of hand keypoints
        scale: Scale factor for distance to gripper width mapping
        offset: Offset for gripper width

    Returns:
        Normalized gripper width in [0, 1], where 1 = open, 0 = closed
    """
    thumb_tip = joint_pos[MANO_THUMB_TIP]
    index_tip = joint_pos[MANO_INDEX_TIP]
    distance = float(np.linalg.norm(thumb_tip - index_tip))
    return float(np.clip(scale * distance + offset, 0.0, 1.0))


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to axis-angle representation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ]) / (2 * np.sin(angle))
    return angle * axis


def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to 3x3 rotation matrix using Rodrigues formula."""
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-6:
        return np.eye(3)
    axis = axis_angle / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _interpolate_rotation(R1: np.ndarray, R2: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between two rotation matrices using axis-angle."""
    # Compute relative rotation: R_rel = R2 @ R1.T
    R_rel = R2 @ R1.T
    # Convert to axis-angle and scale
    aa = _rotation_matrix_to_axis_angle(R_rel)
    aa_scaled = aa * alpha
    # Apply scaled rotation to R1
    return _axis_angle_to_rotation_matrix(aa_scaled) @ R1


def _hand_rot_to_robot_rot(hand_rot: np.ndarray, base_rot: np.ndarray) -> np.ndarray:
    """
    Convert hand rotation from camera frame to robot end-effector rotation.

    The hand rotation from MediaPipe is in camera frame. We need to map it
    to the robot's end-effector frame while maintaining intuitive control.

    Args:
        hand_rot: 3x3 rotation matrix from MediaPipe (camera frame)
        base_rot: 3x3 base/initial rotation of the robot end-effector

    Returns:
        3x3 rotation matrix for robot end-effector
    """
    # Transform from camera frame to robot frame
    # Camera: X-right, Y-down, Z-forward (into scene)
    # Robot: X-forward, Y-left, Z-up (typically)
    # This mapping may need adjustment based on camera mounting
    camera_to_robot = np.array([
        [0, 0, 1],   # Robot X <- Camera Z
        [-1, 0, 0],  # Robot Y <- -Camera X
        [0, -1, 0],  # Robot Z <- -Camera Y
    ], dtype=np.float32)

    # Transform hand rotation to robot frame
    hand_rot_robot = camera_to_robot @ hand_rot @ camera_to_robot.T

    return hand_rot_robot


class RateLimiter:
    """Simple rate limiter for control loops."""
    def __init__(self, frequency: float, warn: bool = False):
        self.dt = 1.0 / float(frequency)
        self.next_time = time.monotonic() + self.dt
        self.warn = warn

    def sleep(self):
        now = time.monotonic()
        if self.next_time > now:
            time.sleep(self.next_time - now)
        elif self.warn:
            print("[RateLimiter] Missed tick")
        self.next_time = max(self.next_time + self.dt, time.monotonic() + self.dt)


class Deadman:
    """Keyboard-based deadman switch for safe teleoperation."""
    def __init__(self):
        self.motion_enabled = False
        self.reset_requested = False
        self.sensor_reset_requested = False
        self.quit_requested = False
        self.rotation_toggle_requested = False

        if HAS_PYNPUT:
            self._listener = keyboard.Listener(on_press=self._on_press)
            self._listener.start()
        else:
            print("WARNING: pynput not available, keyboard controls disabled")
            print("         Install with: pip install pynput")

    def _on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.motion_enabled = not self.motion_enabled
                state = "ENABLED" if self.motion_enabled else "PAUSED"
                print(f"[deadman] Motion {state}")
            elif hasattr(key, 'char') and key.char == "r":
                self.reset_requested = True
                print("[deadman] Reset reference requested")
            elif hasattr(key, 'char') and key.char == "c":
                self.sensor_reset_requested = True
                print("[deadman] Sensor reset requested")
            elif hasattr(key, 'char') and key.char == "t":
                self.rotation_toggle_requested = True
                print("[deadman] Rotation tracking toggle requested")
            elif hasattr(key, 'char') and key.char == "q":
                self.quit_requested = True
                print("[deadman] Quit requested")
        except Exception:
            pass

    def consume_reset(self) -> bool:
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False

    def consume_sensor_reset(self) -> bool:
        if self.sensor_reset_requested:
            self.sensor_reset_requested = False
            return True
        return False

    def consume_rotation_toggle(self) -> bool:
        if self.rotation_toggle_requested:
            self.rotation_toggle_requested = False
            return True
        return False


class HandTrackerThread(threading.Thread):
    """Background thread for hand detection with optional RGB-D support."""
    def __init__(self, capture, detector: SingleHandDetector,
                 img_width: int = 640, img_height: int = 480,
                 use_depth: bool = False, align: "rs.align" = None,
                 depth_scale: float = 0.001):
        super().__init__(daemon=True)
        self.capture = capture
        self.detector = detector
        self.img_width = img_width
        self.img_height = img_height
        self.use_depth = use_depth
        self.align = align  # RealSense alignment object
        self.depth_scale = depth_scale  # Convert depth units to meters
        self.latest_result = None
        self.latest_img = None
        self.latest_depth_img = None  # For visualization
        self.latest_keypoint_2d = None
        self.latest_wrist_3d = None  # 3D wrist position (x_px, y_px, z_meters)
        self.latest_ts = None
        self.fps = 0.0
        self._fps_count = 0
        self._fps_t0 = time.monotonic()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def _get_depth_at_point(self, depth_frame, x: int, y: int, window_size: int = 5) -> float:
        """Get depth value at a point, averaging over a small window for robustness."""
        if depth_frame is None:
            return 0.0

        h, w = depth_frame.shape[:2] if hasattr(depth_frame, 'shape') else (
            depth_frame.get_height(), depth_frame.get_width()
        )

        # Clamp coordinates
        x = max(window_size, min(w - window_size - 1, int(x)))
        y = max(window_size, min(h - window_size - 1, int(y)))

        # Get depth values in window
        depths = []
        for dy in range(-window_size, window_size + 1):
            for dx in range(-window_size, window_size + 1):
                if hasattr(depth_frame, 'get_distance'):
                    d = depth_frame.get_distance(x + dx, y + dy)
                else:
                    d = float(depth_frame[y + dy, x + dx]) * self.depth_scale
                if d > 0.1 and d < 2.0:  # Valid depth range (10cm to 2m)
                    depths.append(d)

        if len(depths) > 0:
            return float(np.median(depths))
        return 0.0

    def run(self):
        while not self.stop_event.is_set():
            try:
                depth_frame = None
                depth_image = None

                # Read frame
                if isinstance(self.capture, rs.pipeline):
                    frames = self.capture.wait_for_frames()

                    # Align depth to color if using depth
                    if self.use_depth and self.align is not None:
                        frames = self.align.process(frames)

                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    img_bgr = np.asanyarray(color_frame.get_data())

                    # Get depth frame if enabled
                    if self.use_depth:
                        depth_frame = frames.get_depth_frame()
                        if depth_frame:
                            depth_image = np.asanyarray(depth_frame.get_data())
                else:
                    ret, img_bgr = self.capture.read()
                    if not ret:
                        continue

                h, w = img_bgr.shape[:2]
                self.img_width = w
                self.img_height = h

                # Convert to RGB for MediaPipe
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Detect hand
                num_box, joint_pos, keypoint_2d, wrist_rot = self.detector.detect(img_rgb)

                # Extract wrist position
                wrist_3d = None
                if keypoint_2d is not None:
                    wrist_landmark = keypoint_2d.landmark[MANO_WRIST]
                    wrist_x = wrist_landmark.x * w  # x in pixels
                    wrist_y = wrist_landmark.y * h  # y in pixels

                    # Get depth: either from depth camera or MediaPipe estimate
                    if self.use_depth and depth_frame is not None:
                        wrist_z = self._get_depth_at_point(depth_frame, wrist_x, wrist_y)
                        if wrist_z <= 0:
                            # Fallback to MediaPipe estimate if depth invalid
                            wrist_z = 0.5  # Default 0.5m
                    else:
                        # Use MediaPipe's relative z (scaled heuristically)
                        # MediaPipe z is relative depth, roughly in same scale as hand size
                        wrist_z = 0.5 - wrist_landmark.z * 0.5  # Heuristic: ~0.5m base distance

                    wrist_3d = np.array([wrist_x, wrist_y, wrist_z], dtype=np.float32)

                with self.lock:
                    if joint_pos is not None:
                        self.latest_result = (joint_pos, wrist_rot)
                    else:
                        self.latest_result = None
                    self.latest_img = img_bgr.copy()
                    self.latest_depth_img = depth_image
                    self.latest_keypoint_2d = keypoint_2d
                    self.latest_wrist_3d = wrist_3d
                    self.latest_ts = time.monotonic()

                    # Update FPS
                    self._fps_count += 1
                    dt = self.latest_ts - self._fps_t0
                    if dt >= 1.0:
                        self.fps = self._fps_count / dt
                        self._fps_count = 0
                        self._fps_t0 = self.latest_ts

            except Exception as e:
                print(f"[HandTracker] Error: {e}")
                continue

    def get_latest(self):
        with self.lock:
            return self.latest_result, self.latest_wrist_3d, self.latest_ts

    def get_latest_vis(self):
        with self.lock:
            return self.latest_img, self.latest_keypoint_2d, self.latest_depth_img, self.latest_ts

    def get_fps(self) -> float:
        with self.lock:
            return float(self.fps)


def main():
    parser = argparse.ArgumentParser(description="Realtime hand tracking -> Franka arm + gripper")

    # Camera settings
    parser.add_argument("--camera-path", default=None, help="OpenCV camera path (default: 0)")
    parser.add_argument("--use-realsense", action="store_true", help="Use RealSense camera")
    parser.add_argument("--use-depth", action=argparse.BooleanOptionalAction, default=True,
                        help="Use RGB-D (depth) for 3D tracking (requires RealSense, enabled by default)")
    parser.add_argument("--rs-model", default="D435", help="Preferred RealSense model")
    parser.add_argument("--rs-serial", default=None, help="RealSense serial (optional)")
    parser.add_argument("--fps", type=float, default=30.0, help="Camera/control FPS")

    # Hand tracking settings
    parser.add_argument("--hand-type", choices=["Left", "Right"], default="Right",
                        help="Which hand to track")
    parser.add_argument("--selfie", action="store_true", help="Selfie mode (mirror image)")

    # Franka settings
    parser.add_argument("--tac-foundation", default="/home/yxma/tac_foundation",
                        help="Path to tac_foundation repo")
    parser.add_argument("--frankapy-repo", default="~/frankapy",
                        help="Path to frankapy source repo")
    parser.add_argument("--arm-id", choices=["left", "right"], default="left",
                        help="Which Franka arm to control")
    parser.add_argument("--use-gripper", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable gripper control")

    # Tracking/control settings
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Position scale for wrist deltas")
    parser.add_argument("--axis-map", default="1,1,1",
                        help="Axis scale mapping for xyz deltas (e.g., '1,-1,1')")
    parser.add_argument("--coord-order", default="zxy",
                        help="Camera to robot coordinate mapping (e.g., 'zxy' means cam_z->robot_x, cam_x->robot_y, cam_y->robot_z)")
    parser.add_argument("--show-coords", action="store_true",
                        help="Print camera and robot coordinates for debugging")
    parser.add_argument("--max-step", type=float, default=0.03,
                        help="Max translation per step (m)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Smoothing factor for delta [0..1]")
    parser.add_argument("--lpf-alpha", type=float, default=0.0,
                        help="Extra low-pass on target [0..1], 0 disables")
    parser.add_argument("--gripper-scale", type=float, default=12.5,
                        help="Scale thumb-to-index distance to gripper width [0..1]")
    parser.add_argument("--gripper-offset", type=float, default=0.0,
                        help="Offset for gripper width [0..1]")
    parser.add_argument("--max-age", type=float, default=0.5,
                        help="Max age (s) for hand tracking result")
    parser.add_argument("--use-speed-limit", action=argparse.BooleanOptionalAction, default=False,
                        help="Use FrankaPy speed-limited interpolation")

    # Rotation tracking settings
    parser.add_argument("--track-rotation", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable rotation tracking (arm follows hand orientation)")
    parser.add_argument("--rot-scale", type=float, default=1.0,
                        help="Scale factor for rotation (1.0 = 1:1 mapping)")
    parser.add_argument("--rot-alpha", type=float, default=0.3,
                        help="Smoothing factor for rotation [0..1]")

    # Force feedback settings (finger sensor -> gripper force)
    parser.add_argument("--use-force-feedback", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable force feedback: finger sensor controls gripper force")
    parser.add_argument("--tactile-only", action="store_true",
                        help="Disable arm tracking, use tactile sensor for gripper only")
    parser.add_argument("--finger-port", default="/dev/ttyUSB1",
                        help="Serial port for finger tactile sensor")
    parser.add_argument("--force-scale", type=float, default=1.0,
                        help="Scale factor from finger force to gripper force")
    parser.add_argument("--force-offset", type=float, default=0.0,
                        help="Offset for gripper force command")
    parser.add_argument("--max-gripper-force", type=float, default=40.0,
                        help="Maximum gripper force (N)")
    parser.add_argument("--contact-threshold", type=float, default=5.0,
                        help="Force threshold to detect contact and switch to force mode")
    parser.add_argument("--tactile-alpha", type=float, default=0.5,
                        help="Temporal smoothing for tactile sensor [0..1]")
    parser.add_argument("--viz", action=argparse.BooleanOptionalAction, default=True,
                        help="Show realtime visualization")
    parser.add_argument("--viz-scale", type=float, default=0.75,
                        help="Scale for visualization window")

    args = parser.parse_args()

    # Add paths for imports
    tidybot2_path = os.path.join(args.tac_foundation, "third_party", "tidybot2")
    _add_path(tidybot2_path)
    frankapy_repo = os.path.expanduser(args.frankapy_repo)
    if os.path.isdir(frankapy_repo):
        _add_path(frankapy_repo)

    # Stub out ik_solver if pink/mujoco not available (not needed for Cartesian teleop)
    if importlib.util.find_spec("pink") is None or importlib.util.find_spec("mujoco") is None:
        stub = types.ModuleType("ik_solver")
        class IKSolverPink:
            def __init__(self, *args, **kwargs):
                pass
            def fk(self, *args, **kwargs):
                return None
        stub.IKSolverPink = IKSolverPink
        sys.modules["ik_solver"] = stub

    # Import FrankaPy (after adding paths)
    try:
        from frankapy_arm import FrankaPyArm
        from constants import ADELSONLAB_FRANKA_ARM_CONFIG
    except ImportError as e:
        print(f"ERROR: Could not import FrankaPy components: {e}")
        print(f"       Make sure tac_foundation path is correct: {args.tac_foundation}")
        return 1

    # Initialize deadman switch
    deadman = Deadman()

    # Initialize Franka arm
    print("Initializing FrankaPy...")
    arm_cfg = ADELSONLAB_FRANKA_ARM_CONFIG[args.arm_id]
    arm = FrankaPyArm(config=arm_cfg, use_gripper=args.use_gripper)

    # Home the robot first
    print("Homing robot to initial position...")
    arm.reset_joints(duration=5, block=True)

    # Start impedance control
    arm.start_impedance_control(duration=10000.0, use_smooth_teleop=True)

    # Get initial arm state after homing
    state = arm.get_state()
    initial_pos = np.array(state["arm_pos"], dtype=np.float32)
    fixed_rot = FrankaPyArm._quat_to_rotation_matrix(state["arm_quat"])

    # Move arm down 0.2m along Z axis to finish initialization
    print("Moving arm to starting position (Z - 0.2m)...")
    target_pos = initial_pos.copy()
    target_pos[2] -= 0.2  # Move down 0.2m in Z
    arm.send_pose_goal(
        {"translation": target_pos, "rotation": fixed_rot},
        use_speed_limit=True
    )
    time.sleep(2.0)  # Wait for movement to complete

    # Update initial state after repositioning
    state = arm.get_state()
    last_target = np.array(state["arm_pos"], dtype=np.float32)
    fixed_rot = FrankaPyArm._quat_to_rotation_matrix(state["arm_quat"])
    print(f"Robot ready at position: {last_target}")

    # Initialize camera (skip in tactile-only mode)
    align = None
    depth_scale = 0.001
    use_depth = args.use_depth
    capture = None
    detector = None
    tracker = None

    if not args.tactile_only:
        print("Starting camera...")
        if args.use_realsense:
            if not HAS_REALSENSE:
                print("ERROR: pyrealsense2 not installed. Install with: pip install pyrealsense2")
                arm.stop_impedance_control(silent=True)
                arm.close()
                return 1

            pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_serial = args.rs_serial or _select_realsense_serial(args.rs_model)
            if rs_serial:
                rs_config.enable_device(rs_serial)
                print(f"Using RealSense serial: {rs_serial}")

            # Enable color stream
            rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, int(args.fps))

            # Enable depth stream if requested
            if use_depth:
                rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, int(args.fps))
                print("RGB-D mode enabled - using depth camera for 3D tracking")

            profile = pipeline.start(rs_config)

            # Get depth scale if using depth
            if use_depth:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                print(f"Depth scale: {depth_scale}")
                # Create alignment object to align depth to color
                align = rs.align(rs.stream.color)

            capture = pipeline
        else:
            if use_depth:
                print("WARNING: --use-depth requires --use-realsense, disabling depth")
                use_depth = False
            camera_path = int(args.camera_path) if args.camera_path and args.camera_path.isdigit() else (args.camera_path or 0)
            capture = cv2.VideoCapture(camera_path)
            if not capture.isOpened():
                print(f"ERROR: Could not open camera: {camera_path}")
                arm.stop_impedance_control(silent=True)
                arm.close()
                return 1

        # Initialize hand detector
        print("Initializing hand detector...")
        detector = SingleHandDetector(hand_type=args.hand_type, selfie=args.selfie)
    else:
        print("Tactile-only mode: skipping camera and hand detector")

    # Initialize tactile sensor for force feedback
    finger_sensor = None
    force_controller = None

    if args.use_force_feedback:
        print("Initializing finger tactile sensor for force control...")

        # Initialize finger sensor (combined thumb + index)
        print(f"  Finger sensor: {args.finger_port}")
        finger_sensor = TactileSensor(
            args.finger_port,
            name="finger",
            alpha=args.tactile_alpha
        )

        # Start sensor and wait for initialization
        print("  Starting finger sensor (calibrating baseline)...")
        if not finger_sensor.start(timeout=30.0):
            print("ERROR: Failed to initialize finger sensor")
            arm.stop_impedance_control(silent=True)
            arm.close()
            return 1

        # Create finger force controller (hybrid: position when no contact, force when contact)
        force_controller = FingerForceController(
            finger_sensor,
            force_scale=args.force_scale,
            force_offset=args.force_offset,
            max_force=args.max_gripper_force,
            contact_threshold=args.contact_threshold,
            close_speed=0.04,
            alpha=0.2,
        )
        print("  Force feedback initialized (hybrid mode)!")

    # Start hand tracking thread (skip in tactile-only mode)
    if not args.tactile_only:
        tracker = HandTrackerThread(
            capture, detector,
            use_depth=use_depth,
            align=align,
            depth_scale=depth_scale
        )
        tracker.start()

    # Control state
    axis_map = _parse_axis_map(args.axis_map)
    # coord_indices, coord_signs = _parse_coord_order(args.coord_order)
    # print(f"[coord] Coordinate mapping: camera->{args.coord_order} to robot xyz")
    # print(f"[coord] Camera axis indices: {coord_indices}, signs: {coord_signs}")
    print(f"[coord] Using hardcoded coordinate mapping (see lines 752-770 to adjust)")
    rate = RateLimiter(frequency=args.fps)
    last_wrist = None
    last_delta = np.zeros(3, dtype=np.float32)
    tracking_valid = False
    lpf_target = None

    # Rotation tracking state
    track_rotation = args.track_rotation
    last_hand_rot = None  # Last hand rotation from MediaPipe
    current_rot = fixed_rot.copy()  # Current robot rotation (starts at initial pose)
    last_delta_rot = np.zeros(3, dtype=np.float32)  # Smoothed rotation delta (axis-angle)

    # FPS tracking
    loop_fps = 0.0
    loop_count = 0
    loop_t0 = time.monotonic()
    last_fps_print = time.monotonic()

    # Signal handler
    def _signal_handler(_sig, _frame):
        deadman.quit_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Visualization window
    if args.viz:
        cv2.namedWindow("Franka Retargeting", cv2.WINDOW_NORMAL)

        # Create tactile sensor visualization window if force feedback is enabled
        if args.use_force_feedback and finger_sensor is not None:
            TACTILE_WINDOW_WIDTH = 16 * 50  # 800 pixels
            TACTILE_WINDOW_HEIGHT = 16 * 50  # 800 pixels
            cv2.namedWindow("Tactile Sensor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tactile Sensor", TACTILE_WINDOW_WIDTH, TACTILE_WINDOW_HEIGHT)

    print("\n" + "="*60)
    print("Franka Arm Retargeting Ready!")
    print("="*60)
    print("Controls:")
    print("  Space: Toggle motion enable/disable")
    print("  r: Reset reference position")
    print("  t: Toggle rotation tracking on/off")
    print("  c: Reset tactile sensor baseline")
    print("  q: Quit")
    print("")
    print("Modes:")
    print(f"  Rotation tracking: {'ON' if track_rotation else 'OFF'} (press 't' to toggle)")
    print(f"  Force feedback:    {'ON' if args.use_force_feedback else 'OFF'}")
    if args.use_force_feedback:
        print(f"    Finger port:  {args.finger_port}")
        print(f"    Force scale:  {args.force_scale}, max: {args.max_gripper_force}N")
    print("="*60 + "\n")

    try:
        while not deadman.quit_requested:
            # Get hand tracking result (skip in tactile-only mode)
            result = None
            wrist_3d = None
            result_ts = None
            joint_pos = None
            hand_detected = False

            if tracker is not None:
                result, wrist_3d, result_ts = tracker.get_latest()

                # Check for valid tracking data
                if result_ts is not None and (time.monotonic() - result_ts) <= args.max_age:
                    if result is not None and wrist_3d is not None:
                        joint_pos, wrist_rot = result
                        if _is_valid_keypoints(joint_pos):
                            hand_detected = True

            # Handle reset request (always check, even without tracking)
            if deadman.consume_reset():
                tracking_valid = False
                last_wrist = None
                last_delta = np.zeros(3, dtype=np.float32)
                last_hand_rot = None
                last_delta_rot = np.zeros(3, dtype=np.float32)
                # Reset to current arm position and rotation
                state = arm.get_state()
                last_target = np.array(state["arm_pos"], dtype=np.float32)
                current_rot = FrankaPyArm._quat_to_rotation_matrix(state["arm_quat"])
                print("[reset] Reference position and rotation reset")

            # Handle rotation tracking toggle request
            if deadman.consume_rotation_toggle():
                track_rotation = not track_rotation
                state_str = "ON" if track_rotation else "OFF"
                print(f"[rotation] Rotation tracking {state_str}")
                if not track_rotation:
                    # When disabling, reset rotation state
                    last_hand_rot = None
                    last_delta_rot = np.zeros(3, dtype=np.float32)

            # Handle sensor reset request
            if deadman.consume_sensor_reset():
                if finger_sensor is not None:
                    finger_sensor.reset_baseline()
                    # Also reset force controller state
                    if force_controller is not None:
                        force_controller.smoothed_force = 0.0
                        force_controller.current_width = 1.0
                        force_controller.in_contact = False

            # Process tracking if hand detected
            if hand_detected:
                # wrist_3d is [x_pixels, y_pixels, z_meters]
                # Convert to robot coordinates:
                # - X (pixels) -> robot Y (left/right)
                # - Y (pixels) -> robot Z (up/down)
                # - Z (meters) -> robot X (forward/back)

                if use_depth:
                    # RGB-D mode: z is actual depth in meters
                    # Use camera intrinsics approximation for x,y conversion
                    # Assume fx ~ fy ~ 600 for 640x480 resolution
                    focal_length = 600.0
                    cx, cy = 320.0, 240.0
                    z_meters = wrist_3d[2] if wrist_3d[2] > 0.1 else 0.5

                    # Convert pixel coordinates to meters using pinhole model
                    x_meters = (wrist_3d[0] - cx) * z_meters / focal_length
                    y_meters = (wrist_3d[1] - cy) * z_meters / focal_length

                    # ============================================================
                    # COORDINATE MAPPING: Camera frame → Robot base frame
                    # ============================================================
                    # Camera frame:
                    #   x_meters: horizontal (right is positive)
                    #   y_meters: vertical (down is positive)
                    #   z_meters: depth (away from camera is positive)
                    #
                    # Robot base frame:
                    #   X: forward
                    #   Y: left
                    #   Z: up
                    #
                    # ADJUST THIS MAPPING based on your camera mounting:
                    wrist_pos = np.array([
                        -x_meters,   # Robot X: camera horizontal (left/right)
                        -z_meters,   # Robot Y: camera depth (forward/back)
                        -y_meters,   # Robot Z: camera vertical (up/down)
                    ], dtype=np.float32)
                    # ============================================================


                    print(f"[wrist_3d] Raw: x={wrist_3d[0]:.1f}px, y={wrist_3d[1]:.1f}px, z={wrist_3d[2]:.3f}m")
                    print(f"[coord] Camera: x={x_meters:.3f}, y={y_meters:.3f}, z={z_meters:.3f} | "
                        f"Robot: x={wrist_pos[0]:.3f}, y={wrist_pos[1]:.3f}, z={wrist_pos[2]:.3f}")
                else:
                    # RGB-only mode: use pixel-based heuristic
                    pixel_to_meter = 0.0008
                    wrist_pos = np.array([
                        -(wrist_3d[0] - 320) * pixel_to_meter,  # X pixels -> robot X
                        -(wrist_3d[2] - 0.5) * 0.5,  # Z: depth estimate -> robot Y
                        -(wrist_3d[1] - 240) * pixel_to_meter,  # Y pixels -> robot Z
                    ], dtype=np.float32)

                # Initialize tracking on first valid detection
                if not tracking_valid or last_wrist is None:
                    last_wrist = wrist_pos
                    if track_rotation:
                        last_hand_rot = wrist_rot.copy()
                    tracking_valid = True
                    lpf_target = None
                else:
                    # Compute position delta
                    delta = wrist_pos - last_wrist
                    last_wrist = wrist_pos

                    # Apply scale and axis mapping
                    delta = delta * args.scale * axis_map

                    # Clamp delta magnitude
                    delta_norm = float(np.linalg.norm(delta))
                    if delta_norm > args.max_step > 0:
                        delta = delta * (args.max_step / delta_norm)

                    # Smooth delta with exponential moving average
                    delta = args.alpha * delta + (1.0 - args.alpha) * last_delta
                    last_delta = delta

                    # Compute target position
                    target = last_target + delta

                    # Optional extra low-pass filter on target
                    if args.lpf_alpha > 0.0:
                        if lpf_target is None:
                            lpf_target = target.copy()
                        lpf_target = args.lpf_alpha * target + (1.0 - args.lpf_alpha) * lpf_target
                        target = lpf_target

                    # Compute rotation delta if tracking rotation
                    target_rot = current_rot  # Default: keep current rotation
                    if track_rotation and last_hand_rot is not None:
                        # Compute relative rotation: how hand rotated since last frame
                        # delta_rot_hand = wrist_rot @ last_hand_rot.T
                        delta_rot_hand = wrist_rot @ last_hand_rot.T
                        last_hand_rot = wrist_rot.copy()

                        # Convert to axis-angle for scaling and smoothing
                        delta_aa = _rotation_matrix_to_axis_angle(delta_rot_hand)

                        # Apply rotation scale
                        delta_aa = delta_aa * args.rot_scale

                        # Smooth rotation delta
                        delta_aa = args.rot_alpha * delta_aa + (1.0 - args.rot_alpha) * last_delta_rot
                        last_delta_rot = delta_aa

                        # Convert hand rotation delta to robot frame
                        # Transform from camera/hand frame to robot frame
                        camera_to_robot = np.array([
                            [0, 0, 1],   # Robot X <- Camera Z
                            [-1, 0, 0],  # Robot Y <- -Camera X
                            [0, -1, 0],  # Robot Z <- -Camera Y
                        ], dtype=np.float32)

                        # Transform the rotation axis to robot frame
                        delta_aa_robot = camera_to_robot @ delta_aa

                        # Apply delta rotation to current robot rotation
                        delta_rot_robot = _axis_angle_to_rotation_matrix(delta_aa_robot)
                        target_rot = delta_rot_robot @ current_rot

                    # Send commands if motion enabled
                    if deadman.motion_enabled and not args.tactile_only:
                        # Debug: show delta being applied
                        if np.linalg.norm(delta) > 0.0001:
                            rot_info = ""
                            if track_rotation:
                                rot_info = f", rot_delta={np.linalg.norm(last_delta_rot)*180/np.pi:.1f}deg"
                            print(f"[motion] delta={delta*1000}mm{rot_info}, target={target}")

                        arm.send_pose_goal(
                            {"translation": target, "rotation": target_rot},
                            use_speed_limit=args.use_speed_limit
                        )

                        # Gripper control (with hand tracking)
                        if args.use_gripper and joint_pos is not None:
                            # Get position-based width estimate from hand tracking
                            width_norm = _compute_gripper_width_norm(
                                joint_pos, args.gripper_scale, args.gripper_offset
                            )

                            # Hybrid force control if enabled
                            if force_controller is not None:
                                # Hybrid: position when no contact, force when contact
                                grip_width, grip_force, in_contact = force_controller.update(width_norm)
                                debug_info = force_controller.get_debug_info()
                                # Periodically print force feedback debug info
                                if int(time.monotonic() * 2) % 2 == 0:
                                    mode = "FORCE" if in_contact else "POS"
                                    print(f"[{mode}] finger={debug_info['finger_force']:.1f} "
                                          f"-> width={grip_width:.2f} force={grip_force:.1f}N")
                                arm.goto_gripper(grip_width, force=grip_force, block=False)
                            else:
                                arm.goto_gripper(width_norm, block=False)

                        last_target = target
                        if track_rotation:
                            current_rot = target_rot
            else:
                tracking_valid = False

            # Tactile-only mode: gripper control independent of hand tracking
            if args.tactile_only and deadman.motion_enabled and force_controller is not None:
                # Use fixed open position as baseline, let tactile control the grip
                grip_width, grip_force, in_contact = force_controller.update(1.0)  # 1.0 = open
                debug_info = force_controller.get_debug_info()

                # When force is below threshold, actively open the gripper
                if not in_contact:
                    grip_width = 1.0  # Ensure gripper opens fully
                    grip_force = 5.0  # Low force for opening

                # Periodically print force feedback debug info
                if int(time.monotonic() * 2) % 2 == 0:
                    mode = "FORCE" if in_contact else "OPEN"
                    print(f"[TACTILE-ONLY {mode}] finger={debug_info['finger_force']:.1f} "
                          f"-> width={grip_width:.2f} force={grip_force:.1f}N")
                arm.goto_gripper(grip_width, force=grip_force, block=False)

            # Update FPS
            loop_count += 1
            now = time.monotonic()
            dt = now - loop_t0
            if dt >= 1.0:
                loop_fps = loop_count / dt
                loop_count = 0
                loop_t0 = now

            # Print status periodically
            if now - last_fps_print >= 1.0:
                motion_status = "ACTIVE" if deadman.motion_enabled else "PAUSED"
                if args.tactile_only:
                    print(f"[FPS] loop={loop_fps:.1f} | Motion: {motion_status} | TACTILE-ONLY")
                else:
                    track_status = "TRACKING" if hand_detected else "NO HAND"
                    print(f"[FPS] loop={loop_fps:.1f} tracker={tracker.get_fps():.1f} | "
                          f"Motion: {motion_status} | Hand: {track_status}")
                last_fps_print = now

            # Visualization - ALWAYS run this regardless of tracking (skip in tactile-only)
            if args.viz and tracker is not None:
                img, kp2d, depth_img, _ = tracker.get_latest_vis()
                if img is not None:
                    vis = img.copy()

                    # Draw hand skeleton if detected
                    if kp2d is not None:
                        vis = detector.draw_skeleton_on_image(vis, kp2d, style="default")

                    # Add status text
                    motion_status = "MOTION: ON" if deadman.motion_enabled else "MOTION: OFF (Space to enable)"
                    color = (0, 255, 0) if deadman.motion_enabled else (0, 0, 255)
                    cv2.putText(vis, motion_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    track_status = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
                    track_color = (0, 255, 0) if hand_detected else (0, 165, 255)
                    cv2.putText(vis, track_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, track_color, 2)

                    # Show depth and rotation info
                    mode_parts = []
                    mode_parts.append("RGB-D" if use_depth else "RGB")
                    mode_parts.append("ROT" if track_rotation else "POS-only")
                    mode_parts.append("FORCE" if force_controller else "POS-grip")
                    mode_status = " | ".join(mode_parts)
                    if use_depth and wrist_3d is not None:
                        mode_status += f" | z={wrist_3d[2]:.2f}m"
                    cv2.putText(vis, mode_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 200, 0), 2)
                    cv2.putText(vis, f"FPS: {loop_fps:.1f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Show force feedback info if enabled
                    if force_controller is not None:
                        debug_info = force_controller.get_debug_info()
                        mode = "FORCE" if debug_info['in_contact'] else "POS"
                        mode_color = (0, 0, 255) if debug_info['in_contact'] else (0, 255, 0)
                        force_text = (f"[{mode}] Finger: {debug_info['finger_force']:.0f} "
                                      f"-> W={debug_info['current_width']:.2f} F={debug_info['gripper_force_cmd']:.1f}N")
                        cv2.putText(vis, force_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, mode_color, 2)

                    # Resize if needed
                    if args.viz_scale != 1.0:
                        vis = cv2.resize(
                            vis,
                            (int(vis.shape[1] * args.viz_scale), int(vis.shape[0] * args.viz_scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                    cv2.imshow("Franka Retargeting", vis)

            # Tactile sensor visualization (independent of camera/tracker)
            if args.use_force_feedback and finger_sensor is not None:
                # Get contact map from sensor
                contact_map = finger_sensor.get_contact_map()

                # Scale to 0-255 and convert to uint8
                contact_map_scaled = (contact_map * 255).astype(np.uint8)

                # Apply color map (VIRIDIS)
                tactile_colormap = cv2.applyColorMap(contact_map_scaled, cv2.COLORMAP_VIRIDIS)

                # Resize for better visibility
                TACTILE_WINDOW_WIDTH = 16 * 50
                TACTILE_WINDOW_HEIGHT = 16 * 50
                tactile_resized = cv2.resize(
                    tactile_colormap,
                    (TACTILE_WINDOW_WIDTH, TACTILE_WINDOW_HEIGHT),
                    interpolation=cv2.INTER_NEAREST
                )

                # Get force information
                finger_force = finger_sensor.get_total_force()
                max_force_val = finger_sensor.get_max_force()

                # Get gripper command info if force controller is active
                gripper_force_cmd = 0.0
                gripper_width_cmd = 1.0
                control_mode = "N/A"
                if force_controller is not None:
                    debug_info = force_controller.get_debug_info()
                    gripper_force_cmd = debug_info['gripper_force_cmd']
                    gripper_width_cmd = debug_info['current_width']
                    control_mode = "FORCE" if debug_info['in_contact'] else "POS"

                # Prepare text
                sensor_text = f"Tactile Force: {finger_force:.1f}"
                max_text = f"Peak Cell: {max_force_val:.2f}"
                gripper_text = f"Gripper Cmd: {gripper_force_cmd:.1f} N"
                max_gripper_text = f"Max Limit: {args.max_gripper_force:.0f} N"
                mode_text = f"Mode: {control_mode}"
                width_text = f"Width: {gripper_width_cmd:.2f}"

                # Draw larger semi-transparent background for all text
                cv2.rectangle(tactile_resized, (5, 5), (400, 230), (0, 0, 0), -1)
                cv2.rectangle(tactile_resized, (5, 5), (400, 230), (255, 255, 255), 2)

                # Draw text with better organization
                y_offset = 35
                line_height = 32

                # Tactile sensor info
                cv2.putText(tactile_resized, "TACTILE SENSOR:", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
                y_offset += line_height
                cv2.putText(tactile_resized, sensor_text, (25, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                y_offset += line_height
                cv2.putText(tactile_resized, max_text, (25, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                # Gripper command info
                y_offset += line_height + 5
                cv2.putText(tactile_resized, "GRIPPER COMMAND:", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                y_offset += line_height

                # Highlight force command based on magnitude
                force_ratio = gripper_force_cmd / args.max_gripper_force
                force_color = (255, 255, 255) if force_ratio < 0.7 else (0, 165, 255)  # Orange if high
                cv2.putText(tactile_resized, gripper_text, (25, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, force_color, 2)
                y_offset += line_height
                cv2.putText(tactile_resized, max_gripper_text, (25, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.imshow("Tactile Sensor", tactile_resized)

            # Always process keyboard for visualization windows
            if args.viz:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    deadman.quit_requested = True
                elif key == ord(" "):  # Space key
                    deadman.motion_enabled = not deadman.motion_enabled
                    state_str = "ENABLED" if deadman.motion_enabled else "PAUSED"
                    print(f"[cv2] Motion {state_str}")
                elif key == ord("r"):
                    deadman.reset_requested = True
                    print("[cv2] Reset reference requested")
                elif key == ord("t"):
                    deadman.rotation_toggle_requested = True
                    print("[cv2] Rotation tracking toggle requested")
                elif key == ord("c"):
                    deadman.sensor_reset_requested = True
                    print("[cv2] Sensor reset requested")

            rate.sleep()

    finally:
        print("\nShutting down...")

        # Stop tactile sensor
        if finger_sensor is not None:
            print("Stopping finger sensor...")
            finger_sensor.stop()

        # Stop hand tracker
        if tracker is not None:
            tracker.stop_event.set()
            tracker.join(timeout=1.0)

        # Stop camera
        if capture is not None:
            if args.use_realsense:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            else:
                try:
                    capture.release()
                except Exception:
                    pass

        # Close visualization
        if args.viz:
            cv2.destroyAllWindows()

        # Stop robot
        try:
            arm.stop_impedance_control(silent=True)
            arm.close()
        except Exception as e:
            print(f"Error closing arm: {e}")

        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())