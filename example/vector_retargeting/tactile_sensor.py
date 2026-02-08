#!/usr/bin/env python3
"""
Tactile sensor interface for 16x16 tactile arrays via serial.

Supports initialization with baseline calibration and real-time force reading.
"""

import numpy as np
import serial
import threading
import time
from typing import Optional, Tuple


class TactileSensor:
    """
    Thread-safe interface to a 16x16 tactile sensor array.

    Handles:
    - Serial communication at 2Mbaud
    - Initialization with baseline calibration (collects samples for median)
    - Real-time normalized contact data
    - Total force computation
    """

    THRESHOLD = 12
    NOISE_SCALE = 60
    INIT_SAMPLES = 30
    BAUD_RATE = 2000000

    def __init__(self, port: str, name: str = "tactile", alpha: float = 0.5):
        """
        Initialize tactile sensor.

        Args:
            port: Serial port path (e.g., '/dev/ttyUSB0' or 'left_robot_left_finger')
            name: Human-readable name for logging
            alpha: Temporal smoothing factor (0-1, higher = more responsive)
        """
        self.port = port
        self.name = name
        self.alpha = alpha

        # State
        self.median = None  # Baseline from calibration
        self.contact_data_norm = np.zeros((16, 16), dtype=np.float32)
        self.prev_frame = np.zeros((16, 16), dtype=np.float32)
        self.initialized = False
        self.running = False

        # Thread-safe access
        self.lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Serial device
        self._serial: Optional[serial.Serial] = None

    def start(self, timeout: float = 30.0) -> bool:
        """
        Start sensor and wait for initialization.

        Args:
            timeout: Max seconds to wait for initialization

        Returns:
            True if initialized successfully, False otherwise
        """
        if self.running:
            return self.initialized

        try:
            self._serial = serial.Serial(self.port, self.BAUD_RATE)
            self._serial.flush()
        except Exception as e:
            print(f"[{self.name}] Failed to open serial port {self.port}: {e}")
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self.running = True

        # Wait for initialization
        start_time = time.monotonic()
        while not self.initialized and (time.monotonic() - start_time) < timeout:
            time.sleep(0.1)

        if not self.initialized:
            print(f"[{self.name}] Initialization timed out after {timeout}s")
            return False

        print(f"[{self.name}] Initialized successfully")
        return True

    def stop(self):
        """Stop the sensor reading thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
        self.running = False

    def _read_loop(self):
        """Main reading loop (runs in background thread)."""
        data_tac = []
        current = None
        backup = None

        # Phase 1: Collect samples for baseline calibration
        while not self._stop_event.is_set() and len(data_tac) < self.INIT_SAMPLES:
            if self._serial.in_waiting > 0:
                try:
                    line = self._serial.readline().decode('utf-8').strip()
                except Exception:
                    line = ""

                if len(line) < 10:
                    # Check if we have a complete 16x16 frame
                    if current is not None and len(current) == 16:
                        try:
                            backup = np.array(current, dtype=np.float32)
                            # Verify shape is correct (16, 16)
                            if backup.shape == (16, 16):
                                data_tac.append(backup)
                        except Exception:
                            pass  # Skip malformed frames
                    current = []
                    continue

                if current is not None:
                    try:
                        str_values = line.split()
                        int_values = [int(val) for val in str_values]
                        # Only append if we have exactly 16 values (valid row)
                        if len(int_values) == 16:
                            current.append(int_values)
                    except Exception:
                        pass

        if self._stop_event.is_set():
            return

        # Compute median baseline
        data_tac = np.array(data_tac)
        self.median = np.median(data_tac, axis=0)

        with self.lock:
            self.initialized = True

        print(f"[{self.name}] Calibration complete (collected {len(data_tac)} samples)")

        # Phase 2: Continuous reading
        current = None
        backup = None

        while not self._stop_event.is_set():
            if self._serial.in_waiting > 0:
                try:
                    line = self._serial.readline().decode('utf-8').strip()
                except Exception:
                    line = ""

                if len(line) < 10:
                    # Check if we have a complete 16x16 frame
                    if current is not None and len(current) == 16:
                        try:
                            temp_array = np.array(current, dtype=np.float32)
                            # Verify shape is correct (16, 16)
                            if temp_array.shape == (16, 16):
                                backup = temp_array
                        except Exception:
                            pass  # Skip malformed frames
                    current = []

                    if backup is not None:
                        # Subtract baseline and threshold
                        contact_data = backup - self.median - self.THRESHOLD
                        contact_data = np.clip(contact_data, 0, 100)

                        # Normalize
                        if np.max(contact_data) < self.THRESHOLD:
                            contact_data_norm = contact_data / self.NOISE_SCALE
                        else:
                            contact_data_norm = contact_data / np.max(contact_data)

                        # Temporal filtering
                        contact_data_norm = (
                            self.alpha * contact_data_norm +
                            (1.0 - self.alpha) * self.prev_frame
                        )
                        self.prev_frame = contact_data_norm

                        with self.lock:
                            self.contact_data_norm = contact_data_norm.astype(np.float32)

                    continue

                if current is not None:
                    try:
                        str_values = line.split()
                        int_values = [int(val) for val in str_values]
                        # Only append if we have exactly 16 values (valid row)
                        if len(int_values) == 16:
                            current.append(int_values)
                    except Exception:
                        pass

    def get_contact_map(self) -> np.ndarray:
        """Get current normalized contact map (16x16, values 0-1)."""
        with self.lock:
            return self.contact_data_norm.copy()

    def get_total_force(self, exclude_top_n: int = 3) -> float:
        """
        Get total contact force (sum of all cells, excluding top N peaks).

        Args:
            exclude_top_n: Number of highest cells to exclude from sum (default: 3)
                          Set to 0 to include all cells.

        Returns:
            Total force excluding the top N peak values
        """
        with self.lock:
            if exclude_top_n <= 0:
                return float(np.sum(self.contact_data_norm))

            # Flatten array and find top N values
            contact_flat = self.contact_data_norm.flatten()

            # Get indices of top N values
            if len(contact_flat) <= exclude_top_n:
                return 0.0  # Edge case: excluding more cells than exist

            top_n_indices = np.argpartition(contact_flat, -exclude_top_n)[-exclude_top_n:]
            top_n_sum = np.sum(contact_flat[top_n_indices])

            # Return total minus top N peaks
            return float(np.sum(contact_flat) - top_n_sum)

    def get_max_force(self) -> float:
        """Get maximum contact force (peak cell value)."""
        with self.lock:
            return float(np.max(self.contact_data_norm))

    def get_contact_area(self, threshold: float = 0.1) -> int:
        """Get number of cells with contact above threshold."""
        with self.lock:
            return int(np.sum(self.contact_data_norm > threshold))

    def is_initialized(self) -> bool:
        """Check if sensor has completed initialization."""
        with self.lock:
            return self.initialized

    def reset_baseline(self):
        """Reset baseline to current readings (re-zero the sensor)."""
        with self.lock:
            # Set current readings as new baseline by adding current contact to median
            if self.median is not None:
                # Accumulate current offset into baseline
                self.median = self.median + self.contact_data_norm * self.NOISE_SCALE
                self.contact_data_norm = np.zeros((16, 16), dtype=np.float32)
                self.prev_frame = np.zeros((16, 16), dtype=np.float32)
                print(f"[{self.name}] Baseline reset")


class TactileSensorPair:
    """
    Manages a pair of tactile sensors (e.g., thumb+index finger).
    """

    def __init__(self, port1: str, port2: str, name1: str = "sensor1", name2: str = "sensor2",
                 alpha: float = 0.5):
        """
        Initialize sensor pair.

        Args:
            port1: Serial port for first sensor
            port2: Serial port for second sensor
            name1: Name for first sensor
            name2: Name for second sensor
            alpha: Temporal smoothing factor
        """
        self.sensor1 = TactileSensor(port1, name1, alpha)
        self.sensor2 = TactileSensor(port2, name2, alpha)

    def start(self, timeout: float = 30.0) -> bool:
        """Start both sensors and wait for initialization."""
        ok1 = self.sensor1.start(timeout)
        ok2 = self.sensor2.start(timeout)
        return ok1 and ok2

    def stop(self):
        """Stop both sensors."""
        self.sensor1.stop()
        self.sensor2.stop()

    def get_total_force(self) -> float:
        """Get combined total force from both sensors."""
        return self.sensor1.get_total_force() + self.sensor2.get_total_force()

    def get_max_force(self) -> float:
        """Get maximum force across both sensors."""
        return max(self.sensor1.get_max_force(), self.sensor2.get_max_force())

    def get_forces(self) -> Tuple[float, float]:
        """Get individual total forces from each sensor."""
        return self.sensor1.get_total_force(), self.sensor2.get_total_force()

    def is_initialized(self) -> bool:
        """Check if both sensors are initialized."""
        return self.sensor1.is_initialized() and self.sensor2.is_initialized()


class FingerForceController:
    """
    Hybrid controller: position control when no contact, force control when contact detected.

    Uses finger tactile sensor to command gripper. When finger force is below threshold,
    uses position from hand tracking. When contact is detected, switches to force control.
    """

    def __init__(
        self,
        finger_sensor,  # TactileSensor or TactileSensorPair
        force_scale: float = 1.0,
        force_offset: float = 0.0,
        max_force: float = 40.0,
        contact_threshold: float = 5.0,
        close_speed: float = 0.02,
        alpha: float = 0.3,
    ):
        """
        Initialize finger force controller.

        Args:
            finger_sensor: Single TactileSensor or TactileSensorPair
            force_scale: Scale factor from sensor force to gripper force
            force_offset: Offset added to gripper force command
            max_force: Maximum gripper force (N)
            contact_threshold: Force threshold to detect contact and switch to force mode
            close_speed: How fast to close gripper in force mode (per update)
            alpha: Smoothing factor for force output
        """
        self.finger_sensor = finger_sensor
        self.force_scale = force_scale
        self.force_offset = force_offset
        self.max_force = max_force
        self.contact_threshold = contact_threshold
        self.close_speed = close_speed
        self.alpha = alpha

        # State
        self.smoothed_force = 0.0
        self.current_width = 1.0  # Start fully open
        self.in_contact = False

    def update(self, position_estimate: float) -> Tuple[float, float, bool]:
        """
        Compute gripper width and force using hybrid control.

        Args:
            position_estimate: Position from hand tracking (0=closed, 1=open)

        Returns:
            Tuple of (width, force, in_contact):
            - width: Target gripper width (0=closed, 1=open)
            - force: Target gripper force (N)
            - in_contact: Whether contact is detected
        """
        # Get force from finger sensor
        finger_force = self.finger_sensor.get_total_force()

        # Map to target gripper force
        target_force = self.force_scale * finger_force + self.force_offset
        target_force = float(np.clip(target_force, 0.0, self.max_force))

        # Smooth force
        self.smoothed_force = self.alpha * target_force + (1.0 - self.alpha) * self.smoothed_force

        # Detect contact
        self.in_contact = finger_force > self.contact_threshold

        if self.in_contact:
            # Force mode: close gripper incrementally until force is achieved
            # The gripper will close and apply the commanded force
            self.current_width = max(0.0, self.current_width - self.close_speed)
            width = self.current_width
            force = self.smoothed_force
        else:
            # Position mode: follow hand position, use low default force
            width = position_estimate
            self.current_width = position_estimate  # Track position for smooth transition
            force = 5.0  # Low default force for position mode

        return width, force, self.in_contact

    def get_debug_info(self) -> dict:
        """Get debug information about controller state."""
        total_force = self.finger_sensor.get_total_force()
        return {
            "finger_force": total_force,
            "gripper_force_cmd": self.smoothed_force,
            "in_contact": self.in_contact,
            "current_width": self.current_width,
        }