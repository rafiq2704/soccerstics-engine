"""Constant-velocity Kalman filter for ball tracking.

The ball is small, fast, and often hidden behind the kicking leg during
follow-through. When a detection is missing we predict the position so the
track stays continuous; those frames are flagged so the dashboard can show
them differently (amber, dashed) and the physics can down-weight them.

State vector: [x, y, vx, vy]  (pixels, pixels/frame)
"""
import numpy as np


class BallKalman:
    def __init__(self, dt=1.0, process_var=1.0, meas_var=4.0):
        self.dt = dt
        # State transition: x' = x + vx*dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]], dtype=float)
        # We only measure position (x, y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * process_var          # process noise
        self.R = np.eye(2) * meas_var             # measurement noise
        self.P = np.eye(4) * 500.0                # initial uncertainty
        self.x = None                             # unset until first detection

    def _init_state(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 500.0

    def step(self, z):
        """Advance one frame. z = (x, y) detection or None if occluded.

        Returns (pos_xy, predicted_bool).
        """
        if self.x is None:
            if z is None:
                return None, False
            self._init_state(z)
            return (float(self.x[0]), float(self.x[1])), False

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        if z is None:
            # No measurement — return the prediction, flagged
            return (float(self.x[0]), float(self.x[1])), True

        # Update with the measurement
        z = np.array(z, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return (float(self.x[0]), float(self.x[1])), False
