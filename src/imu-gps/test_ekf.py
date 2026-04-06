import numpy as np

class EKF:
    def __init__(self, dt):
        self.dt = dt
        """
        Define state vector [x, y, v, psi]
        [x-position, y-position, speed (scalar), yaw (heading)]
        Speed used instead of velocity since it can be assumed that a car will be traveling in the forward or backward direction in relation to its
        orientation. For example, a car is not going to travel sideways.
        """
        self.x = np.zeros((4, 1))

        """
        State covariance P matrix
        How uncertain the filter is about each part of the state
        Diagonals contain covariances of each measurement, off-diagonals show uncertainty in correlation between measurements
        So for example, if speed is wrong, then position is probably wrong as well.
        Small values in P mean the filter is more confident, large values in P mean more uncertainty
        """
        self.P = np.diag([10, 10, 1, 0.1])

        """
        Process noise Q matrix
        Represents uncertainty in the model
        Increase if motion is unpredictable or filter is too "confident"
        Slow/unresponsive filter? Increase Q values

        This Q matrix controls acceleration and yaw rate
        """
        self.Q = np.diag([1.0, 0.05])

        """
        Measurement noise R matrix
        Based on sensor specs, increase if measurements are noisy
        Noisy output? Increase R values
        """
        self.R_gps = np.diag([6.25, 6.25])  # GPS variance
        self.R_yaw = np.array([[0.01]])     # magenetometer variance (from IMU)

    def predict(self, u):
        a, omega = u.flatten()
        x, y, v, psi = self.x.flatten()
        dt = self.dt

        # Nonlinear motion model: velocity + heading
        # x_pred = x_prev + v_prev * cos(psi_prev) * dt
        # y_pred = y_prev + v_prev * sin(psi_prev) * dt
        # v_pred = v_prev + a * dt
        # psi_pred = psi_prev + omega * dt
        # where a is forward acceleration from IMU and omega is yaw rate from gyro (also on IMU)
        x_new = x + v * np.cos(psi) * dt
        y_new = y + v * np.sin(psi) * dt
        v_new = v + a * dt
        psi_new = psi + omega * dt

        self.x = np.array([[x_new], [y_new], [v_new], [psi_new]])

        """
        Computing Jacobians
        State Transition Jacobian F captures how turning affects position
        Control Jacobian B captures how acceleration and yaw rate affect state
        """
        F = np.array([
            [1, 0, np.cos(psi) * dt, -v * np.sin(psi) * dt],
            [0, 1, np.sin(psi) * dt, v * np.cos(psi) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        B = np.array([
            [0, 0],
            [0, 0],
            [dt, 0],
            [0, dt]
        ])

        # Predict state covariance
        # P_pred = F * P * F^T + B * Q * B^T
        self.P = F @ self.P @ F.T + B @ self.Q @ B.T

    def update_gps(self, z):
        # z = [x_meas, y_meas]
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        z = z.reshape((2,1))
        y = z - H @ self.x

        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def update_yaw(self, yaw_meas):
        H = np.array([[0, 0, 0, 1]])

        z = np.array([[yaw_meas]])
        y = z - H @ self.x

        y[0,0] = self.normalize_angle(y[0,0])

        S = H @ self.P @ H.T + self.R_yaw
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

        self.x[3,0] = self.normalize_angle(self.x[3,0])

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi)