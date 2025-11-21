import time
import numpy as np
from ekf import EKF
from imu import IMU
from gps import GPS

ekf = EKF()
prev_time = time.time()
myIMU = IMU()
myGPS = GPS()

while True:
    # IMU readings
    # ax, ay, gyro_z = read_imu()
    ax, ay, az = myIMU.accel
    gyro_x, gyro_y, gyro_z = myIMU.gyro

    # Compute dt
    now = time.time()
    dt = now - prev_time
    prev_time = now

    # -------- Prediction step --------
    ekf.x = ekf.predict_f(ekf.x, dt, ax, ay, gyro_z)
    F = ekf.F_jacobian(ekf.x, dt, ax, ay)
    ekf.P = F @ ekf.P @ F.T + ekf.Q

    # -------- GPS Update step --------
    gps_x = myGPS.get_lat()
    gps_y = myGPS.get_long()

    H = ekf.H_jacobian(ekf.x)
    z_pred = ekf.H(ekf.x)
    y = np.array([gps_x, gps_y]) - z_pred

    S = H @ ekf.P @ H.T + ekf.R
    K = ekf.P @ H.T @ np.linalg.inv(S)

    ekf.x = ekf.x + K @ y
    ekf.P = (np.eye(5) - K @ H) @ ekf.P

    print("Estimated state:", ekf.x)
