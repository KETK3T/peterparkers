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
    time.sleep(0.5)
    # IMU readings
    ax, ay, az = myIMU.accel
    gyro_x, gyro_y, gyro_z = myIMU.gyro

    # Compute dt using timestamps
    now = time.time()
    dt = now - prev_time
    prev_time = now

    ### Prediction step -- propagates uncertainty forward in time
    # State prediction
    ekf.x = ekf.predict_f(ekf.x, dt, ax, ay, gyro_z)
    # Finds the jacobian of the motion model to determine F matrix
    F = ekf.F_jacobian(ekf.x, dt, ax, ay)
    # Covariance prediction
    ekf.P = F @ ekf.P @ F.T + ekf.Q

    ### Update step
    # currently the loop is always running, but the GPS has a refresh rate of ~5Hz, need to research way to only perform
    # update step when GPS receives new data
    gps_x = myGPS.get_lat()
    gps_y = myGPS.get_long()

    # Finds H matrix by calculating the Jacobian of the measurement model
    H = ekf.H_jacobian(ekf.x)
    # Grabs estimate from state vector
    z_pred = ekf.H(ekf.x)
    # Compute the innovation (measurement residual)
    y = np.array([gps_x, gps_y]) - z_pred

    # Innovation covariance
    S = H @ ekf.P @ H.T + ekf.R
    # Kalman gain
    K = ekf.P @ H.T @ np.linalg.inv(S)

    # Updates the state estimate
    ekf.x = ekf.x + K @ y
    # Updates covariance
    ekf.P = (np.eye(5) - K @ H) @ ekf.P

    # Prints state vector
    # position in x and y directions
    # velocity in x and y directions
    # heading in radians
    print(ekf)
