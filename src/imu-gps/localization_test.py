import time
import numpy as np
from ekf import EKF
from imu import IMU
from gps import GPS
import threading

class Localization:
    def __init__(self):
        self.ekf = EKF()
        self.prev_time = time.time()
        self.myIMU = IMU()
        self.myGPS = GPS()

        self.state = None
        self.running = False
        self.lock = threading.Lock()
        self.start() #put into a condition func, if sensors are good, then start

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop)
        self.thread.daemon = True
        self.thread.start()

    def _update_loop(self):

        while self.running:
            # IMU readings
            ax, ay, az = self.myIMU.accel
            gyro_x, gyro_y, gyro_z = self.myIMU.gyro

            # Compute dt using timestamps
            now = time.time()
            dt = now - self.prev_time
            self.prev_time = now

            ### Prediction step -- propagates uncertainty forward in time
            # State prediction
            self.ekf.x = self.ekf.predict_f(self.ekf.x, dt, ax, ay, gyro_z)
            # Finds the jacobian of the motion model to determine F matrix
            F = self.ekf.F_jacobian(self.ekf.x, dt, ax, ay)
            # Covariance prediction
            self.ekf.P = F @ self.ekf.P @ F.T + self.ekf.Q

            ### Update step
            # currently the loop is always running, but the GPS has a refresh rate of ~5Hz, need to research way to only perform
            # update step when GPS receives new data
            gps_x = self.myGPS.get_lat()
            gps_y = self.myGPS.get_long()

            # Finds H matrix by calculating the Jacobian of the measurement model
            H = self.ekf.H_jacobian(self.ekf.x)
            # Grabs estimate from state vector
            z_pred = self.ekf.H(self.ekf.x)
            # Compute the innovation (measurement residual)
            y = np.array([gps_x, gps_y]) - z_pred

            # Innovation covariance
            S = H @ self.ekf.P @ H.T + self.ekf.R
            # Kalman gain
            K = self.ekf.P @ H.T @ np.linalg.inv(S)

            # Updates the state estimate
            self.ekf.x = self.ekf.x + K @ y
            # Updates covariance
            self.ekf.P = (np.eye(5) - K @ H) @ self.ekf.P

            # position in x and y directions
            # velocity in x and y directions
            # heading in radians

            with self.lock:
                self.state = self.ekf.x

            # Update if need
            time.sleep(0.5)

    def getstate(self):
        with self.lock:
            return self.state
            #returns self.ekf.x
    
    """
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
    # position in lat and long coordinate
    # velocity in x and y directions
    # heading in radians
    print(ekf)


    #File is reopened each file loop in append mode so that data can be added in a new line
    with open(path, "a", newline="", encoding="utf-8") as f:
        cWriter = csv.writer(f)
        cWriter.writerow([count, ekf.x[0], ekf.x[1], ekf.x[2], ekf.x[3], ekf.x[4], myGPS.get_lat(), myGPS.get_long(),tot_time,dt,
                          myIMU.get_accel()[0], myIMU.get_accel()[1], myIMU.get_accel()[2], myIMU.get_gyro()[0],
                          myIMU.get_gyro()[1], myIMU.get_gyro()[2]])
        count += 1

    tot_time += dt
"""



