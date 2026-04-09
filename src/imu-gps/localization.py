import time
import numpy as np
#from ekf import EKF
from imu import IMU
from gps import GPS
from test_ekf import EKF
import csv
from datetime import datetime
import os

ekf = EKF()
prev_time = time.time()
myIMU = IMU()
myGPS = GPS()
tot_time = 0

# This function converts latitude and longitude to local Cartesian coordinates (x, y) in meters relative to an origin point (lat0, lon0).
def latlon_toxy(lat, lon, lat0, lon0):
    R = 6378137  # Earth radius in meters

    lat = np.radians(lat)
    lon = np.radians(lon)

    x = (lon - lon0) * R * np.cos(lat0)
    y = (lat - lat0) * R

    return x, y

def tilt_compensated_yaw(accel, mag):
    ax, ay, az = accel
    mx, my, mz = mag

    # Normalize accelerometer
    norm_a = np.sqrt(ax**2 + ay**2 + az**2)
    ax /= norm_a
    ay /= norm_a
    az /= norm_a

    # Pitch and roll
    pitch = np.arcsin(-ax)  # rotation around y-axis
    roll = np.arctan2(ay, az) # rotation around x-axis

    # Tilt compensation for magnetometer
    mx_comp = mx * np.cos(pitch) + mz * np.sin(pitch)
    my_comp = mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll) - mz * np.sin(roll) * np.cos(pitch)

    # Yaw calculation
    yaw = np.arctan2(-my_comp, mx_comp)  # rotation around z-axis, negative sign depends on sensor frame

    return yaw


# For creating test data
# Creates a new CSV file and prints headers, file closes before while loop

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('output', exist_ok=True)
path = f"output/output_{timestamp}.csv"
count = 1 # csv row id

with open(path, "w", newline="", encoding="utf-8") as f:
    cWriter = csv.writer(f)
    cWriter.writerow(["id", "latitude", "longitude", "speed", "heading",
                      "gps-lat", "gps-long", "tot-time", "dt", "imu-ax", "imu-ay", "imu-az",
                      "imu-gx","imu-gy","imu-gz",])

print(f"Saving to: {path}")
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
# Initial GPS reading to set the origin of the state vector
# EKF expects a linear Cartesian system...so coordinates will be converted to Cardinal measurements
lat_init, long_init = myGPS.get_lat(), myGPS.get_long()


while True:
    #current_time = time.time()
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    dt = min(dt, 0.02)  # Cap dt to 20 ms to prevent large jumps
    ekf.dt = dt

    ax, ay, _ = myIMU.get_accel()
    psi = ekf.x[3, 0]  # current heading

    # Project acceleration onto heading direction
    a_forward = ax

    omega = myIMU.get_gyro()[2]

    u = np.array([[a_forward],
                  [omega]])
    # Prediction (IMU rate)
    ekf.predict(u)


    accel = myIMU.get_accel()

    if myGPS.new_data:
        mag = myIMU.get_magn()
        mag_yaw = tilt_compensated_yaw(accel, mag)
        ekf.update_yaw(mag_yaw)
    if myIMU.new_data:
        lat, lon = myGPS.get_data()
        x, y = latlon_toxy(lat, lon, lat_init, long_init)
        ekf.update_gps(np.array([x, y]))

    '''
    # need to check if data is available so not wasting resources
    # thoughts: could check if there is difference from the last update, and if there is an update then blah blah blah
    # Magnetometer Update
    if mag_available:
        ekf.update_mag(mag_yaw)

    # GPS update
    if gps_available:
        lat, lon = myGPS.get_lat(), myGPS.get_long()
        x, y = latlon_toxy(lat, lon, lat_init, long_init)
        ekf.update_gps(np.array([x, y]))
    '''

    # Print and save state
    x, y, v, psi = ekf.x.flatten()
    print(f"x={x:.2f}, y={y:.2f}, v={v:.2f}, psi={np.degrees(psi):.1f} deg")

    with open(path, "a", newline="", encoding="utf-8") as f:
        cWriter = csv.writer(f)
        cWriter.writerow([count, x, y, v, psi, myGPS.get_lat(), myGPS.get_long(),tot_time,dt,
                          myIMU.get_accel()[0], myIMU.get_accel()[1], myIMU.get_accel()[2], myIMU.get_gyro()[0],
                          myIMU.get_gyro()[1], myIMU.get_gyro()[2]])
        count += 1
    
    time.sleep(0.0001)  # Sleep to prevent busy loop, adjust as needed for IMU rate (562 Hz?)