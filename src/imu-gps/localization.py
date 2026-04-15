import time
import numpy as np
#from ekf import EKF
from imu import IMU
from gps import GPS
from test_ekf import EKF
import csv
from datetime import datetime
import os

# This function converts latitude and longitude to local Cartesian coordinates (x, y) in meters relative to an origin point (lat0, lon0).
def latlon_toxy(lat, lon, lat0, lon0):  # lat0 and lon0 are the initial starting GPS coordinates
    R = 6378137  # Earth radius in meters

    lat = np.radians(lat)
    lon = np.radians(lon)

    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    x = (lon - lon0) * R * np.cos(lat0)
    y = (lat - lat0) * R

    return x, y

# This function calculates the yaw (heading) of the car. It uses pitch and roll to compensate for possible tilted angle of the IMU.
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
path = f"output/output_{timestamp}.csv" # Creates new csv file titled ouput_YMD_HMS.csv
count = 1 # csv row id

with open(path, "w", newline="", encoding="utf-8") as f:
    cWriter = csv.writer(f)
    cWriter.writerow(["id", "latitude", "longitude", "speed", "heading",
                      "gps-lat", "gps-long", "dt", "imu-ax", "imu-ay", "imu-az",
                      "imu-gx","imu-gy","imu-gz", "imu-mx", "imu-my", "imu-mz",])

print(f"Saving to: {path}")

ekf = EKF()
myIMU = IMU()
myGPS = GPS()
prev_time = time.time()

# Initial GPS reading to set the origin of the state vector
# EKF expects a linear Cartesian system...so coordinates will be converted to Cardinal measurements
lat_init = 0
long_init = 0

# This takes into account startup time for the GPS.
while True:
    lat_init, long_init = myGPS.get_lat(), myGPS.get_long()

    if lat_init != 0 and long_init != 0:
        break

    time.sleep(0.1)

last_gps_time = 0
last_mag_time = 0

# Main loop, it should be running as fast as the fastest sensor output
while True:
    #current_time = time.time()
    current_time = time.time()
    dt = current_time - prev_time
    prev_time = current_time

    dt = min(dt, 0.02)  # Cap dt to 20 ms to prevent large jumps (dt clamp)

    # This is to ensure that consistent values are used for all of the updates within a single iteration of the loop
    accel = myIMU.get_accel()
    gyro = myIMU.get_gyro()
    mag = myIMU.get_magn()
    lat = myGPS.get_lat()
    lon = myGPS.get_long()

    # Project acceleration onto heading direction
    a_forward = accel[0]
    a_forward = np.clip(a_forward, -5.0, 5.0)
    omega = gyro[2]
    omega = np.clip(omega, -3.0, 3.0)
    u = np.array([[a_forward],
                  [omega]])

    # Prediction (IMU rate)
    ekf.predict(u, dt)

    # Updates GPS whenever there is new data
    if current_time - last_gps_time > 0.2:
        x, y = latlon_toxy(lat, lon, lat_init, long_init)
        ekf.update_gps(np.array([x, y]))
        last_gps_time = current_time

    # Updates yaw whenever there is new data from the magnetometer
    if current_time - last_mag_time >= 0.01:
        mag_yaw = tilt_compensated_yaw(accel, mag)  # THIS IS IN RADIANS
        ekf.update_yaw(mag_yaw)
        last_mag_time = current_time

    # velocity clamp, essentially if the filter believes the user is barely moving, it forces speed down to 0
    if abs(ekf.x[2, 0]) < 0.2:
        ekf.x[2,0] = 0

    # Print and save state
    x, y, v, mag_yaw = ekf.x.flatten()
    print(f"x={x:.2f}, y={y:.2f}, v={v:.2f}, mag_yaw={np.degrees(mag_yaw):.1f} rad")

    with open(path, "a", newline="", encoding="utf-8") as f:
        cWriter = csv.writer(f)
        cWriter.writerow([count, x, y, v, mag_yaw, lat, lon, dt,
                          accel[0], accel[1], accel[2],
                          gyro[0], gyro[1], gyro[2],
                          mag[0], mag[1], mag[2]])
        count += 1
    
    time.sleep(0.0008)  # Sleep to prevent busy loop, adjust as needed for IMU rate (562 Hz?)