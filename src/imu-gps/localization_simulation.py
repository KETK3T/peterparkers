# To run
# Be in peterparkers directory
# python ./src/imu-gps/localization_simulation.py output_for_sim/[filename].csv

import time

import numpy as np
#from ekf import EKF
from test_ekf import EKF
import csv
from datetime import datetime
import os
import sys

class SimGps:
    def __init__(self, lat, lon):
        self.latitude = lat
        self.longitude = lon

    def get_lat(self):
        return self.latitude

    def get_long(self):
        return self.longitude

class SimImu:
    def __init__(self, ax, ay, az, gx, gy, gz, mx, my, mz):
        self.accel = ax, ay, az
        self.gyro = gx, gy, gz
        self.magnetic = mx, my, mz

    def get_accel(self):
        return self.accel

    def get_gyro(self):
        return self.gyro

    def get_magn(self):
        return self.magnetic



yaw_offset = 0.0
yaw_offset_initialized = False

prev_gps_x = None
prev_gps_y = None
prev_gps_time = None

def blend_angle(old_angle, new_angle, alpha):
    diff = ekf.normalize_angle(new_angle - old_angle)
    return ekf.normalize_angle(old_angle + alpha * diff)

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

def choose_gps_origin(data, min_valid_samples=5, stability_radius_m=5.0):
    valid_points = []

    for row in data:
        lat = float(row["gps-lat"])
        lon = float(row["gps-long"])

        # Filters out invalid GPS points, such as (0,0)
        if lat == 0 and lon == 0:
            continue

        if not (-90.0 <= lat <= 90.0 and -180.0 <- lon <= 180.0):
            continue

        valid_points.append((lat, lon))

        if len(valid_points) >= min_valid_samples:
            # Check stability
            lat0, lon0 = valid_points[0]
            
            stable = True
            for lat_i, lon_i in valid_points:
                x, y = latlon_toxy(lat_i, lon_i, lat0, lon0)
                if np.hypot(x, y) > stability_radius_m:
                    stable = False
                    break

            if stable:
                lat_avg = np.mean([p[0] for p in valid_points])
                lon_avg = np.mean([p[1] for p in valid_points])
                return lat_avg, lon_avg

            valid_points.pop(0)

    raise ValueError("Could not find a stable initial GPS fix in the CSV.")


# For creating test data
# Creates a new CSV file and prints headers, file closes before while loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('output', exist_ok=True)
path = f"output/sim_output_{timestamp}.csv" # Creates new csv file titled ouput_YMD_HMS.csv
count = 1 # csv row id

with open(path, "w", newline="", encoding="utf-8") as f:
    cWriter = csv.writer(f)
    cWriter.writerow(["id", "latitude", "longitude", "speed", "heading",
                      "gps-lat", "gps-long", "dt", "imu-ax", "imu-ay", "imu-az",
                      "imu-gx","imu-gy","imu-gz", "imu-mx", "imu-my", "imu-mz",])

print(f"Saving to: {path}")

# This is going get csv file from the command line
if len(sys.argv) < 2:
    print("Error: Need 1 output csv file")
    sys.exit(1)

input_file = sys.argv[1]

with open(input_file, "r") as f:
    reader = csv.DictReader(f)   # uses header row automatically
    data = list(reader)  # convert to list so we can index rows

ekf = EKF()
myIMU = SimImu(0,0,0,0,0,0,0,0,0)
myGPS = SimGps(0,0)

data_idx = 0 #refering to data in data = list(reader), used to read rows one by one for gps and imu data

# Initial GPS reading to set the origin of the state vector
# EKF expects a linear Cartesian system...so coordinates will be converted to Cardinal measurements
lat_init, long_init = choose_gps_origin(data)
print(f"GPS origin set to: {lat_init}, {long_init}")

while data_idx < len(data):
    #Reads one row per iteration
    row = data[data_idx]

    # Simulating retrieving data from the sensors
    dt = float(row["dt"])
    dt = min(dt, 0.02)  # Cap dt to 20 ms to prevent large jumps (dt clamp)

    myIMU.accel = float(row["imu-ax"]), float(row["imu-ay"]), float(row["imu-az"])
    myIMU.gyro = float(row["imu-gx"]), float(row["imu-gy"]), float(row["imu-gz"])
    myIMU.magnetic = float(row["imu-mx"]), float(row["imu-my"]), float(row["imu-mz"])

    myGPS.latitude = float(row["gps-lat"])
    myGPS.longitude = float(row["gps-long"])

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
    #if current_time - last_gps_time > 0.2:
    gps_x, gps_y = latlon_toxy(lat, lon, lat_init, long_init)

    # GPS gate
    dist_to_state = np.linalg.norm([
        gps_x - ekf.x[0, 0],
        gps_y - ekf.x[1, 0]
    ])

    if dist_to_state < 25.0:
        ekf.update_gps(np.array([gps_x, gps_y]))

    # Auto-calibrate yaw offset using GPS heading
    if prev_gps_x is not None and prev_gps_time is not None:
        dx = gps_x - prev_gps_x
        dy = gps_y - prev_gps_y
        dist = np.hypot(dx, dy)
    #gps_dt = current_time - prev_gps_time

    # Only calibrates when GPS movement is big enough to give a meaningful heading, when the user is traveling roughly straight, and updates slowly so noise does not jerk the heading around
        if dist > 3.0 and abs(gyro[2]) < 0.2:
            gps_heading = np.arctan2(dy, dx)
            mag_yaw_raw = tilt_compensated_yaw(accel, mag)

            candidate_offset = ekf.normalize_angle(gps_heading - mag_yaw_raw)

            if not yaw_offset_initialized:
                yaw_offset = candidate_offset
                yaw_offset_initialized = True
            else:
                yaw_offset = blend_angle(yaw_offset, candidate_offset, alpha=0.05)

    prev_gps_x = gps_x
    prev_gps_y = gps_y
    prev_gps_time = data_idx

    # Updates yaw whenever there is new data from the magnetometer
    #if current_time - last_mag_time >= 0.01:
    mag_yaw = tilt_compensated_yaw(accel, mag)  # THIS IS IN RADIANS
    mag_yaw = ekf.normalize_angle(mag_yaw + yaw_offset)  #TODO: YAW_OFFSET is not referenced
    ekf.update_yaw(mag_yaw)


    # This currently commented out to see if the clamp is causing the zero-velocity issue for output
    # velocity clamp, essentially if the filter believes the user is barely moving, it forces speed down to 0
    if abs(ekf.x[2, 0]) < 0.02:
        ekf.x[2, 0] = 0

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

    data_idx += 1 #Moves to next row in csv file
    time.sleep(0.0008)  # Sleep to prevent busy loop, adjust as needed for IMU rate (562 Hz?)



