from flask import Flask, Response
import json
import time
import sys
import csv

# Import your localization components
from localization_simulation import (
    SimImu, SimGps, EKF,
    latlon_toxy,
    tilt_compensated_yaw, blend_angle,
)
import numpy as np

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

def sensor_stream():
    yaw_offset = 0.0
    yaw_offset_initialized = False

    prev_gps_x = 0
    prev_gps_y = 0
    prev_gps_time = 0

    # This is going get csv file from the command line
    if len(sys.argv) < 2:
        print("Error: Need 1 output csv file")
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r") as f:
        reader = csv.DictReader(f)  # uses header row automatically
        data = list(reader)  # convert to list so we can index rows

    ekf = EKF()
    imu = SimImu(0,0,0,0,0,0,0,0,0)
    gps = SimGps(0,0)

    data_idx = 0  # refering to data in data = list(reader), used to read rows one by one for gps and imu data

    # Initial GPS reading to set the origin of the state vector
    # EKF expects a linear Cartesian system...so coordinates will be converted to Cardinal measurements
    lat_init = 0
    long_init = 0

    while True:
        # Reads one row per iteration
        row = data[data_idx]

        # Simulating retrieving data from the sensors
        dt = float(row["dt"])
        dt = min(dt, 0.02)  # Cap dt to 20 ms to prevent large jumps (dt clamp)

        imu.accel = float(row["imu-ax"]), float(row["imu-ay"]), float(row["imu-az"])
        imu.gyro = float(row["imu-gz"]), float(row["imu-gy"]), float(row["imu-gz"])
        imu.mag = float(row["imu-mx"]), float(row["imu-my"]), float(row["imu-mz"])

        gps.latitude = float(row["gps-lat"])
        gps.longitude = float(row["gps-long"])

        # This is to ensure that consistent values are used for all of the updates within a single iteration of the loop
        accel = imu.get_accel()
        gyro = imu.get_gyro()
        mag = imu.get_magn()
        lat = gps.get_lat()
        lon = gps.get_long()

        # Project acceleration onto heading direction
        a_forward = accel[0]
        a_forward = np.clip(a_forward, -5.0, 5.0)
        omega = gyro[2]
        omega = np.clip(omega, -3.0, 3.0)
        u = np.array([[a_forward], [omega]])

        # EKF prediction
        ekf.predict(u, dt)

        # Updates GPS whenever there is new data
        # if current_time - last_gps_time > 0.2:
        gps_x, gps_y = latlon_toxy(lat, lon, lat_init, long_init)

        # GPS gate
        dist_to_state = np.linalg.norm([
            gps_x - ekf.x[0, 0],
            gps_y - ekf.x[1, 0]
        ])

        if dist_to_state < 25.0:
            ekf.update_gps(np.array([gps_x, gps_y]))

        # Auto-calibrate yaw offset using GPS heading
        # if prev_gps_x is not None and pre_gps_time is not None:
        dx = gps_x - prev_gps_x
        dy = gps_y - prev_gps_y
        dist = np.hypot(dx, dy)
        # gps_dt = current_time - prev_gps_time

        # Only calibrates when GPS movement is big enough to give a meaningful heading, when the user is traveling roughly straight, and updates slowly so noise does not jerk the heading around
        if dist > 3.0 and abs(gyro[2]) < 0.2:
            gps_heading = np.arctan2(dy, dx)
            mag_yaw_raw = tilt_compensated_yaw(accel, mag)

            candidate_offset = ekf.normalize_angle(gps_heading - mag_yaw_raw)

            if not yaw_offset_initialized:
                yaw_offset = candidate_offset
                yaw_offset_initialized = True
            else:
                yaw_offset = blend_angle(ekf, yaw_offset, candidate_offset, alpha=0.05)

        prev_gps_x = gps_x
        prev_gps_y = gps_y

        # Updates yaw whenever there is new data from the magnetometer
        # if current_time - last_mag_time >= 0.01:
        mag_yaw = tilt_compensated_yaw(accel, mag)  # THIS IS IN RADIANS
        mag_yaw = ekf.normalize_angle(mag_yaw + np.pi + yaw_offset)
        ekf.update_yaw(mag_yaw)

        # This currently commented out to see if the clamp is causing the zero-velocity issue for output
        # velocity clamp, essentially if the filter believes the user is barely moving, it forces speed down to 0
        if abs(ekf.x[2, 0]) < 0.02:
            ekf.x[2, 0] = 0

        # Print and save state
        x, y, v, mag_yaw = ekf.x.flatten()
        #print(f"x={x:.2f}, y={y:.2f}, v={v:.2f}, mag_yaw={np.degrees(mag_yaw):.1f} rad")

        info = {
            "x": float(x),
            "y": float(y),
            "v": float(v),
            "mag_yaw": float(mag_yaw),
            "lat": float(lat),
            "lon": float(lon)
        }

        # SSE format
        yield f"data: {json.dumps(info)}\n\n"

        data_idx += 1  # Moves to next row in csv file
        if data_idx >= len(data):
            data_idx = 0  # loop again instead of breaking
            #print("Done!")
            #break
        time.sleep(0.01)  # Sleep to prevent busy loop, adjust as needed for IMU rate (562 Hz?)


@app.route('/stream')
def stream():
    return Response(sensor_stream(), mimetype='text/event-stream',
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)