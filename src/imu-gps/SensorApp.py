# App libraries
from flask import Flask, Response
import json
import time

# Import your localization components
# Localization libraries
from localization import (
    IMU, GPS, EKF,
    latlon_toxy,
    tilt_compensated_yaw, blend_angle,
)
import numpy as np

# CSV libraries (for data collection and testing purposes)
import csv
from datetime import datetime
import os

app = Flask(__name__)

def sensor_stream():
    # For testing purposes
    # This creates a CSV file that will store the raw data and the EKF predictions for each loop iteration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('output', exist_ok=True)
    path = f"output/output_{timestamp}.csv"
    count = 1

    with open(path, "w", newline="", encoding="utf-8") as f:
        cWriter = csv.writer(f)
        cWriter.writerow(["id", "latitude", "longitude", "speed", "heading",
                          "gps-lat", "gps-long", "dt", "imu-ax", "imu-ay", "imu-az",
                          "imu-gx", "imu-gy", "imu-gz", "imu-mx", "imu-my", "imu-mz", ])

    print(f"Saving to: {path}")

    yaw_offset = 0.0    # Yaw = heading
    yaw_offset_initialized = False

    prev_gps_x = None
    prev_gps_y = None
    prev_gps_time = None

    ekf = EKF()
    myIMU = IMU()
    myGPS = GPS()

    prev_time = time.time()

    # Initialize GPS origin
    # Ensures that the GPS has locked onto a point before initializing the origin
    # Prevents false points like (0,0)
    while True:
        lat_init, lon_init = myGPS.get_lat(), myGPS.get_long()
        if lat_init != 0 and lon_init != 0:
            break
        time.sleep(0.1)

    last_gps_time = 0
    last_mag_time = 0

    while True:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        dt = min(dt, 0.02)  # Cap dt to 20 ms to prevent large jumps (dt clamp)

        # This is to ensure that consistent values are used for all updates within a single iteration of the loop
        accel = myIMU.get_accel()
        gyro = myIMU.get_gyro()
        mag = myIMU.get_magn()
        lat = myGPS.get_lat()
        lon = myGPS.get_long()

        # Project acceleration onto heading direction
        # The logic here is that the car is either traveling forwards or backwards, even while turning it is still technically
        # maintaining acceleration in the direction that the x-direction of the IMU is facing. The car does not travel sideways,
        # so we do not need accel[1]
        a_forward = accel[0]
        a_forward = np.clip(a_forward, -5.0, 5.0)   # Prevents bad numbers from hurting the EKF predictions
        omega = gyro[2]
        omega = np.clip(omega, -3.0, 3.0)   # Prevents bad numbers from hurting the EKF predictions
        u = np.array([[a_forward], [omega]])

        # EKF prediction
        ekf.predict(u, dt)

        # GPS update
        if current_time - last_gps_time > 0.2:  # GPS has an ODR of 5 Hz
            gps_x, gps_y = latlon_toxy(lat, lon, lat_init, lon_init)    # Converts lat and long into Cartesian coordinates b/c IMU measures in meters and typical GPS coords are angular

            # GPS gate to prevent bad numbers from hurting EKF predictions
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
                gps_dt = current_time - prev_gps_time

                # Only calibrates when GPS movement is big enough to give a meaningful heading, when the user is
                # traveling roughly straight, and updates slowly so noise does not jerk the heading around
                if gps_dt > 0 and dist > 3.0 and abs(gyro[2]) < 0.2:
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
            prev_gps_time = current_time
            last_gps_time = current_time

        # Magnetometer update
        if current_time - last_mag_time >= 0.01:    # Magnetometer has an ODR of 100 Hz
            mag_yaw = tilt_compensated_yaw(accel, mag)  # THIS IS IN RADIANS
            mag_yaw = ekf.normalize_angle(mag_yaw + np.pi + yaw_offset)
            ekf.update_yaw(mag_yaw)
            last_mag_time = current_time

        # velocity clamp, essentially if the filter believes the user is barely moving, it forces speed down to 0
        if abs(ekf.x[2, 0]) < 0.02:
            ekf.x[2, 0] = 0

        # Extract state
        x, y, v, mag_yaw = ekf.x.flatten()
        print(f"x={x:.2f}, y={y:.2f}, v={v:.2f}, mag_yaw={np.degrees(mag_yaw):.1f} degrees")    # Converts heading to degrees for easier comprehension

        # Saves state in testing output CSV file
        with open(path, "a", newline="", encoding="utf-8") as f:
            cWriter = csv.writer(f)
            cWriter.writerow([count, x, y, v, mag_yaw, lat, lon, dt,
                              accel[0], accel[1], accel[2],
                              gyro[0], gyro[1], gyro[2],
                              mag[0], mag[1], mag[2]])
            count += 1

        data = {
            "x": float(x),
            "y": float(y),
            "v": float(v),
            "mag_yaw": float(mag_yaw),
            "lat": float(lat),
            "lon": float(lon)
        }

        # SSE format
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(0.01)  # Sleep to prevent busy loop, adjusted to 0.01 to match the fastest ODR (magnetometer ODR 100Hz)


@app.route('/stream')
def stream():
    return Response(sensor_stream(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)