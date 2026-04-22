from flask import Flask, Response
import json
import time
import os
import sys

sensor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../imu-gps'))
sys.path.append(sensor_path)

# Import your localization components
from localization import (
    IMU, GPS, EKF,
    latlon_toxy,
    tilt_compensated_yaw, blend_angle,
)
import numpy as np

app = Flask(__name__)

def sensor_stream():
    yaw_offset = 0.0
    yaw_offset_initialized = False

    prev_gps_x = None
    prev_gps_y = None
    prev_gps_time = None

    ekf = EKF()
    myIMU = IMU()
    myGPS = GPS()

    prev_time = time.time()

    # Initialize GPS origin
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
        u = np.array([[a_forward], [omega]])

        # EKF prediction
        ekf.predict(u, dt)

        # GPS update
        if current_time - last_gps_time > 0.2:
            gps_x, gps_y = latlon_toxy(lat, lon, lat_init, lon_init)
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
                gps_dt = current_time - prev_gps_time

                # Only calibrates when GPS movement is big enough to give a meaningful heading, when the user is traveling roughly straight, and updates slowly so noise does not jerk the heading around
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
        if current_time - last_mag_time >= 0.01:
            mag_yaw = tilt_compensated_yaw(accel, mag)  # THIS IS IN RADIANS
            mag_yaw = ekf.normalize_angle(mag_yaw + np.pi + yaw_offset)
            ekf.update_yaw(mag_yaw)
            last_mag_time = current_time

        # This currently commented out to see if the clamp is causing the zero-velocity issue for output
        # velocity clamp, essentially if the filter believes the user is barely moving, it forces speed down to 0
        if abs(ekf.x[2, 0]) < 0.02:
            ekf.x[2, 0] = 0

        # Extract state
        x, y, v, mag_yaw = ekf.x.flatten()
        print(f"x={x:.2f}, y={y:.2f}, v={v:.2f}, mag_yaw={np.degrees(mag_yaw):.1f} rad")

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

        time.sleep(0.0008)  # Sleep to prevent busy loop, adjust as needed for IMU rate (562 Hz?)


@app.route('/stream')
def stream():
    return Response(sensor_stream(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True, threaded=True, use_reloader=False)