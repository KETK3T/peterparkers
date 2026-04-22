from flask import Flask, Response
import json
import time

# Import your localization components
from localization import (
    IMU, GPS, EKF,
    latlon_toxy,
    tilt_compensated_yaw, blend_angle,
    prev_gps_x, prev_gps_y, prev_gps_time,
    yaw_offset, yaw_offset_initialized
)
import numpy as np

app = Flask(__name__)

def sensor_stream():
    ekf = EKF()
    imu = IMU()
    gps = GPS()

    prev_time = time.time()

    # Initialize GPS origin
    while True:
        lat_init, lon_init = gps.get_lat(), gps.get_long()
        if lat_init != 0 and lon_init != 0:
            break
        time.sleep(0.1)

    last_gps_time = 0
    last_mag_time = 0

    while True:
        current_time = time.time()
        dt = min(current_time - prev_time, 0.02) # Cap dt to 20 ms to prevent large jumps (dt clamp)
        prev_time = current_time

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
        if current_time - last_mag_time > 0.01:
            mag_yaw = tilt_compensated_yaw(accel, mag)
            ekf.update_yaw(mag_yaw)
            last_mag_time = current_time

        # Extract state
        x, y, v, mag_yaw = ekf.x.flatten()

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

        time.sleep(0.01)  # ~100 Hz


@app.route('/stream')
def stream():
    return Response(sensor_stream(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True, threaded=True)