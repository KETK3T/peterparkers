import numpy as np
from imu import IMU
from gps import GPS
from test_ekf import EKF


def blend_angle(ekf, old_angle, new_angle, alpha):
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
    norm_m = np.sqrt(mx**2 + my**2 + mz**2)

    if norm_a < 1e-6 or norm_m < 1e-6:
        return None

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
