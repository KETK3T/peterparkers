# pip install filterpy
import math
import filterpy.kalman as kf
from fontTools.ttLib.sfnt import sfntDirectoryFormat

### IMU data needed
# ax, ay, az = accel_x, accel_y, accel_z
# mx, my, mz = mag_x, mag_y, mag_z

# dim_x is the number of state variables
    # position in x and y directions
    # velocity in x and y directions
    # heading
# dim_z is the number of measurement variables
    # GPS position in x and y direction
ekf = kf.ExtendedKalmanFilter(5, 2)
ekf.x = []

while (True):
    ekf.predict_update()

def calc_head(ax, ay, az, mx, my, mz):
    # Calculating roll and pitch
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    # Tilt compensation because IMU might not stay flat
    mx2 = mx * math.cos(pitch) + mz * math.sin(pitch)
    my2 = mx * math.sin(roll) * math.sin(pitch) + my * math.cos(roll) - mz * math.sin(roll) * math.cos(pitch)

    # Determining heading
    # Verify by testing pointing north, signs may need to change if orientation is wrong
    yaw_mag = math.atan2(my2, mx2) # returns radians, range [-pi, pi]
    return yaw_mag