from imu import IMU
from gps import GPS
import time

myimu = IMU()
mygps = GPS()

while True: #while loop acts as the application loop
    print(myimu) #subsitute for sending data
    print(mygps)
    time.sleep(0.5)
