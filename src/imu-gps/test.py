from imu import IMU
import time

myimu = IMU()

while True: #while loop acts as the application loop
    print(myimu) #subsitute for sending data
    time.sleep(0.5)
