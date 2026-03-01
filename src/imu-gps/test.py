from imu import IMU
from gps import GPS
import time
from localization_test import Localization


#myimu = IMU()
#mygps = GPS()
mylocal = Localization()

while True:
    state = mylocal.getstate()
    print(state)

    state_dict = {
        "latitude": state[0],
        "longitude": state[1],
        "velocity_x": state[2],
        "velocity_y": state[3],
        "heading": state[4]
    }
    time.sleep(0.5)

'''
state = mylocal.getstate()
print(state)

state_dict = {
     "latitude": state[0],
     "longitude": state[1],
     "velocity_x": state[2],
     "velocity_y": state[3],
     "heading": state[4]
}
'''
#print(myimu)
#print(mygps)
#print(mylocal.ekf)

'''
while True:
    print(mylocal.myGPS)
    print(mylocal.myIMU)
    print(mylocal.ekf)
    time.sleep(0.5)
'''

