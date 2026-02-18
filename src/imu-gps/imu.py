import threading
import board
import adafruit_icm20x

#if need ax,ay,az (or other measurements)=> 
#myimu = IMU()
#ax, ay, az = myimu.get_accel()
#print(myimu) => should be used in a while True loop
#once application closes make sure run myimu.stop() for clean stop

class IMU:
    def __init__(self):
        
        i2c = board.I2C()
        self.sensor = adafruit_icm20x.ICM20948(i2c)
        #self.rate = rate_hz
        self.running = True
        
        self.accel = self.sensor.acceleration# m/s^2
        self.gyro =  self.sensor.gyro #rad/s
        self.magnetic = self.sensor.magnetic # uT

        # start background thread
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()
    
    def __str__(self):
        return (
            f"Acceleration: X:{self.accel[0]:.3f}, Y:{self.accel[1]:.3f}, Z:{self.accel[2]:.3f} m/s^2\n"
            f"Gyroscope: X:{self.gyro[0]:.3f}, Y:{self.gyro[1]:.3f}, Z:{self.gyro[2]:.3f} rads/s\n"
            f"Magnetometer: X:{self.magnetic[0]:.3f}, Y:{self.magnetic[1]:.3f}, Z:{self.magnetic[2]:.3f} uT\n"
        )

    def update_loop(self):
        #period = 1 / self.rate
        while self.running:
            self.accel = self.sensor.acceleration
            self.gyro = self.sensor.gyro
            self.magnetic = self.sensor.magnetic
            #time.sleep(period)

    def get_accel(self):
        return self.accel
    
    def get_gyro(self):
        return self.gyro

    def get_magn(self):
        return self.magnetic

    def stop(self):
        self.running = False
        self.thread.join()
