import serial
import time
import pynmea2
import threading

class GPS:
    def __init__(self, rate_hz=5):
        self.ser = serial.Serial(
            port='/dev/ttyTHS1',  # Adjust this for your specific setup
            baudrate=9600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1  # Timeout in seconds
        )

        self.rate = rate_hz
        self.running = True

        self.latitude = None
        self.longitude = None
        self.altitude = None

        # start background thread
        self.thread = threading.Thread(target=self.update_loop, daemon=True)
        self.thread.start()

    def __str__(self):
        return (
            f"Latitude: {self.latitude:.3f}\n"
            f"Longitude: {self.longitude:.3f}\n"
            f"Altitude: {self.altitude:.3f}\n"
        )

    def update_loop(self):
        print("Serial port opened. Waiting for data...")
        period = 1 / self.rate
        try:
            while self.running:
                if self.ser.in_waiting > 0:
                    # Read data from the serial port
                    # ser.readline() reads until a newline character is encountered
                    # ser.read(num_bytes) reads a specified number of bytes
                    data = self.ser.readline().decode().strip()  # Decode and remove whitespace
                    # 'utf-8'
                    if data:
                        if data.find('GGA') > 0:
                            try:
                                msg = pynmea2.parse(data)
                                '''
                                print(msg.timestamp, 'Lat:', round(msg.latitude, 6), 'Lon:', round(msg.longitude, 6),
                                      'Alt:', msg.altitude, 'Sats:', msg.num_sats)
                                '''
                                # print(f"Received: {data}")

                                self.latitude = round(msg.latitude, 6)
                                self.longitude = round(msg.longitude, 6)
                                self.altitude = msg.altitude

                            except Exception as e:
                                print(e)

                    else:
                        print("No data recieved")
                time.sleep(period)  # Small delay to prevent busy-waiting

        except Exception as e:
            print(f"An error occurred: {e}")

    def get_lat(self):
        return self.latitude
    
    def get_long(self):
        return self.longitude
    
    def get_altitude(self):
        return self.altitude
    
    def stop(self):
        self.running = False
        self.ser.close()
        self.thread.join()