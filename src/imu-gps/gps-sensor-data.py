'''
Set up:
enable serial port on jetson
find what serial port points to:
command: ls -l /dev/serial0
(rasp points to ttyS0)
'''

#pip install pynmea2
#pip install pyserial

import serial
import time
import pynmea2

# Configure the serial port
# Replace '/dev/ttyS0' with the correct serial device name for your Pi
# (e.g., '/dev/ttyUSB0' for a USB-to-serial adapter)
# Ensure the baudrate matches the device you are communicating with.
ser = serial.Serial(
    port='/dev/ttyS0',  # Adjust this for your specific setup
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1  # Timeout in seconds
)

print("Serial port opened. Waiting for data...")


try:
    while True:
        if ser.in_waiting > 0:
            #print("Hello")
            # Read data from the serial port
            # ser.readline() reads until a newline character is encountered
            # ser.read(num_bytes) reads a specified number of bytes
            data = ser.readline().decode().strip() # Decode and remove whitespace
            #'utf-8'
            if data:
                if data.find('GGA') > 0:
                    try:
                        msg = pynmea2.parse(data)
                        print(msg.timestamp,'Lat:',round(msg.latitude,6),'Lon:',round(msg.longitude,6),'Alt:',msg.altitude,'Sats:',msg.num_sats)
                        #print(f"Received: {data}")
                    except Exception as e:
                        print(e)
                
            else:
                print("No data recieved")
        time.sleep(0.1) # Small delay to prevent busy-waiting

except KeyboardInterrupt:
    print("Program terminated by user.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    ser.close()
    print("Serial port closed.")
