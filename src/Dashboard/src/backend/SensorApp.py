from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time

import sys
import os


sensor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../imu-gps'))
sys.path.append(sensor_path)

from localization_test import Localization

app = Flask(__name__)
CORS(app)

myLocalization = Localization()


@app.route("/state")
def get_sensor_state():

    state = myLocalization.getstate()

    state_dict = {
        "longitude": float(state[0]),
        "latitude": float(state[1]),
        "velocity_x": float(state[2]),
        "velocity_y": float(state[3]),
        "heading": float(state[4])
    }

    return jsonify(state_dict)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)
    #should show up on localhost:5000/state for testing

'''
Alt code if this doesn't work

myLocalization = Localization()

state_dict = {
        "longitude": float(state[0]),
        "latitude": float(state[1]),
        "velocity_x": float(state[2]),
        "velocity_y": float(state[3]),
        "heading": float(state[4])
}

@app.route("/state")
def get_sensor_state():
    return jsonify(state_dict)
    
def get_state_loop():
    while True:
        state = myLocalization.getstate()
        print("Sending data...")
        time.sleep(1)

if __name__ == "__main__":
    threading.Thread(target=get_state_loop, daemon=True).start()
    app.run(host="localhost", port=5000, debug=False)
    #or app.run(port=5000)
    #should show up on localhost:5000/state for testing
'''