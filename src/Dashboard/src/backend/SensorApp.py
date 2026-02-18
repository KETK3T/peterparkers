from flask import Flask, jsonify
from flask_cors import CORS

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