from flask import Flask, jsonify
from flask_cors import CORS

import sys
import os


path_to_add = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../imu-gps'))
sys.path.append(path_to_add)

from localization-test import Localization

app = Flask(__name__)
CORS(app)

myLocalization = Localization()


@app.route("/sensor")
def get_sensor_state():
    state = myLocalization.getstate()

    state_dict = {
        "longitude": float(state.x[0]),
        "latitude": float(state.x[1]),
        "velocity_x": float(state.x[2]),
        "velocity_y": float(state.x[3]),
        "heading": float(state.x[4])
    }

    return jsonify(state)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)