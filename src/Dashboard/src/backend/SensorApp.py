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
    return jsonify(state)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)