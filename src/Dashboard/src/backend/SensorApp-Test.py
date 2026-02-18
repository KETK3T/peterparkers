from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time

app = Flask(__name__)
CORS(app)

current_state = {
    "longitude": 0.0,
    "latitude": 0.0,
    "vx": 0.0,
    "vy": 0.0,
    "heading": 0.0
}

@app.route("/state")
def state():
    return jsonify(current_state)

def fake_sensor():
    while True:
        current_state["longitude"] += 1
        current_state["latitude"] += 1
        current_state["vx"] = 0
        current_state["vy"] = 0
        current_state["heading"] = 0

        print("Sending zero state...")
        time.sleep(1)  # simulate 1Hz update

if __name__ == "__main__":
    threading.Thread(target=fake_sensor, daemon=True).start()
    app.run(port=5000)
