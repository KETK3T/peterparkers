from flask import Flask, jsonify
from flask_cors import CORS
#import localization.py class

app = Flask(__name__)
CORS(app)

# Initialize ONCE
# this won't work because of while loop
#localization.run()


@app.route("/sensor")
def get_sensor_state():
    state = []
    #state = localization.efk.get_state()
    return jsonify(state)

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=False)