import asyncio
import websockets
import csv
import json
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/state', methods=['GET'])
def send_coords():
    coords = []
    with open(CSV_FILE, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            coords.append({
                "lat": float(row["lat"]),
                "long": float(row["long"])
            })
    return jsonify(coords)
                # await websocket.send(json.dumps(data))
                # print("Sent:", data)
                #
                # await asyncio.sleep(0.5)


# @app.route('/state', methods=['GET'])
# def send_coords():
#     with open(CSV_FILE, newline='') as csvfile:
#         reader = csv.DictReader(csvfile)
#         rows = list(reader)
#         print("COLUMNS:", reader.fieldnames)  # see exact column names
#         print("FIRST ROW:", rows[0])          # see raw values
#         last = rows[-1]
#         return jsonify({
#             "lat": float(last["lat"]),
#             "long": float(last["long"])
#         })

PORT = 8765
CSV_FILE = "./src/Dashboard/src/backend/temporary_coordinates.txt"

if __name__ == "__main__":
    app.run(host="localhost", port=PORT, use_reloader=False, threaded=True)

# async def send_coordinates(websocket):
#     print("Client connected")
#
#     while True:
#         with open(CSV_FILE, newline='') as csvfile:
#             reader = csv.DictReader(csvfile)
#
#             for row in reader:
#                 data = {
#                     "lat": float(row["lat"]),
#                     "long": float(row["long"])
#                 }
#
#                 await websocket.send(json.dumps(data))
#                 print("Sent:", data)
#
#                 await asyncio.sleep(0.5)  # 0.5 second timer
#
#         # Loop back to beginning of file
#
# async def main():
#     async with websockets.serve(send_coordinates, "localhost", PORT):
#         print(f"WebSocket server running on ws://localhost:{PORT}")
#         await asyncio.Future()  # run forever
#
# asyncio.run(main())