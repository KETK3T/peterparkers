import asyncio
import websockets
import csv
import json

PORT = 8765
CSV_FILE = "./src/Dashboard/src/backend/temporary_coordinates.txt"

async def send_coordinates(websocket):
    print("Client connected")

    while True:
        with open(CSV_FILE, newline='') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                data = {
                    "lat": float(row["lat"]),
                    "long": float(row["long"])
                }

                await websocket.send(json.dumps(data))
                print("Sent:", data)

                await asyncio.sleep(2)  # 0.5 second timer

        # Loop back to beginning of file

async def main():
    async with websockets.serve(send_coordinates, "localhost", PORT):
        print(f"WebSocket server running on ws://localhost:{PORT}")
        await asyncio.Future()  # run forever

asyncio.run(main())
