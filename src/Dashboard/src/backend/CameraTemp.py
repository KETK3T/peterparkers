from flask import Flask, Response
from flask_cors import CORS
import cv2

app = Flask(__name__)
CORS(app)

camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # This yields an MJPEG frame
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
@app.route('/api/users')
def get_users():
    return {"users": ["Camera running properly!"]}

@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(debug=True,host='localhost', port=8000)