from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# Load the custom YOLOv8 model
model = YOLO('D:/FreshAndRotten2/FreshAndRotten/runs/detect/train/weights/best.pt')

camera = None  # Global variable for the video stream


def generate_frames():
    """Generator function to process the video frames and perform detection."""
    global camera
    while True:
        if camera is None:
            break
        success, frame = camera.read()
        if not success:
            break

        # Run YOLOv8 detection on the frame
        results = model(frame, imgsz=640)  # Adjust `imgsz` if needed
        annotated_frame = results[0].plot()  # Annotate detections

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Render the main page with the Detect button and video placeholder."""
    return render_template('index.html')


@app.route('/start_video')
def start_video():
    """Start the video stream and perform real-time detection."""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Use the laptop's default camera
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_video')
def stop_video():
    """Stop the video stream."""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Video stopped."


if __name__ == '__main__':
    app.run(debug=True)


