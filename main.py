from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import base64

app = Flask(__name__)
model = YOLO('yolov8n.pt')

# فتح الفيديو
cap = cv2.VideoCapture('1.mp4')
# فتح ملفات الفيديو
videos = [
    cv2.VideoCapture("1.mp4"),
    cv2.VideoCapture("2.mp4"),
    cv2.VideoCapture("14.mp4"),
    cv2.VideoCapture("4.mp4")
]
def generate_framess():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # تطبيق YOLO على الإطار
        results = model(frame, classes=[0])  # كشف الأشخاص فقط
        count = len(results[0].boxes)
        annotated = results[0].plot()

        # تحويل الإطار للبث
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
def generate_frames(video_index):
    cap = videos[video_index]
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # إعادة التشغيل عند الانتهاء
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('dashboard.html')  # تأكد أن فيه الفيديوهات بالـ IDات المطلوبة

@app.route('/video_feed1')
def video_feed1():
    return Response(generate_framess(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_frames(1), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3')
def video_feed3():
    return Response(generate_frames(2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed4')
def video_feed4():
    return Response(generate_frames(3), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
