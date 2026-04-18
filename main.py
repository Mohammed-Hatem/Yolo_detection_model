import sys

import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from ultralytics import YOLO

WEIGHTS_PATH = "/home/mohammed/PycharmProjects/JupyterProject/runs/detect/train4/weights/best.pt"
TARGET_NAME = "green crab"
CONF_THRESHOLD = 0.70


class CrabCounterApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Green Crab Detection")

        self.video_label = QLabel("Starting camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.count_label = QLabel("Green crabs: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        self.save_button = QPushButton("Save Snapshot")
        self.save_button.clicked.connect(self.save_snapshot)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.count_label)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.model = YOLO(WEIGHTS_PATH)
        names = self.model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        self.target_ids = [i for i, n in items if str(n).strip().lower() == TARGET_NAME]
        if not self.target_ids:
            raise ValueError(f"Class '{TARGET_NAME}' not found in model.names")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.last_frame = None

    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            classes=self.target_ids,
            device="cuda",
            verbose=False,
        )
        boxes = results[0].boxes
        count = int(len(boxes)) if boxes is not None else 0
        annotated_bgr = results[0].plot()
        self.last_frame = annotated_bgr

        self.count_label.setText(f"Green crabs: {count}")

        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def save_snapshot(self):
        if self.last_frame is None:
            return
        cv2.imwrite("snapshot.jpg", self.last_frame)

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()


app = QApplication(sys.argv)
window = CrabCounterApp()
window.resize(960, 720)
window.show()
sys.exit(app.exec_())


