# mello_core/detector.py

import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="models/yolov8n-seg.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, frame):
        results = self.model.predict(frame, verbose=False)
        detections = []

        if not results or not results[0].boxes:
            return detections

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, xyxy)

            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })

        return detections