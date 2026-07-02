from ultralytics import YOLO
from typing import List


_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # fast + lightweight
    return _model


def detect_objects(frame_path: str) -> List[str]:

    model = load_model()
    results = model(frame_path, verbose=False)

    names = model.names
    objects = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            objects.append(names[cls_id])

    return list(set(objects))  # remove duplicates