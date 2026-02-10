
print("Model path:", cfg["model_path"])
print("Device:", cfg["device"])
print("Conf:", cfg["yolo_conf"])
print("IoU:", cfg["yolo_iou"])
print("Interval:", cfg["yolo_interval"])

import cv2
import json
import yaml
from detector.yolo_detector import YoloDetector
from logic.object_state import TrackedObject
from tracker.person_tracker import PersonTracker
from tracker.tracked_object import TrackedObject, ObjectState
from utils.draw import draw_tracked_object

cap = cv2.VideoCapture("input/demo.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "output/annotated.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps, (w, h)
)

with open("configs/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

detector = YoloDetector(
    model_path="models/yolo_custom.pt",
    device="cpu"
)
tracked_object = TrackedObject(obj_id=1)

person_tracker = PersonTracker(
    iou_threshold=cfg["person_tracker"]["iou_threshold"]
)

near_person = tracked_object.nearest_person(person_tracks)

events = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    persons, objects = detector.detect(frame)
    obj_bbox = objects[0] if objects else None
    person_tracks = person_tracker.update(persons)

    prev_state = tracked_object.state
    tracked_object.update(obj_bbox, persons)

    if tracked_object.state != prev_state:
        events.append({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 2),
            "object_id": tracked_object.id,
            "from": prev_state.name,
            "to": tracked_object.state.name,
            "near_person": near_person
        })

    # 描画（超簡易）
    if tracked_object.last_bbox:
        x1, y1, x2, y2 = tracked_object.last_bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame,
            tracked_object.state.name,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,255,0), 2
        )
    frame = draw_tracked_object(frame, tracked_object, cfg)
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

with open("output/events.json", "w") as f:
    json.dump(events, f, indent=2)

