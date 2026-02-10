from ultralytics import YOLO

class YoloDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg["model_path"])
        self.person_cls = cfg["person_class_id"]
        self.object_cls = set(cfg["object_class_ids"])

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.cfg["yolo_conf"],
            iou=self.cfg["yolo_iou"],
            device=self.cfg["device"],
            verbose=False
        )

        persons = []
        objects = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == self.person_cls:
                    persons.append((x1, y1, x2, y2))
                elif cls in self.object_cls:
                    objects.append((x1, y1, x2, y2))

        return persons, objects
