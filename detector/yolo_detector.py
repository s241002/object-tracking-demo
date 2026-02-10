from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu"):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame):
        results = self.model(
            frame,
            conf=0.3,
            iou=0.7,
            device=self.device,
            verbose=False
        )

        persons = []
        objects = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # クラスIDは学習データに合わせて調整
                if cls == 0:        # person
                    persons.append((x1, y1, x2, y2))
                elif cls == 1:      # bag（例）
                    objects.append((x1, y1, x2, y2))

        return persons, objects
