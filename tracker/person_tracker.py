import itertools

class PersonTracker:
    def __init__(self, iou_threshold=0.3):
        self.tracks = {}
        self.next_id = itertools.count(1)
        self.iou_threshold = iou_threshold

    def update(self, detections):
        updated_tracks = {}

        for det in detections:
            matched_id = None
            for pid, track in self.tracks.items():
                if self.iou(det, track["bbox"]) > self.iou_threshold:
                    matched_id = pid
                    break

            if matched_id is None:
                matched_id = next(self.next_id)

            updated_tracks[matched_id] = {
                "id": matched_id,
                "bbox": det
            }

        self.tracks = updated_tracks
        return list(self.tracks.values())

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)

        return inter / (area_a + area_b - inter)
