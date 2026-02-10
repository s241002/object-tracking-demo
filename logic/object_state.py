from enum import Enum
import math

class ObjectState(Enum):
    UNSEEN   = 0
    PRESENT  = 1
    MOVED    = 2
    OCCLUDED = 3
    MISSING  = 4
    REMOVED  = 5


def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_distance(b1, b2):
    x1, y1 = bbox_center(b1)
    x2, y2 = bbox_center(b2)
    return math.hypot(x1 - x2, y1 - y2)


def iou(b1, b2):
    if not b1 or not b2:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class TrackedObject:
    def __init__(self, obj_id, cfg):
        self.id = obj_id
        self.cfg = cfg

        self.state = ObjectState.UNSEEN
        self.last_bbox = None

        self.move_counter = 0
        self.missing_counter = 0

    def update(self, detected_bbox, person_bboxes):
        # --- 初回検出 ---
        if self.state == ObjectState.UNSEEN and detected_bbox:
            self.state = ObjectState.PRESENT
            self.last_bbox = detected_bbox
            return

        # --- 検出あり ---
        if detected_bbox:
            self.missing_counter = 0

            if self.last_bbox:
                dist = center_distance(self.last_bbox, detected_bbox)
                if dist > self.cfg["move_distance_px"]:
                    self.move_counter += 1
                else:
                    self.move_counter = 0

                if self.move_counter >= self.cfg["move_confirm_frames"]:
                    self.state = ObjectState.MOVED
                else:
                    self.state = ObjectState.PRESENT

            self.last_bbox = detected_bbox
            return

        # --- 検出なし ---
        self.missing_counter += 1

        # 遮蔽判定
        for pb in person_bboxes:
            if iou(pb, self.last_bbox) > self.cfg["occlusion_iou"]:
                self.state = ObjectState.OCCLUDED
                return

        if self.missing_counter >= self.cfg["removed_frames"]:
            self.state = ObjectState.REMOVED
        elif self.missing_counter >= self.cfg["missing_frames"]:
            self.state = ObjectState.MISSING
