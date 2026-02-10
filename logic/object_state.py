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


class TrackedObject:
    def __init__(self, obj_id):
        self.id = obj_id
        self.state = ObjectState.UNSEEN
        self.last_bbox = None
        self.missing_count = 0

    def update(self, detected_bbox, person_bboxes):
        # 初回検出
        if self.state == ObjectState.UNSEEN and detected_bbox:
            self.state = ObjectState.PRESENT
            self.last_bbox = detected_bbox
            return

        # 検出されている場合
        if detected_bbox:
            self.missing_count = 0

            if self.last_bbox:
                dist = center_distance(self.last_bbox, detected_bbox)
                if dist > 30:  # px（仮）
                    self.state = ObjectState.MOVED
                else:
                    self.state = ObjectState.PRESENT

            self.last_bbox = detected_bbox
            return

        # 検出されていない場合
        self.missing_count += 1

        # 人との重なりチェック（簡易）
        for pb in person_bboxes:
            if self._overlap(pb, self.last_bbox):
                self.state = ObjectState.OCCLUDED
                return

        if self.missing_count > 30:
            self.state = ObjectState.REMOVED
        else:
            self.state = ObjectState.MISSING

    def _overlap(self, b1, b2):
        if not b1 or not b2:
            return False
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        return (x2 - x1) > 0 and (y2 - y1) > 0

