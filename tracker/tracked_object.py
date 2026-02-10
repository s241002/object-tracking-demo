from enum import Enum, auto

class ObjectState(Enum):
    INIT = auto()
    PRESENT = auto()
    OCCLUDED = auto()
    MISSING = auto()


class TrackedObject:
    def __init__(self, object_id, cfg):
        self.id = object_id
        self.cfg = cfg
        self.bbox = None
        self.state = ObjectState.INIT
        self.missing_count = 0

    def update(self, obj_bbox, person_tracks):
        if obj_bbox is not None:
            self.bbox = obj_bbox
            self.missing_count = 0
            self.state = ObjectState.PRESENT
        else:
            self.missing_count += 1
            if self.missing_count > self.cfg["object"]["missing_frames"]:
                self.state = ObjectState.MISSING
            else:
                self.state = ObjectState.OCCLUDED

    # ★さっきの関数はここ
    def nearest_person(self, person_tracks, dist_thresh=50):
        if self.bbox is None:
            return None

        ox1, oy1, ox2, oy2 = self.bbox
        ocx = (ox1 + ox2) / 2
        ocy = (oy1 + oy2) / 2

        for p in person_tracks:
            px1, py1, px2, py2 = p["bbox"]
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2

            dist = ((ocx - pcx)**2 + (ocy - pcy)**2) ** 0.5
            if dist < dist_thresh:
                return p["id"]

        return None
