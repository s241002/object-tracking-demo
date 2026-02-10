import cv2

def draw_tracked_object(frame, tracked_object, cfg):
    if not cfg.get("draw_bbox", True):
        return frame

    if tracked_object.last_bbox is None:
        return frame

    x1, y1, x2, y2 = tracked_object.last_bbox
    state_name = tracked_object.state.name

    # config から色取得（BGR）
    color = cfg["state_colors"].get(state_name, [255, 255, 255])
    color = tuple(int(c) for c in color)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if cfg.get("draw_state_text", True):
        cv2.putText(
            frame,
            state_name,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame
