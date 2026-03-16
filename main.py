import cv2
import numpy as np

from models.detector import Detector
from models.depth_model import DepthEstimator
from utils.fusion import estimate_distance
from utils.birdseye import create_map


detector = Detector()
depth_model = DepthEstimator()

cap = cv2.VideoCapture(0)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    # resize for faster processing
    frame = cv2.resize(frame, (640,480))

    # ---- DEPTH ESTIMATION ----
    depth = depth_model.predict(frame)

    # ---- DEPTH HEATMAP ----
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype("uint8")
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

    # overlay depth on camera frame
    overlay = cv2.addWeighted(frame, 0.6, depth_vis, 0.4, 0)

    # ---- OBJECT DETECTION ----
    boxes, classes = detector.detect(frame)

    objects = []

    for i, box in enumerate(boxes):

        # estimate distance
        distance = estimate_distance(box, depth)

        x1, y1, x2, y2 = box.astype(int)

        class_id = int(classes[i])
        name = detector.model.names[class_id]

        label = f"{name} {distance:.2f}m"

        # draw bounding box
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),2)

        # draw label
        cv2.putText(
            overlay,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        # object center
        center_x = (x1 + x2) / 2

        # save for bird's-eye map
        objects.append((name, distance, center_x))

    # ---- TOP VIEW MAP ----
    map_img = create_map(objects)

    # ---- DISPLAY WINDOWS ----
    cv2.imshow("Depth Aware Detection", overlay)
    cv2.imshow("Top View Map", map_img)

    # exit on ESC
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()