import numpy as np
from utils.calibration import FOCAL_LENGTH

def estimate_distance(box, depth_map):

    x1, y1, x2, y2 = box.astype(int)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    depth_value = depth_map[cy, cx]

    # normalize depth
    depth_norm = depth_value / depth_map.max()

    distance = depth_norm * 5  # scale to meters approx

    return distance