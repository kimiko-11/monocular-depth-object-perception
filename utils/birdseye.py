import cv2
import numpy as np

def create_map(objects):

    map_img = np.zeros((500,500,3), dtype=np.uint8)

    robot_x = 250
    robot_y = 450

    # draw robot
    cv2.circle(map_img,(robot_x,robot_y),8,(0,255,0),-1)

    for name, distance, x in objects:

        map_x = int(robot_x + (x-320)*0.5)
        map_y = int(robot_y - distance*80)

        cv2.circle(map_img,(map_x,map_y),6,(0,0,255),-1)
        cv2.putText(map_img,name,(map_x+5,map_y),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

    return map_img