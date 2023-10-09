import cv2
import numpy as np
from object_detection import ObjectDetection
import math

#Initialize ObjectDetection
od = ObjectDetection()

capture = cv2.VideoCapture("objectRecognition\los_angeles.mp4")

count = 0

center_points_prev_frame = []
trackingObjects = {}
track_id = 0
while True:
    ret, frame = capture.read()

    count += 1
    if not ret:
        break

    center_points_cur_frame = []
    classids, scores, boxes = od.detect(frame)

    for box in boxes:
        x, y, w, h = box
        cx = int((x + x + w)/2)
        cy = int((y + y + h)/2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if count <= 2 :
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    trackingObjects[track_id] = pt
                    track_id += 1

    else:
        tracking_objects_copy = trackingObjects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                # Update IDs position
                if distance < 20:
                    trackingObjects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
            # Remove IDs lost
            if not object_exists:
                trackingObjects.pop(object_id)
        # Add new IDs found
        for pt in center_points_cur_frame:
            trackingObjects[track_id] = pt
            track_id += 1

    for object_id, pt in trackingObjects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), )
    cv2.imshow("Frame", frame)

    center_points_prev_frame = center_points_cur_frame.copy()



    key = cv2.waitKey(0)
    if key == 27:
        break


capture.release()
cv2.destroyAllWindows()