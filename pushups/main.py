import cv2
import time
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def angle(a, b, c):
    d = np.arctan2(c[1] - b[1], c[0] - b[0])
    e = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_ = np.degrees(d - e)
    angle_ = angle_ + 360 if angle_ < 0 else angle_
    return 360 - angle_ if angle_ > 180 else angle_

def process(image, keypoints):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    try:
        if left_elbow[0] > 0:
            angle_elbow = angle(left_shoulder, left_elbow, left_wrist)
            x, y = int(left_elbow[0]) + 10, int(left_elbow[1]) + 10
        else:
            angle_elbow = angle(right_shoulder, right_elbow, right_wrist)
            x, y = int(right_elbow[0]) + 10, int(right_elbow[1]) + 10

        cv2.putText(image, f"{int(angle_elbow)}", (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 50, 50), 2)
        return angle_elbow
    except ZeroDivisionError:
        return None


path = Path(__file__).parent
model_path = path / "yolo11n-pose.pt"
model = YOLO(model_path)
cap = cv2.VideoCapture(0)
last_time = time.time()
count = 0
flag = False
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter("out.mp4", fourcc, 10, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - last_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)
    last_time = curr_time
    results = model(frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    if not results:
        continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    keypoints = keypoints[0]
    if not keypoints:
        continue

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    angle_ = process(annotated, keypoints)
    if angle_ is not None:
        if angle_ > 150 and not flag:
            count += 1
            flag = True
        elif angle_ < 90 and flag:
            flag = False

    cv2.putText(annotated, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.imshow("Pose", annotated)
    writer.write(annotated)

writer.release()
cap.release()
cv2.destroyAllWindows()
