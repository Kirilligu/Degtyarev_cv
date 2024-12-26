import cv2
import numpy as np

lower = np.array([6, 120, 100])
upper = np.array([26, 255, 255])
video = cv2.VideoCapture('output.avi')
img = 0
while True:
    success, frame = video.read()
    if not success:
        break
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower, upper)
    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_l = [cnt for cnt in contour if cv2.contourArea(cnt) > 100]
    if len(contour_l) == 9:
        img += 1

print(f"Кол-во изображений: {img}")
video.release()
