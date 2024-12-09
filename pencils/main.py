import cv2
import numpy as np
total = 0
for i in range(1, 13):
    img = cv2.imread(f"images/img ({i}).jpg", cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = 0
    for c in cnts:
        rect = cv2.boxPoints(cv2.minAreaRect(c)).astype(int)
        w = np.linalg.norm(rect[0] - rect[1])
        h = np.linalg.norm(rect[0] - rect[3])
        if (h > 3 * w and h > 900) or (w > 3 * h and w > 900):
            cnt += 1
    total += cnt
    print(f"Карандашей на изображении {i}: {cnt}")
print("Общее количество карандашей:", total)
