import cv2

def objects(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    contur, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cir_cnt = 0
    squ_cnt = 0
    for contour in contur:
        if cv2.contourArea(contour) < 100:
            continue
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            squ_cnt += 1
        elif len(approx) > 4:
            cir_cnt += 1

    return cir_cnt, squ_cnt

img = "figures.png"
cir, squ = objects(img)
print(f"Кругов: {cir}, Квадратов: {squ}")
print(f"Общее количество объектов: {cir + squ}")
