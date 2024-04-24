import cv2
import numpy as np

def draw_rectangle(event, x, y, flags, param):
    global point, img
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        point.append((x, y))
        cv2.rectangle(img, point[0], point[1], (0, 255, 0), 2)
        cv2.imshow("img", img)

img = cv2.imread('./data/test1.png')
clone = img.copy()
cv2.namedWindow("img")
cv2.setMouseCallback("img", draw_rectangle)

while True:
    cv2.imshow("img", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        img = clone.copy()
    elif key == ord('c'):
        break
cv2.destroyAllWindows()

if len(point) == 2:
    roi = img[min(point[0][1], point[1][1]):max(point[0][1], point[1][1]),
                min(point[0][0], point[1][0]):max(point[0][0], point[1][0])]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('threshed', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    filled = thresholded.copy()  
    for contour in contours:
        cv2.drawContours(filled, [contour], -1, 255, -1)

    cv2.imshow('filled attempt', filled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 250000
    min_aspect_ratio = 0.8
    max_aspect_ratio = 1.2
    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                valid_contours.append(contour)

    largest_contour = max(valid_contours, key=cv2.contourArea)
    cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 3)

    cv2.imshow('lg valid cont', roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()