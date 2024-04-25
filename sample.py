import cv2
import numpy as np
import cropper
import time

def draw_rectangle(event, x, y, flags, param):
    global point, img
    if event == cv2.EVENT_LBUTTONDOWN:
        point = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        point.append((x, y))
        cv2.rectangle(img, point[0], point[1], (0, 255, 0), 2)
        cv2.imshow("img", img)

def select_and_crop(fpath):
    global img
    img = cv2.imread(fpath)
    clone = img.copy()
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", draw_rectangle)

    while True:
        cv2.imshow("img", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            img = clone.copy()
        elif key == ord('c'):
            break
        else:
            point = [(0, 0), (img.shape[1], img.shape[0])]
            break
    cv2.destroyAllWindows()

    if len(point) == 2:
        cropper.crop_img(fpath)

def crop_no_selection(fpath):
    img = cv2.imread(fpath)
    cv2.imshow("orig",img)
    time.sleep(0.5)
    cv2.destroyAllWindows()
    cropper.crop_img(fpath)


# img = cv2.imread('./data/test1.png')
# img = cv2.imread('./data/19-1.png')
# img = cv2.imread('./data/19-2.png')
# img = cv2.imread('./data/19-3.png')
# img = cv2.imread('./data/19-4.png')
# img = cv2.imread('./data/19-5.png')

# select_and_crop("./data/fs1.png")
# select_and_crop("./data/fs3.png")
crop_no_selection("./data/fs3.png")