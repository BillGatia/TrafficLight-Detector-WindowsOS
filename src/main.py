#!/usr/bin/env python
# coding: utf-8
# created by hevlhayt@foxmail.com 
# Date: 2016/1/15 
# Time: 19:20
#
import os
import cv2
import numpy as np


def detect(filepath, file):
    # Backwards-compatible file-based entrypoint: read image and delegate
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_path = os.path.join(filepath, file)
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image:", img_path)
        return

    cimg = detect_frame(img)

    result_dir = os.path.join(filepath, 'result')
    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite(os.path.join(result_dir, file), cimg)
    cv2.imshow('detected results', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_frame(img):
    """Run detection on a BGR image (frame). Returns annotated image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cimg = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,120,80])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,120,80])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # Supplemental range for overexposed green lights (low S, high V)
    lower_green_wash = np.array([30,15,170])
    upper_green_wash = np.array([95,120,255])
    lower_yellow = np.array([15,150,100])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    maskg_wash = cv2.inRange(hsv, lower_green_wash, upper_green_wash)
    maskg = cv2.bitwise_or(maskg, maskg_wash)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    # morphological closing to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    maskr = cv2.morphologyEx(maskr, cv2.MORPH_CLOSE, kernel)
    maskg = cv2.morphologyEx(maskg, cv2.MORPH_CLOSE, kernel)
    masky = cv2.morphologyEx(masky, cv2.MORPH_CLOSE, kernel)

    size = img.shape

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=12, minRadius=2, maxRadius=40)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 70,
                                 param1=50, param2=9, minRadius=2, maxRadius=40)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=7, minRadius=2, maxRadius=40)

    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            xi = int(i[0]); yi = int(i[1]); ri = int(i[2])
            if xi > size[1] or yi > size[0] or yi > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):
                    yy = yi + m
                    xx = xi + n
                    if yy < 0 or xx < 0 or yy >= size[0] or xx >= size[1]:
                        continue
                    h += maskr[yy, xx]
                    s += 1
            if s > 0 and h / s > 60:
                cv2.circle(cimg, (xi, yi), ri+10, (0, 255, 0), 2)
                cv2.circle(maskr, (xi, yi), ri+30, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(xi, yi), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            xi = int(i[0]); yi = int(i[1]); ri = int(i[2])
            if xi > size[1] or yi > size[0] or yi > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):
                    yy = yi + m
                    xx = xi + n
                    if yy < 0 or xx < 0 or yy >= size[0] or xx >= size[1]:
                        continue
                    h += maskg[yy, xx]
                    s += 1
            if s > 0 and h / s > 100:
                cv2.circle(cimg, (xi, yi), ri+10, (0, 255, 0), 2)
                cv2.circle(maskg, (xi, yi), ri+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(xi, yi), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            xi = int(i[0]); yi = int(i[1]); ri = int(i[2])
            if xi > size[1] or yi > size[0] or yi > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):
                    yy = yi + m
                    xx = xi + n
                    if yy < 0 or xx < 0 or yy >= size[0] or xx >= size[1]:
                        continue
                    h += masky[yy, xx]
                    s += 1
            if s > 0 and h / s > 65:
                cv2.circle(cimg, (xi, yi), ri+10, (0, 255, 0), 2)
                cv2.circle(masky, (xi, yi), ri+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(xi, yi), font, 1,(255,0,0),2,cv2.LINE_AA)

    return cimg

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera")

    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video")
                break
            annotated = detect_frame(frame)
            cv2.imshow('detected results', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    cap.release()
    cv2.destroyAllWindows()




# if __name__ == '__main__':

#     # path = os.path.abspath('..')+'\\light\\'
#     path = "D:\\programLearning\\SchoolOfTheRobotTechnology\\ProjectDesign2\\TrafficLight-Detector-master\\TrafficLight-Detector-master\\light"
#     for f in os.listdir(path):
#         print (f)
#         if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png') or f.endswith('.PNG'):
#             detect(path, f)
