import keyboard
import cv2
import numpy as np
import math
import os
import time
import pyautogui
import image_cleanup

def hand_crop(crop_img,img):
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    
    # applying gaussian blur
    # applying gaussian blur
    gauss_blur = image_cleanup.gaussian_method(grey)

    # thresholding: Otsu's Binarization method
    thresh_img = image_cleanup.otsu_method(gauss_blur)

    # show thresholded image
    cv2.imshow('Thresholded', thresh_img)

    # check OpenCV version to avoid unpacking error
    (version, _, _) = cv2.__version__.split('.')

    if version == '3':
        image, contours, hierarchy = cv2.findContours(thresh_img.copy(), \
               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(thresh_img.copy(),cv2.RETR_TREE, \
               cv2.CHAIN_APPROX_NONE)

    # find contour with max area
    max_contour = max(contours, key = lambda x: cv2.contourArea(x))

    # create bounding rectangle around the contour (can skip below two lines)
    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(max_contour)

    # draw_contours contours
    draw_contours = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(draw_contours, [max_contour], 0, (0, 255, 0), 0)
    cv2.drawContours(draw_contours, [hull], 0,(0, 0, 255), 0)

    # finding convex hull
    hull = cv2.convexHull(max_contour, returnPoints=False)

    # finding convexity convex_defects
    convex_defects = cv2.convexityconvex_defects(max_contour, hull)
    count_convex_defects = 0
    cv2.drawContours(thresh_img, contours, -1, (0, 255, 0), 3)

    # applying Cosine Rule to find angle for all convex_defects (between fingers)
    # with angle > 90 degrees and ignore convex_defects
    for i in range(convex_defects.shape[0]):
        s,e,f,d = convex_defects[i,0]

        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        if d>9000:
            count_convex_defects+=1
        # find length of all sides of triangle
        '''a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

        # apply cosine rule here
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

        # ignore angles > 90 and highlight rest with red dots
        if angle <= 90:
            count_convex_defects += 1
            cv2.circle(crop_img, far, 1, [0,0,255], -1)
        #dist = cv2.pointPolygonTest(max_contour,far,True)

        # draw a line from start to end i.e. the convex points (finger tips)
        # (can skip this part)
        cv2.line(crop_img,start, end, [0,255,0], 2)
        cv2.circle(crop_img,far,5,[0,0,255],-1)'''

    # define actions required
    count_convex_defects+=1
    if count_convex_defects==2 or count_convex_defects==1:
        keyboard.press_and_release('-')

        
    else:
        cv2.putText(img,"0", (50, 50),\
            cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    ha=0
    
    #time.sleep(1)
    # show appropriate images in windows
    #cv2.imshow('Gesture', img)
    all_img = np.hstack((draw_contours, crop_img))
    cv2.imshow('Contours', all_img)
