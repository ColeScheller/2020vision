import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

cap = cv2.VideoCapture(0)

# hue = [27.51798561151079, 50.98976109215017]
# sat = [57.32913669064748, 67.8839590443686]
# lum = [0, 255.0]
hue = [50, 100]
sat = [100, 200]
lum = [100, 255.0]

low_thresh = 50
high_thresh = 150

counter = 0

while(True):
    ret, img = cap.read()
    #img = cv2.medianBlur(img,5)

    #Change colorspace to HLS & Threshold image just to green vals, then erode image to kill noise
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    bw_img = cv2.inRange(hls_img, (hue[0], lum[0], sat[0]),  (hue[1], lum[1], sat[1]))
    img_erode = cv2.erode(bw_img, None, None, None, 4)
    img_edges = cv2.Canny(img_erode, low_thresh, high_thresh)
    

    #Detect lines using Hough Transform
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 50
    max_line_gap = 20
    line_image = np.copy(img) * 0
    lines_raw = cv2.HoughLinesP(img_edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
    lines = list(lines_raw)
    #LINES LIST SYNTAX
    #NAME NUMBER    NOT SURE   SELECT X (0) OR Y (1)
    #lines [0]      [1]        [x/y]
    
    #Draw lines on image
    #cv2.line(img, [lines[0][0], lines[0][1]], [lines[0][3], lines[0][4]], color[255, 0, 0])
    
    cv2.imshow('anything', img_edges)

    #Start prints
    os.system('clear')
    counter += 1
    print("Loop #: ", counter)
    for i in range(0, len(lines)):
        print(lines[i][0][0])
    
    #End of loop operations

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
