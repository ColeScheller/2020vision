import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

# hue = [27.51798561151079, 50.98976109215017]
# sat = [57.32913669064748, 67.8839590443686]
# lum = [0, 255.0]
hue = [50, 100]
sat = [100, 200]
lum = [100, 255.0]

while(True):
    print("Enter loop")
    ret, img = cap.read()
    #img = cv2.medianBlur(img,5)

    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    bw_img = cv2.inRange(hls_img, (hue[0], lum[0], sat[0]),  (hue[1], lum[1], sat[1]))

    img_erode = cv2.erode(bw_img, None)
    
    cv2.imshow('anything',img_erode)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
