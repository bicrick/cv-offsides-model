import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def drawLines(red_pos,white_pos,vp,img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    x = int(np.round(vp[0]))
    y = int(np.round(vp[1]))
    for pos in red_pos:
        cv2.line(img,pos,(x,y),(0,0,255),1)
    for pos in white_pos:
        cv2.line(img,pos,(x,y),(255,255,255),1)
    cv2.imwrite('Output/BoxWithLines.jpg',img)

