import numpy as np
import cv2
import math
import json
import sys
import os
import warnings
from vanishingPoints import *
from playerDetection import get_field_positions

#Goal Direction
goalDirection = 'right'

#Dataset Path
dataset_path = '\Dataset\Offside_Images'
cur_path = os.getcwd()
fileNames = []
tempFileNames = os.listdir(cur_path+dataset_path)
for fileName in tempFileNames:
    fileNames.append(cur_path+dataset_path+str(fileName))

#keeper = [4.4931905696916325e-06, 4.450801979411523e-06, 5.510516736414265e-07, 0.00021567314734519837, 0.002188183807439825, 0.0015186984125557716, 0.7527352297592997, 1.0, 0.20787746170678337]
#referee = [8.72783130847647e-06, 1.5868784197229944e-07, 0.0, 0.0010298840944002235, 0.0002880184331797235, 0.002688172043010753, 0.3064516129032258, 0.05913978494623656, 1.0]

    #BEGIN DEBUG SECTION FOR ONE IMAGE
    imageForVanishingPoints = cv2.imread(cur_path+dataset_path+"/469.jpg")
    #imageForVanishingPoints = cv2.imread("/Users/alexalzaga/Documents/UNI/COMPUTER VISION/FinalProject/images/OG.jpeg")

print('Starting Vanishing Point calculation')
vertical_vanishing_point = get_vanishing_point_v(imageForVanishingPoints, goalDirection)
horizontal_vanishing_point = get_vanishing_point_h(imageForVanishingPoints)
print('Finished Vanishing Point calculation')

print('Starting Player Detection')
playerDetectionImg = get_field_positions('Dataset/Offside_Images/',5)
print('Ending Player Detection')
