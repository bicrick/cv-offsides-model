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

    #BEGIN DEBUG SECTION FOR ONE IMAGE
    imageForVanishingPoints = cv2.imread(cur_path+dataset_path+"/469.jpg")
    #imageForVanishingPoints = cv2.imread("/Users/alexalzaga/Documents/UNI/COMPUTER VISION/FinalProject/images/OG.jpeg")

print('Starting Vanishing Point calculation')
vertical_vanishing_point = get_vanishing_point_v(imageForVanishingPoints, goalDirection)
horizontal_vanishing_point = get_vanishing_point_h(imageForVanishingPoints)
print('Finished Vanishing Point calculation')

print('Starting Player Detection and Classification')
red_pos,white_pos = get_field_positions('Dataset/Offside_Images/',24)
print('Ending Player Detection')

print('Beginning Offsides Calculations')
#Offside Calculate
    #Draw offside line
print('Ending Ofssides Calculation')
print('Finished')
