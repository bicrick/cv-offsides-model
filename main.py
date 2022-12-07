import numpy as np
import cv2
import math
import json
import sys
import os
import warnings
from vanishingPoints import *

#Goal Direction
goalDirection = 'right'

#Dataset Path
dataset_path = '\CV_Offsides_Model\Dataset\Offside_Images'
cur_path = os.getcwd()
fileNames = []
tempFileNames = os.listdir(cur_path+dataset_path)
for fileName in tempFileNames:
    fileNames.append(cur_path+dataset_path+str(fileName))


#BEGIN DEBUG SECTION FOR ONE IMAGE
imageForVanishingPoints = cv2.imread(cur_path+dataset_path+"/0.jpg")
#imageForVanishingPoints = cv2.imread("/Users/alexalzaga/Documents/UNI/COMPUTER VISION/FinalProject/images/OG.jpeg")

print('Staring Vanishing Point calculation')
vertical_vanishing_point = get_vanishing_point_v(imageForVanishingPoints, goalDirection)
horizontal_vanishing_point = get_vanishing_point_h(imageForVanishingPoints)
# cv2.imwrite(vanishing_point_viz_base_path+tempFileNames[file_itr], imageForVanishingPoints)
print('Finished Vanishing Point calculation')