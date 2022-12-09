import numpy as np
import cv2
import math
import json
import sys
import os
import warnings
from vanishingPoints import *
from playerDetection import get_field_positions
from offsidesCalculation import *

#Goal Direction
goalDirection = 'right'

#Dataset Path
dataset_path = '\Dataset\Offside_Images'
cur_path = os.getcwd()
fileNames = []
tempFileNames = os.listdir(cur_path+dataset_path)
for fileName in tempFileNames:
    fileNames.append(cur_path+dataset_path+str(fileName))

initial_image = cv2.imread(cur_path+dataset_path+"/24.jpg")

imageForVanishingPoints = initial_image.copy()

print('Starting Vanishing Point calculation')
vertical_vanishing_point = get_vanishing_point_v(imageForVanishingPoints, goalDirection)
horizontal_vanishing_point = get_vanishing_point_h(imageForVanishingPoints)
print(horizontal_vanishing_point)
print('Finished Vanishing Point calculation')

print('Starting Player Detection and Classification')
red_pos,white_pos,img_final = get_field_positions('Dataset/Offside_Images/',24)
#https://towardsdatascience.com/football-players-tracking-identifying-players-team-based-on-their-jersey-colors-using-opencv-7eed1b8a1095
print('Ending Player Detection')

print('Beginning Offsides Calculations')
drawLines(red_pos,white_pos,horizontal_vanishing_point,img_final)
    #Connect Points to Vanishing Lines
    #Find the defender closest to the goal(white)
    #Find the attacker closest to the goal(red)
#Draw offside line
print('Ending Ofssides Calculation')
print('Finished')
