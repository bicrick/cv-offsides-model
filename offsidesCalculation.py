import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

def drawLines(red_pos,white_pos,vp,img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    x = int(np.round(vp[0]))
    y = int(np.round(vp[1]))
    for pos in red_pos:
        cv2.line(img,pos,(x,y),(0,0,255),2)
    for pos in white_pos:
        cv2.line(img,pos,(x,y),(255,255,255),2)
    return img

def determineOffsides(red_pos,white_pos,vp,img,goalDirection):
    #Determine attackers/defenders of players
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    defender = []
    attacker = []
    attackColor = []
    defendColor = []
    
    #Attack White, Defend Red
    defendColor = (0,0,255)
    attackColor = (255,255,255)
    defender = red_pos
    attacker = white_pos


    # #Attack Red, Defend White
    # defendColor = (255,255,255)
    # attackColor = (0,0,255)
    # defender = white_pos
    # attacker = red_pos

    attackerAnglePos = []
    defenderAnglePos=[]
    for pos in attacker:
        angle = calc_angle(vp,pos,goalDirection)
        entry = {'pos':pos,'angle':angle}
        attackerAnglePos.append(entry)
    for pos in defender:
        angle = calc_angle(vp,pos,goalDirection)
        entry = {'pos':pos,'angle':angle}
        defenderAnglePos.append(entry)

    attackerSorted = sorted(attackerAnglePos, key=lambda x : x['angle'])
    defenderSorted = sorted(defenderAnglePos, key=lambda x : x['angle'])

    lastAttacker = attackerSorted[0]

    lastDefender = defenderSorted[0]

    drawLine(lastAttacker['pos'],vp,img,attackColor)
    drawLine(lastDefender['pos'],vp,img,defendColor)

    if(lastAttacker['angle']<lastDefender['angle']):
        print('------------------Offsides Detected!-------------------')
        cv2.putText(img,"OFF",(lastAttacker['pos'][0]-20,lastAttacker['pos'][1]+100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,255),5)
    else:
        print('-----------------No Offsides Detected!-----------------')
        cv2.putText(img,"ON",(lastAttacker['pos'][0]-20,lastAttacker['pos'][1]+100),cv2.FONT_HERSHEY_SIMPLEX,4,(0,255,255),5)
    
    return img


def calc_angle(vp,pos,goalDirection):
    b = np.array(vp)
    c = np.array(pos)
    reference = (0,vp[1])
    a= np.array(reference)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    if goalDirection == 'left':
        if reference[0] > vp[0]:
            angle = -1 * angle
    if goalDirection == 'right':
        if reference[0] < vp[0]:
            angle = -1 * angle     
        
    return angle

def drawLine(pos,vp,img,color):
    x = int(np.round(vp[0]))
    y = int(np.round(vp[1]))
    cv2.line(img,pos,(x,y),color,2)