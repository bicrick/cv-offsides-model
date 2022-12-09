"""
identify players in the image .
author:Hamza Oukaddi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image

def get_field_positions(root,im):
    """Identify peoples (players and people standing by) in the image
    and write their positions to a csv file. 
    Also, save the image with the identified people highlighted.
    

    Args:
        im (path): path to the image
    
    """
    img = plt.imread(root+str(im)+'.jpg')
    # copy of OG image to show boxes
    img_final = img

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ## mask of green (36,0,0) ~ (70, 255,255)
    mask_g = cv2.inRange(hsv_img, (36, 100, 100), (86, 255, 255))
    mask = np.invert(mask_g)
    kernel = np.ones((50,50),np.uint8)
    mask_crowd = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_crowd = np.invert(mask_crowd)

    # Open the mask image as numpy array and convert to 3-channels
    npMask=np.array(Image.fromarray(mask_crowd).convert("RGB"))

    # Make a binary array identifying where the mask is white
    cond = npMask>128

    # Select image or mask according to condition array
    pixels=np.where(cond, img, npMask)
    result=Image.fromarray(pixels)
    result.save('result.png')
    img = np.array(result)

    classes = None 
    
    # read coco class names
    with open('yolo_cnn/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # size of image
    Width = img.shape[1]
    Height = img.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolo_cnn/yolov3.weights', 'yolo_cnn/yolov3.cfg')

    # create input blob 
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    #create bounding box 
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
    
    #check if is people detection
    player_number=0
    u_im=[] # player postion
    boxes_player=[]
    for i in indices:
        box = boxes[i]
        if class_ids[i]==0:
            boxes_player.append(box)

            label = str(f"player{player_number}") 
            player_number +=1
            cv2.rectangle(img_final, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
            cv2.putText(img_final, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            bottom_right_corner=(round(box[0]+box[2]),round(box[1]+box[3]))
            u_im.append(bottom_right_corner)
            
    # save image with identified people highlighted
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,1,1)
    # remove axis
    ax1.axis('off')
    ax1.imshow(img_final)
    plt.savefig('Output/'+str(im)+'Box.jpg', bbox_inches='tight')
    plt.title("detection")
    
    # save coordinte of players
    with open('Output/'+str(im)+'Coords.csv', 'w',newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(u_im)

    return img