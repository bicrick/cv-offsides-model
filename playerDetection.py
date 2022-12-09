import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image

def get_field_positions(root,im,goalDirection):
    
    img = plt.imread(root+str(im)+'.jpg')
    # copy of OG image to show boxes
    img_final = img.copy()
    img_unaltered = cv2.imread(root+str(im)+'.jpg')

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
    #result.save('result.png')
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
    white_number=0
    red_number=0
    red_pos=[]
    white_pos=[]
    boxes_player=[]
    for i in indices:
        box = boxes[i]
        if class_ids[i]==0:
            boxes_player.append(box)
            masked_outputs=[]
            cropped_img = get_cropped_image(img_unaltered,box)

            red = get_red_mask(cropped_img)
            white = get_white_mask(cropped_img)

            redCountGray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
            whiteCountGray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)

            redCount = cv2.countNonZero(redCountGray)
            whiteCount = cv2.countNonZero(whiteCountGray)
            

            #print(str(player_number)+' : '+ str(redCount))
            #print(str(player_number)+' : '+ str(whiteCount))

            if(redCount+whiteCount>400):
                if(redCount>whiteCount):
                    #Draw Red 
                    label = str(f"player{red_number}")
                    cv2.rectangle(img_final, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 0, 0), 2)
                    cv2.putText(img_final, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    #Increment Red Defenders
                    red_number+=1

                    if goalDirection== 'left':
                        bottom_left_corner=(round(box[0]),round(box[1]+box[3]))
                        red_pos.append(bottom_left_corner)
                        cv2.circle(img_final,bottom_left_corner,radius=2,color=(0,0,0),thickness=3)
                    else:
                        bottom_right_corner=(round(box[0]+box[2]),round(box[1]+box[3]))
                        red_pos.append(bottom_right_corner)
                        cv2.circle(img_final,bottom_right_corner,radius=2,color=(0,0,0),thickness=3)
                else:
                    #Draw White Box
                    label = str(f"player{white_number}")
                    cv2.rectangle(img_final, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 255, 255), 2)
                    cv2.putText(img_final, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #Increment White Defenders
                    white_number+=1

                    if goalDirection== 'left':
                        bottom_left_corner=(round(box[0]),round(box[1]+box[3]))
                        white_pos.append(bottom_left_corner)
                        cv2.circle(img_final,bottom_left_corner,radius=2,color=(0,0,0),thickness=3)
                    else:
                        bottom_right_corner=(round(box[0]+box[2]),round(box[1]+box[3]))
                        white_pos.append(bottom_right_corner)
                        cv2.circle(img_final,bottom_right_corner,radius=2,color=(0,0,0),thickness=3)
            
    # save image with identified people highlighted
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(1,1,1)
    # remove axis
    ax1.axis('off')
    ax1.imshow(img_final)
    plt.savefig('Output/'+str(im)+'Box.jpg', bbox_inches='tight')
    plt.title("detection")

    return red_pos,white_pos,img_final

def get_cropped_image(img,box:list) -> list:
    crop=img[round(box[1]):round(box[1]+box[3]),round(box[0]):round(box[0]+box[2])] 
    return crop

def get_red_mask(cropped_img):
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower1 = np.array([0, 50, 50])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([170,50,50])
    upper2 = np.array([180,255,255])

    lower_mask = cv2.inRange(cropped_img, lower1, upper1)
    upper_mask = cv2.inRange(cropped_img, lower2, upper2)

    full_mask = lower_mask + upper_mask

    result = cv2.bitwise_and(cropped_img, cropped_img, mask=full_mask)
    return result

def get_white_mask(cropped_img):
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # white mask, used for 0.jpg, 24.jpg
    #lower = np.array([0, 0, 231])
    #upper = np.array([180, 18, 255])
    # blue mask, used for 479.jpg
    lower = np.array([90, 50, 70])
    upper = np.array([128, 255, 255])
    mask = cv2.inRange(cropped_img,lower,upper)
    result = cv2.bitwise_and(cropped_img,cropped_img,mask=mask)
    return result

#COLOR RANGES IN HSV: {color: upper, lower}
# color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
#                   'white': [[180, 18, 255], [0, 0, 231]],
#                   'red1': [[180, 255, 255], [159, 50, 70]],
#                   'red2': [[9, 255, 255], [0, 50, 70]],
#                   'green': [[89, 255, 255], [36, 50, 70]],
#                   'blue': [[128, 255, 255], [90, 50, 70]],
#                   'yellow': [[35, 255, 255], [25, 50, 70]],
#                   'purple': [[158, 255, 255], [129, 50, 70]],
#                   'orange': [[24, 255, 255], [10, 50, 70]],
#                   'gray': [[180, 18, 230], [0, 0, 40]]}