import cv2
import numpy as np
import math


def get_vertical_lines(image, side):
    img_v = image
    finalLines = []
    finalLinesParameters = []

    linesFound = False
    while not linesFound:
        # mask to select football field (green)
        hsv_img = cv2.cvtColor(img_v, cv2.COLOR_BGR2HSV)
        ## mask of green (36,0,0) ~ (70, 255,255)
        mask_g = cv2.inRange(hsv_img, (36, 100, 100), (70, 255, 255))
        target = cv2.bitwise_and(img_v, img_v, mask=mask_g)
        #cv2.imwrite('masked.jpg', target)
        edges = cv2.Canny(target, 155, 250, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 190)
        if lines.any():
            if len(lines) > 2:
                linesFound = True

    linesFound = False

    if side == 'left':
        angleMaxLimit = 20
        angleMinLimit = 70
    else:
        angleMaxLimit = 150
        angleMinLimit = 105

    rLimit = 300
    while linesFound == False:
        for line in lines:
            for r, theta in line:
                isLineValid = True
                a = np.cos(theta)
                b = np.sin(theta)

                # check angle of line for validity
                if (theta * 180 * 7 / 22) > angleMinLimit and (theta * 180 * 7 / 22) < angleMaxLimit:
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))

                    # check validity of lines
                    if len(finalLines) > 0:
                        for lineParams in finalLinesParameters:
                            if abs(lineParams[0] - r) < rLimit:
                                isLineValid = False
                        # if they don't intersect there's no vanishing point
                        for line in finalLines:
                            if not line_intersection(line, [[x1, y1], [x2, y2]]):
                                isLineValid = False
                        # if duplicate we discard
                        if [[x1, y1], [x2, y2]] in finalLines or [[x2, y2], [x1, y1]] in finalLines:
                            isLineValid = False
                    if isLineValid:
                        finalLines.append([[x1, y1], [x2, y2]])
                        #print(x1, y1, x2, y2)
                        finalLinesParameters.append([r, theta])
                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        # cv2.putText(image, str((theta * 180 * 7 / 22)) ,(int((x2))  ,  int((y2))) , cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)
                        # cv2.putText(image, str((theta * 180 * 7 / 22)) ,(int((x1))  ,  int((y1))) , cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 2, cv2.LINE_AA)
        if len(finalLines) < 2:
            if rLimit >= 75:
                rLimit -= 10
            else:
                angleMinLimit -= 1
                angleMaxLimit += 1
                rlimit = 100
        else:
            linesFound = True
    
    #cv2.namedWindow("Model Image V", cv2.WINDOW_NORMAL) 
    #cv2.imshow("Model Image V", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return finalLines


def get_horizontal_lines(image):
    img_h = image
    finalLines = []
    finalLinesParameters = []

    linesFound = False
    while not linesFound:
        hsv_img = cv2.cvtColor(img_h, cv2.COLOR_BGR2HSV)
        mask_g = cv2.inRange(hsv_img, (36, 100, 100), (70, 255, 255))
        target = cv2.bitwise_and(img_h, img_h, mask=mask_g)
        # cv2.imwrite('masked.jpg', target)
        edges = cv2.Canny(target, 155, 250, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 190)
        if lines.any():
            if len(lines) > 2:
                linesFound = True

    linesFound = False
    angleMaxLimit = 120
    angleMinLimit = 0
    rLimit = 200
    while not linesFound:
        for line in lines:
            for r, theta in line:
                isLineValid = True
                a = np.cos(theta)
                b = np.sin(theta)

                if angleMinLimit < (theta * 180 * 7 / 22) < angleMaxLimit:
                    x0 = a * r
                    y0 = b * r
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    if len(finalLines) > 0:
                        for lineParameters in finalLinesParameters:
                            if abs(lineParameters[0] - r) < rLimit:
                                isLineValid = False
                        for line in finalLines:
                            if not line_intersection(line, [[x1, y1], [x2, y2]]):
                                isLineValid = False
                        if [[x1, y1], [x2, y2]] in finalLines or [[x2, y2], [x1, y1]] in finalLines:
                            isLineValid = False
                    if isLineValid:
                        finalLines.append([[x1, y1], [x2, y2]])
                        finalLinesParameters.append([r, theta])
                        cv2.line(image,(x1,y1), (x2,y2), (0,0,255),1)
        if len(finalLines) < 2:
            if rLimit >= 75:
                rLimit -= 10
            else:
                angleMinLimit -= 1
                angleMaxLimit += 1
        else:
            linesFound = True


    #cv2.namedWindow("Model Image H", cv2.WINDOW_NORMAL) 
    #cv2.imshow("Model Image H", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return finalLines


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return [x, y]


def find_intersections(lines):
    intersections = []
    for i, line1 in enumerate(lines):
        for line2 in lines[i + 1:]:
            if not line1 == line2:
                intersection = line_intersection(line1, line2)
                if intersection:
                    intersections.append(intersection)

    return intersections


def get_vanishing_point_v(img, side):
    lines = get_vertical_lines(img, side)
    intersectionPoints = find_intersections(lines)
    vanishingPointX = 0.0
    vanishingPointY = 0.0

    # average the vanishing point
    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]
    
    return (vanishingPointX / len(intersectionPoints), vanishingPointY / len(intersectionPoints))


def get_vanishing_point_h(img):
    lines = get_horizontal_lines(img)
    intersectionPoints = find_intersections(lines)
    vanishingPointX = 0.0
    vanishingPointY = 0.0

    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]

    return (vanishingPointX / len(intersectionPoints), vanishingPointY / len(intersectionPoints))



