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
        cv2.imwrite('masked.jpg', target)
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
                        print(x1, y1, x2, y2)
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

    cv2.imshow("Model Image V", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
                        # cv2.line(image,(x1,y1), (x2,y2), (0,0,255),1)
        if len(finalLines) < 2:
            if rLimit >= 75:
                rLimit -= 10
            else:
                angleMinLimit -= 1
                angleMaxLimit += 1
        else:
            linesFound = True

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


def get_angle_points(vanishing_point, test_point, img, goalDirection):
    reference_point = 0.0, vanishing_point[1]
    a = np.array(reference_point)
    b = np.array(vanishing_point)
    c = np.array(test_point)

    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    if goalDirection == 'left':
        if reference_point[0] > vanishing_point[0]:
            angle = -1 * angle
    if goalDirection == 'right':
        if reference_point[0] < vanishing_point[0]:
            angle = -1 * angle

    return


def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # cv2.line(img , (int(a[0]),int(a[1])) , (int(b[0]),int(b[1])) , (0,0,255) , 2 )
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    return angle


def get_ground_plane(img, vertical_vanishing_point, horizontal_vanishing_point, side):
    verticalLines = get_vertical_lines(img, side)
    horizontalLines = get_horizontal_lines(img)

    # loop until we find intersections between horizontal & vertical
    for i in range(len(verticalLines)):
        for j in range(len(horizontalLines)):
            finalLines = [verticalLines[i], horizontalLines[j]]
            if find_intersections(finalLines):
                break
        if find_intersections(finalLines):
            break

    print(finalLines)
    intersections = find_intersections(finalLines)
    thirdPoint = intersections[0]
    x1 = vertical_vanishing_point[0]
    y1 = vertical_vanishing_point[1]
    z1 = 0
    x2 = horizontal_vanishing_point[0]
    y2 = horizontal_vanishing_point[1]
    z2 = 0
    x3 = thirdPoint[0]
    y3 = thirdPoint[1]
    z3 = 0

    print(vertical_vanishing_point)
    print(horizontal_vanishing_point)
    print(thirdPoint)

    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (-a * x1 - b * y1 - c * z1)

    return [a, b, c, d]


def project_point_on_plane(plane, point, pose, ratio, image):
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]
    if 'rightShoulder' in pose[2].keys() and 'rightHip' in pose[2].keys() and 'leftShoulder' in pose[
        2].keys() and 'leftHip' in pose[2].keys():
        rightUpperBody = (math.sqrt((pose[2]['rightShoulder']['y'] - pose[2]['rightHip']['y']) ** 2 + (
                    pose[2]['rightShoulder']['x'] - pose[2]['rightHip']['x']) ** 2))
        leftUpperBody = (math.sqrt((pose[2]['leftShoulder']['y'] - pose[2]['leftHip']['y']) ** 2 + (
                    pose[2]['leftShoulder']['x'] - pose[2]['leftHip']['x']) ** 2))
        approx_z = ratio * ((rightUpperBody + leftUpperBody) / 2)
        temp_pt = [pose[2]['leftHip']['y'], 0.0]
        angle = get_angle([pose[2]['leftShoulder']['y'], pose[2]['leftShoulder']['x']],
                          [pose[2]['leftHip']['y'], pose[2]['leftHip']['x']], temp_pt)
    elif 'rightShoulder' in pose[2].keys() and 'rightHip' in pose[2].keys():
        if 'leftShoulder' not in pose[2].keys() or 'leftHip' not in pose[2].keys():
            approx_z = ratio * (math.sqrt((pose[2]['rightShoulder']['y'] - pose[2]['rightHip']['y']) ** 2 + (
                        pose[2]['rightShoulder']['x'] - pose[2]['rightHip']['x']) ** 2))
            temp_pt = [pose[2]['rightHip']['y'], 0.0]
            angle = get_angle([pose[2]['rightShoulder']['y'], pose[2]['rightShoulder']['x']],
                              [pose[2]['rightHip']['y'], pose[2]['rightHip']['x']], temp_pt)
    elif 'leftShoulder' in pose[2].keys() and 'leftHip' in pose[2].keys():
        if 'rightShoulder' not in pose[2].keys() or 'rightHip' not in pose[2].keys():
            approx_z = ratio * (math.sqrt((pose[2]['leftShoulder']['y'] - pose[2]['leftHip']['y']) ** 2 + (
                        pose[2]['leftShoulder']['x'] - pose[2]['leftHip']['x']) ** 2))
            temp_pt = [pose[2]['leftHip']['y'], 0.0]
            angle = get_angle([pose[2]['leftShoulder']['y'], pose[2]['leftShoulder']['x']],
                              [pose[2]['leftHip']['y'], pose[2]['leftHip']['x']], temp_pt)

    angle = (angle * 22) / (7 * 180)
    final_z = np.cos(angle) * approx_z
    x1 = point[0]
    y1 = point[1]
    z1 = final_z
    d = abs((a * x1 + b * y1 + c * z1 + d))
    e = (math.sqrt(a * a + b * b + c * c))
    perpendicular_dist = d / e
    y_new = y1 + perpendicular_dist

    return [int(y_new), int(point[0])]


if __name__ == "__main__":
    # Direction of goal
    goalDirection = 'right'

    imageForVanishingPoints = cv2.imread("/Users/alexalzaga/Documents/UNI/COMPUTER VISION/FinalProject/images/OG.jpeg")
    vertical_vanishing_point = get_vanishing_point_v(imageForVanishingPoints, goalDirection)
    horizontal_vanishing_point = get_vanishing_point_h(imageForVanishingPoints)
    # cv2.imwrite(vanishing_point_viz_base_path+tempFileNames[file_itr], imageForVanishingPoints)
    print('Finished Vanishing Point calculation')
