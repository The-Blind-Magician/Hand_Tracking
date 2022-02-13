import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm

WRIST, THUMB_1, THUMB_2, THUMB_3, THUMB_4, INDEX_1, INDEX_2, INDEX_3, INDEX_4, MIDDLE_1, MIDDLE_2, MIDDLE_3, MIDDLE_4 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
RING_1, RING_2, RING_3, RING_4, PINKY_1, PINKY_2, PINKY_3, PINKY_4 = 13, 14, 15, 16, 17, 18, 19, 20

def round_to_multiple(number, multiple):
    return multiple * round(number/multiple)

def get_distance_midpoint(x, y, multiple):
    dist = int(np.linalg.norm(np.array(x) - np.array(y)))
    dist = round_to_multiple(dist, multiple)
    midpoint = [int((x[0] + y[0]) / 2), int((x[1] + y[1]) / 2)]
    return dist, midpoint

def draw_line(img, lmList, a, b, color):
    cv2.line(img, [lmList[a][1], lmList[a][2]], [lmList[b][1], lmList[b][2]], color, 2)
    return

def get_two_points(lmList, a, b):
    return [lmList[a][1], lmList[a][2]], [lmList[b][1], lmList[b][2]]

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

ptime = 0
detector = htm.handDetection()
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img, numHands = detector.findHands(img)
    hand1 = detector.findPosition(img, 0)
    hand2 = []
    if(numHands == 2):
        hand2 = detector.findPosition(img, 1)

    currentTime = time.time()
    fps = 1 / (currentTime - ptime)
    ptime = currentTime

    if len(hand1) != 0:
        a1, b1 = get_two_points(hand1, THUMB_4, INDEX_4)
        draw_line(img, hand1, THUMB_4, INDEX_4, detector.mpDraw.RED_COLOR)
        dist1, midpoint1 = get_distance_midpoint(a1, b1, 5)

        cv2.putText(img, str(dist1), (midpoint1[0], midpoint1[1]), cv2.FONT_HERSHEY_PLAIN, 1, (detector.mpDraw.BLUE_COLOR), 2)
        cv2.circle(img, (midpoint1[0], midpoint1[1]), 2, detector.mpDraw.RED_COLOR, 3)

    if len(hand2) != 0:
        a2, b2 = get_two_points(hand2, THUMB_4, INDEX_4)
        a3, b3 = get_two_points(hand2, INDEX_1, WRIST)
        draw_line(img, hand2, THUMB_4, INDEX_4, detector.mpDraw.RED_COLOR)
        dist2, midpoint2 = get_distance_midpoint(a2, b2, 5)

        dist3, midpoint3 = get_distance_midpoint(a3, b3, 5)
        cv2.putText(img, str(dist3), (midpoint3[0], midpoint3[1]), cv2.FONT_HERSHEY_PLAIN, 1, (detector.mpDraw.BLUE_COLOR), 2)

        cv2.putText(img, str(dist2), (midpoint2[0], midpoint2[1]), cv2.FONT_HERSHEY_PLAIN, 1, (detector.mpDraw.BLUE_COLOR), 2)
        cv2.circle(img, (midpoint2[0], midpoint2[1]), 2, detector.mpDraw.RED_COLOR, 3)

    cv2.putText(img, str(int(fps)), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (detector.mpDraw.GREEN_COLOR), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

