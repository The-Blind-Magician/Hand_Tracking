import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2, 0, 0.75, 0.5) #live update, max hands, confidence detection, confidence tracking
mpDraw = mp.solutions.drawing_utils

ptime = 0
currentTime = 0

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                
            #cv2.line(img, handLms.landmark[0], handLms.landmark[1], mpDraw.BLUE_COLOR)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, mpDraw.DrawingSpec(mpDraw.RED_COLOR), mpDraw.DrawingSpec(mpDraw.GREEN_COLOR))


    currentTime = time.time()
    fps = 1/(currentTime-ptime)
    ptime = currentTime

    cv2.putText(img, str(int(fps)), (5,30), cv2.FONT_HERSHEY_PLAIN, 2, (mpDraw.GREEN_COLOR), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)