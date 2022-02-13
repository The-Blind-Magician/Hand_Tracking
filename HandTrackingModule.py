import cv2
import mediapipe as mp
import time


class handDetection():
    def __init__(self, image_mode=False, max_hands=2, m_complexity=1, min_detection_conf=0.75, min_tracking_conf=0.5):
        self.image_mode = image_mode
        self.max_hands = max_hands
        self.m_complexity = m_complexity
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.image_mode, self.max_hands, self.m_complexity,
                                        self.min_detection_conf, self.min_tracking_conf)  # live update, max hands, confidence detection, confidence tracking
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.mpDraw.DrawingSpec(self.mpDraw.RED_COLOR),
                                               self.mpDraw.DrawingSpec(self.mpDraw.GREEN_COLOR))
            return img, len(self.results.multi_hand_landmarks)
        return img, 0

    def findPosition(self, img, handNum= 0):
        lmList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

        return lmList


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = handDetection()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        currentTime = time.time()
        fps = 1 / (currentTime - ptime)
        ptime = currentTime

        #if len(lmList) != 0:
        #    cv2.line(img, [lmList[4][1], lmList[4][2]], [lmList[8][1], lmList[8][2]], detector.mpDraw.BLUE_COLOR, 2)

        cv2.putText(img, str(int(fps)), (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (detector.mpDraw.GREEN_COLOR), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()