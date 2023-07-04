import time

import cv2
import mediapipe as mp


# Basic to run a webcam
class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5, modelComplexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.modelComplexity = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, int(self.maxHands),
                                        self.modelComplexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        # Verificando se há mão na imagem
        if results.multi_hand_landmarks:
            # Pegando dados individuais de cada mão
            for handLandmarks in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return img


def main():
    previousTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(
            img,
            str(int(fps)),
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (255, 0, 255),
            3
        )
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

    # for id, landmark in enumerate(handLandmarks.landmark):
    # # print(id, landmark)
    # heigth, width, channels = img.shape
    # cx, cy = int(landmark.x * width), int(landmark.y * heigth)
    #
    # print(id, cx, cy)
    #
    #
    # # colocando um circulo na ponta do dedão, apartir do cx, cy
    # # if id == 4:
    # #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
