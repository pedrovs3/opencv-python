import cv2
import mediapipe as mp
import time

# Basic to run a webcam
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

 # Verificando se há mão na imagem
    if results.multi_hand_landmarks:
        # Pegando dados individuais de cada mão
        for handLandmarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandmarks.landmark):
                # print(id, landmark)
                heigth, width, channels = img.shape
                cx, cy = int(landmark.x*width), int(landmark.y*heigth)

                print(id, cx, cy)
                # colocando um circulo na ponta do dedão, apartir do cx, cy
                # if id == 4:
                #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)


    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(
        img,
        str(int(fps)),
        (10, 70),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0,255),
        3
    )
    cv2.imshow("Image", img)
    cv2.waitKey(1)