from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]

        lmList = hand["lmList"]

        print(lmList[0])  # أول نقطة

    cv2.imshow("JoNova", img)

    if cv2.waitKey(1) == 27:
        break