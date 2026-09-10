from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=2)

while True:
    success, img = cap.read()

    hands, img = detector.findHands(img, draw=True)

    print(hands)

    cv2.imshow("JoNova", img)

    if cv2.waitKey(1) == 27:
        break