from cvzone.PoseModule import PoseDetector
import cv2

cap = cv2.VideoCapture(0)

detector = PoseDetector()

while True:
    success, img = cap.read()

    if not success:
        break

    img = detector.findPose(img)

    lmList, bboxInfo = detector.findPosition(img)

    if lmList:
        print("Pose detected")

    cv2.imshow("JoNova Pose Detection", img)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()