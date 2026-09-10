import cv2

video_path = r"C:\Users\User\PycharmProjects\JoNova\dataset\raw\KArSL\01\0152\03_01_0152_(06_04_17_16_54_14)_c.mp4"


cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("JoNova Video", frame)

    if cv2.waitKey(25) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()