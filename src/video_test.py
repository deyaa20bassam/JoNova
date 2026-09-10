import cv2
import os

video_path = r"C:\Users\User\PycharmProjects\JoNova\dataset\raw\KArSL\01\0152\03_01_0152_(06_04_17_16_54_14)_c.mp4"

print("Video path:", video_path)
print("File exists:", os.path.exists(video_path))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video!")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video reached.")
        break

    cv2.imshow("JoNova", frame)

    if cv2.waitKey(25) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()