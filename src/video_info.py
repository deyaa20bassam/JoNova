import cv2
from pathlib import Path

dataset_path = Path(
    r"C:\Users\User\PycharmProjects\JoNova\dataset\raw\KArSL\01"
)

first_video = next(dataset_path.rglob("*.mp4"))

print("Video:", first_video)

cap = cv2.VideoCapture(str(first_video))

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print("Frames:", frames)
print("Width:", width)
print("Height:", height)
print("FPS:", fps)

cap.release()