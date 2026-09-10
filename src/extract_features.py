from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PoseModule import PoseDetector

import cv2
import csv
from pathlib import Path

# Dataset path
dataset_path = Path(
    r"C:\Users\User\PycharmProjects\JoNova\dataset\raw\KArSL\01"
)

# Output path
output_path = Path(
    r"C:\Users\User\PycharmProjects\JoNova\outputs\features"
)

output_path.mkdir(parents=True, exist_ok=True)

# Detectors
hand_detector = HandDetector(maxHands=2)
face_detector = FaceDetector()
pose_detector = PoseDetector()

# Loop through all sign folders
for class_folder in sorted(dataset_path.iterdir()):

    if not class_folder.is_dir():
        continue

    print(f"\nProcessing {class_folder.name} ...")

    csv_file = output_path / f"{class_folder.name}.csv"

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        videos = list(class_folder.glob("*.mp4"))

        for video in videos:

            cap = cv2.VideoCapture(str(video))

            while True:

                success, img = cap.read()

                if not success:
                    break

                row = []

                # HAND
                hands, img = hand_detector.findHands(img)

                if hands:
                    hand = hands[0]

                    for point in hand["lmList"]:
                        row.extend(point)

                # FACE
                img, bboxs = face_detector.findFaces(img)

                if bboxs:
                    x, y, w, h = bboxs[0]["bbox"]
                    row.extend([x, y, w, h])

                # POSE
                img = pose_detector.findPose(img, draw=False)

                lmList, bboxInfo = pose_detector.findPosition(
                    img,
                    draw=False
                )

                if lmList:
                    for lm in lmList:
                        row.extend(lm[1:4])

                if row:
                    writer.writerow(row)

            cap.release()

    print(f"Finished {class_folder.name}")

print("\nAll feature extraction completed!")