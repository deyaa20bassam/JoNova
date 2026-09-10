import cv2
import csv

# استبدليها لاحقاً بنتائج Hand / Face / Pose الحقيقية
def get_hand_features():
    return [0] * 63  # 21 landmark × 3

def get_face_features():
    return [0] * 30  # مؤقتاً

def get_pose_features():
    return [0] * 99  # 33 landmark × 3

video_path = r"PUT_VIDEO_PATH_HERE"

cap = cv2.VideoCapture(video_path)

with open("landmarks.csv", "w", newline="") as f:
    writer = csv.writer(f)

    while True:
        success, frame = cap.read()

        if not success:
            break

        hand_features = get_hand_features()
        face_features = get_face_features()
        pose_features = get_pose_features()

        combined_features = (
            hand_features
            + face_features
            + pose_features
        )

        writer.writerow(combined_features)

cap.release()

print("Features saved successfully!")