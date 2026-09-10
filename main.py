import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque

# =============================================
# 1. تحضير MediaPipe
# =============================================
from mediapipe.solutions import holistic, drawing_utils

holistic_model = holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True
)

POSE_CONNECTIONS = holistic.POSE_CONNECTIONS
HAND_CONNECTIONS = holistic.HAND_CONNECTIONS

# =============================================
# 2. إعدادات النظام
# =============================================
SEQUENCE_LENGTH = 30
LANDMARKS_PER_FRAME = 1662

frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
is_recording = False
current_gesture = None
recorded_gestures = {}


# =============================================
# 3. دالة استخراج النقاط
# =============================================
def extract_landmarks(frame, results):
    """
    استخراج جميع النقاط من الصورة
    """
    landmarks = []

    # نقاط الجسم (33 نقطة)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

    # نقاط اليد اليسرى (21 نقطة)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

    # نقاط اليد اليمنى (21 نقطة)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

    # نقاط الوجه (468 نقطة)
    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks)


# =============================================
# 4. فتح الكاميرا
# =============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("=" * 50)
print("🎥 نظام ترجمة لغة الإشارة - وضع جمع البيانات")
print("=" * 50)
print("\nالتعليمات:")
print("  'R' - ابدأ التسجيل لإشارة جديدة")
print("  'S' - توقف التسجيل والحفظ")
print("  'Q' - خروج")
print("\n")

# =============================================
# 5. حلقة المعالجة الرئيسية
# =============================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(rgb_frame)

    # رسم النقاط
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        drawing_utils.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            HAND_CONNECTIONS
        )

    landmarks = extract_landmarks(frame, results)

    if is_recording:
        frame_sequence.append(landmarks)

        cv2.putText(
            frame,
            f"REC: {current_gesture} ({len(frame_sequence)}/{SEQUENCE_LENGTH})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        if len(frame_sequence) == SEQUENCE_LENGTH:
            print(f"✅ اكتملت الإشارة '{current_gesture}' - اضغط S للحفظ")

    cv2.putText(
        frame,
        "Press 'R' to record, 'S' to save, 'Q' to quit",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1
    )

    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n👋 وداعاً!")
        break

    elif key == ord('r'):
        if not is_recording:
            gesture_name = input("\n🎯 أدخل اسم الإشارة: ")
            current_gesture = gesture_name
            frame_sequence.clear()
            is_recording = True
            print(f"▶️  بدأ التسجيل للإشارة: {gesture_name}")

    elif key == ord('s'):
        if is_recording and len(frame_sequence) == SEQUENCE_LENGTH:
            if current_gesture not in recorded_gestures:
                recorded_gestures[current_gesture] = []

            recorded_gestures[current_gesture].append(list(frame_sequence))

            print(f"💾 تم حفظ نموذج من '{current_gesture}' (إجمالي: {len(recorded_gestures[current_gesture])})")

            is_recording = False
            frame_sequence.clear()
        else:
            print("⚠️  لا يمكن الحفظ - التسجيل لم يكتمل أو لم يبدأ!")

cap.release()
cv2.destroyAllWindows()
holistic_model.close()

if recorded_gestures:
    os.makedirs('data', exist_ok=True)
    with open('data/gestures.pkl', 'wb') as f:
        pickle.dump(recorded_gestures, f)
    print(f"\n✅ تم حفظ {len(recorded_gestures)} إشارة في 'data/gestures.pkl'")
else:
    print("\n⚠️  لم يتم تسجيل أي إشارات!")
