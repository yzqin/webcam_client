import cv2
import mediapipe as mp
import time
from webcam_client.bbox_detector import SingleHandDetector

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture("/dev/video4")
detector = SingleHandDetector()

while cap.isOpened():
    success, image_bgr = cap.read()
    tic = time.time()
    image_bgr.flags.writeable = False
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    num_box, bbox, keypoints_2d, keypoints_3d = detector.detect(image)
    tac = time.time()
    print(f"Detection time: {tac - tic}s, resolution: {image.shape[:2]}")

    if num_box < 1:
        continue
    print("Read hand detection")
    hand_bbox_list = [{"left_hand": None, "right_hand": None}]
    hand_bbox_list[0]["right_hand"] = bbox[0]
    image_bgr = detector.draw_bbox_on_image(image_bgr, bbox)

    cv2.imshow('Test Detection Speed', cv2.flip(image_bgr, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
