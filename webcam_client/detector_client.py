import time
from datetime import datetime
from typing import Union, Optional

import cv2

from webcam_client.bbox_detector import SingleHandDetector
from webcam_client.imagezmq import ImageSender


class WebcamClient:
    def __init__(self, device: Union[str, int] = 0, hand_type="Right", image_host="localhost",
                 image_port: int = 5555, verbose=False):
        self.connection_address = f"tcp://{image_host}:{image_port}"
        self.verbose = verbose

        self.video_capture = cv2.VideoCapture(device)
        self.detector = SingleHandDetector(hand_type=hand_type)
        self.sender: Optional[ImageSender] = None

    def __enter__(self):
        self.sender = ImageSender(connect_to=self.connection_address)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sender.close()
        self.video_capture.release()

    def send(self):
        success, image_bgr = self.video_capture.read()
        now = datetime.now()
        tic = time.time()
        image_bgr.flags.writeable = False
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        num_box, bbox, keypoints_2d, keypoints_3d = self.detector.detect(image)
        tac = time.time()
        if self.verbose:
            print(f"Detection time: {tac - tic}s, resolution: {image.shape[:2]}")

        if num_box < 1:
            return None

        hand_bbox_list = [{"left_hand": None, "right_hand": None}]
        hand_bbox_list[0]["right_hand"] = bbox[0]
        image_cropped = self.detector.crop_bbox_on_image(image_bgr, bbox[0])

        if self.verbose:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (0, 30)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            current_time = now.strftime("%M:%S.%f")
            image_cropped = cv2.putText(image_cropped, current_time, org, font, fontScale, color, thickness,
                                        cv2.LINE_AA)

        self.sender.send_image("test", image_cropped)


def main():
    with WebcamClient(image_host="137.110.198.230", verbose=True) as client:
        while True:
            client.send()


if __name__ == '__main__':
    main()
