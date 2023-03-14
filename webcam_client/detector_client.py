import time
from datetime import datetime
from typing import Union, Optional

import cv2
import numpy as np
import simplejpeg

from webcam_client.bbox_detector import SingleHandDetector
from webcam_client.imagezmq import ImageSender


class WebcamClient:
    def __init__(self, device: Union[str, int] = 0, hand_type="right_hand", image_host="localhost",
                 image_port: int = 5555, use_jpg=True, verbose=False):
        hand_dict = {"left_hand": "Left", "right_hand": "Right"}
        self.connection_address = f"tcp://{image_host}:{image_port}"
        self.verbose = verbose
        self.use_jpg = use_jpg
        self.hand_type = hand_type

        self.video_capture = cv2.VideoCapture(device)
        self.detector = SingleHandDetector(hand_type=hand_dict[hand_type])
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

        meta_data = {"hand_type": self.hand_type, "bbox": bbox[0].tolist(), "timestep": now.timestamp()}
        image_cropped = self.detector.crop_bbox_on_image(image_bgr, bbox[0])

        if self.use_jpg:
            tic = time.time()
            jpg_buffer = simplejpeg.encode_jpeg(np.ascontiguousarray(image_cropped), quality=95, colorspace='BGR')
            tac = time.time()
            if self.verbose:
                print(f"Jpeg encode time: {tac - tic}s, resolution: {image.shape[:2]}")
            self.sender.send_jpg(meta_data, jpg_buffer)
        else:
            self.sender.send_image(meta_data, image_cropped)


def main():
    with WebcamClient(image_host="137.110.198.230", verbose=True, device="/dev/video4") as client:
        while True:
            client.send()


if __name__ == '__main__':
    main()
