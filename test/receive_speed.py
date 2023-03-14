from datetime import datetime

import cv2
import simplejpeg

from webcam_client import imagezmq

USE_JPEG = True

with imagezmq.ImageHub(REQ_REP=True) as image_hub:
    while True:  # show streamed images until Ctrl-C
        if USE_JPEG:
            meta_data, jpg_buffer = image_hub.recv_jpg()
            image = simplejpeg.decode_jpeg(jpg_buffer, colorspace='BGR')
        else:
            meta_data, image = image_hub.recv_image()
        now = datetime.now()
        current_time = now.strftime("%M:%S.%f")

        timestep = meta_data["timestep"]
        diff = now - datetime.fromtimestamp(timestep)
        print(f"Image shape: {image.shape}, duration: {diff.total_seconds()}s")
        cv2.imshow("test", image)  # 1 window for each RPi
        cv2.waitKey(1)
        image_hub.send_reply(b'OK')
