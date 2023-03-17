from argparse import ArgumentParser

import numpy as np

from webcam_client.detector_client import WebcamClient


def parse_args():
    description = '''\
        Launch a webcam client that can communicate with a server based on ZeroMQ.
        It will stream the input from the webcam attached on your computer, e.g. inbuilt camera of MacBook.
        --------------------------------
        Example: python3 run_detector_client --host "137.110.198.230" --mac -v
        '''
    parser = ArgumentParser(description)
    parser.add_argument("--device", "-d", required=False, default=0, type=str,
                        help="Device name of the webcam in OpenCV VideoCapture format. "
                             "Can be either a int number like 0 "
                             "or a path to the WebCam device on Linux like '/dev/video0'")
    parser.add_argument("--host", required=True, type=str,
                        help="The server host address, either IP address or domain name.")
    parser.add_argument("--port", "-p", required=False, default=5555, type=int,
                        help="The port number for server to receive the webcam stream from ZeroMQ. ")
    parser.add_argument("--mac", "-m", required=False, action="store_true", default=True,
                        help="The server host address, either IP address or domain name.")
    parser.add_argument("--verbose", "-v", required=False, action="store_true", default=False,
                        help="The server host address, either IP address or domain name.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.mac:
        camera_mat = np.array([[955.57, 0., 650.67], [0., 955.30, 349.25], [0., 0., 1.]])
    else:
        camera_mat = np.array([[606.29937744, 0., 317.60064697], [0., 606.19647217, 229.66906738], [0., 0., 1.]])

    with WebcamClient(
        camera_mat=camera_mat,
        image_host=args.host,
        image_port=args.port,
        verbose=args.verbose,
        device=args.device,
        use_jpg=True) as client:
        while True:
            client.send()


if __name__ == '__main__':
    main()
