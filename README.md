# Webcam Client

This library is designed for streaming the webcam attached on the computer, e.g. in-built camera of a MacBook. Then
detect the hand bounding box and send the cropped bounding box to the server. It targets on hand-oriented applications.

The downstream applications includes: VR, teleoperation, etc.

## Installation

```shell
pip3 install git+https://github.com/yzqin/webcam_client.git
```

## Running

```shell
run_webcam_client --host "YOUR_SERVER_IP_OR_DOMAIN_NAME" --verbose --mac
```

## Arguments

```shell
usage: 
Launch a webcam client that can communicate with a server based on ZeroMQ.
It will stream the input from the webcam attached on your computer, e.g. inbuilt camera of MacBook.
--------------------------------
Example: python3 run_detector_client --host "123.234.123.000" --mac -v

       [-h] [--device DEVICE] --host HOST [--port PORT] [--mac] [--verbose]

options:
  -h, --help            show this help message and exit
  --device DEVICE, -d DEVICE
                        Device name of the webcam in OpenCV VideoCapture format. Can be either a int number like 0 or a path to the WebCam device on Linux like '/dev/video0'
  --host HOST           The server host address, either IP address or domain name.
  --port PORT, -p PORT  The port number for server to receive the webcam stream from ZeroMQ.
  --mac, -m             Flag to indicate whether you are using MacBook.
  --verbose, -v         Flag for verbose.
```

