"""Setup script """

import os

from setuptools import setup, find_packages


def collect_files(target_dir):
    file_list = []
    for (root, dirs, files) in os.walk(target_dir, followlinks=True):
        for filename in files:
            file_list.append(os.path.join('..', root, filename))
    return file_list


def setup_package():
    root_dir = os.path.dirname(os.path.realpath(__file__))

    packages = find_packages(".")
    print(packages)

    setup(name='image_zmq_client',
          version='0.1.0',
          description='Image client for hand detection using zmq',
          author='Yuzhe Qin',
          author_email='y1qin@ucsd.edu',
          url='https://github.com/yzqin/webcam_client',
          license='MIT',
          packages=packages,
          python_requires='>=3.6,<3.11',
          install_requires=[
              # General dependencies
              "pyzmq",
              "simplejpeg",
              "opencv-python",
              "mediapipe",
          ],
          )


setup_package()
