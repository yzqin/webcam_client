from typing import NamedTuple

import cv2
import mediapipe as mp
import mediapipe.framework as framework
import numpy as np
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.hands import HandLandmark


class MediapipeBBoxHand(SolutionBase):
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        See original mp.solutions.hands.Hands for documentations.
        This class only do one more thing than the original Hands class:
         add hand detection bounding box results into the pipeline with "hand_rects"
        """

        _BINARYPB_FILE_PATH = 'mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.binarypb'
        super().__init__(
            binary_graph_path=_BINARYPB_FILE_PATH,
            side_inputs={
                'model_complexity': model_complexity,
                'num_hands': max_num_hands,
                'use_prev_landmarks': not static_image_mode,
            },
            calculator_params={
                'palmdetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                    min_detection_confidence,
                'handlandmarkcpu__ThresholdingCalculator.threshold':
                    min_tracking_confidence,
            },
            outputs=[
                'multi_hand_landmarks', 'multi_hand_world_landmarks',
                'multi_handedness', 'hand_rects'
            ])

    def process(self, image: np.ndarray) -> NamedTuple:
        """
        See original mp.solutions.hands.Hands for documentations.
        """

        return super().process(input_data={'image': image})


class SingleHandDetector:
    def __init__(self, hand_type="Right", min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hand=1,
                 selfie=False):
        self.hand_detector = MediapipeBBoxHand(
            static_image_mode=False,
            max_num_hands=max_num_hand,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.selfie = selfie
        inverse_hand_dict = {"Right": "Left", "Left": "Right"}
        self.detected_hand_type = hand_type if selfie else inverse_hand_dict[hand_type]

    @staticmethod
    def _mediapipe_bbox_to_numpy(image, hand_rect):
        image_size = np.array(image.shape[:2][::-1])
        center = np.array([hand_rect.x_center, hand_rect.y_center])
        size = np.array([hand_rect.width, hand_rect.height])
        bbox = np.zeros(4)
        bbox[:2] = (center - size / 2) * image_size
        bbox[2:] = size * image_size
        return bbox

    @staticmethod
    def draw_bbox_on_image(image, bboxes,
                           bbox_color=(255, 255, 255),
                           thickness=3):
        image_shape = np.array(image.shape[:2][::-1])
        upper_limit = np.tile(image_shape, 2)
        bboxes[:, 2:] += bboxes[:, :2]
        bboxes = np.clip(bboxes, np.zeros([1, 4]), upper_limit[None]).astype(int)
        for bbox in bboxes:
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          bbox_color, thickness)
        return image

    @staticmethod
    def crop_bbox_on_image(image, bbox):
        image_shape = np.array(image.shape[:2][::-1])
        upper_limit = np.tile(image_shape, 2)
        bbox[2:] += bbox[:2]
        bbox = np.clip(bbox, np.zeros([4]), upper_limit).astype(int)
        return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d: landmark_pb2.NormalizedLandmarkList, style="white"):
        if style == "default":
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())
        elif style == "white":
            landmark_style = {}
            for landmark in HandLandmark:
                landmark_style[landmark] = DrawingSpec(color=(255, 48, 48), circle_radius=4, thickness=-1)

            connections = hands_connections.HAND_CONNECTIONS
            connection_style = {}
            for pair in connections:
                connection_style[pair] = DrawingSpec(thickness=2)

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                keypoint_2d,
                mp.solutions.hands.HAND_CONNECTIONS,
                landmark_style,
                connection_style
            )

        return image

    def detect(self, rgb):
        results = self.hand_detector.process(rgb)
        if not results.multi_hand_landmarks:
            return 0, None, None, None

        desired_hand_num = -1
        for i in range(len(results.multi_hand_landmarks)):
            label = results.multi_handedness[i].ListFields()[0][1][0].label
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None

        bbox = results.hand_rects[desired_hand_num]
        keypoint_3d = results.multi_hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.multi_hand_landmarks[desired_hand_num]
        num_box = len(results.multi_hand_landmarks)
        return num_box, self._mediapipe_bbox_to_numpy(rgb, bbox)[None, :], keypoint_3d, keypoint_2d

    @staticmethod
    def parse_keypoint_3d(keypoint_3d: framework.formats.landmark_pb2.LandmarkList) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d.landmark[i].x
            keypoint[i][1] = keypoint_3d.landmark[i].y
            keypoint[i][2] = keypoint_3d.landmark[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d: landmark_pb2.NormalizedLandmarkList, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d.landmark[i].x
            keypoint[i][1] = keypoint_2d.landmark[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def refill_keypoint_2d(keypoint_2d_array: np.ndarray, img_size):
        keypoints = keypoint_2d_array / np.array([img_size[1], img_size[0]])[None, :]
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(21):
            landmark = landmark_pb2.NormalizedLandmark(x=keypoints[i, 0], y=keypoints[i, 1])
            landmark_list.landmark.append(landmark)
        return landmark_list

    @staticmethod
    def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame
