from typing import NamedTuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions import hands_connections
from mediapipe.python.solutions.drawing_utils import DrawingSpec
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
    def draw_skeleton_on_image(image, keypoint_2d, style="white"):
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
