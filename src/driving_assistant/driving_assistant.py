from typing import Callable
import numpy as np
from ultralytics import YOLO
import cv2

from driving_assistant.boundary import Boundary

class BoundingBox:
    def __init__(self, box, confidence_score, class_name):
        self.center_x, self.center_y, self.width, self.height = box

        self.top_height = int((self.center_y - self.height / 2))
        self.bottom_height = int((self.center_y + self.height / 2))

        self.top_left_x = int((self.center_x - self.width / 2))
        self.top_left_y = self.top_height
        self.bottom_right_x = int((self.center_x + self.width / 2))
        self.bottom_right_y = self.bottom_height

        self.top_left = (self.top_left_x, self.top_left_y)
        self.top_right = (int((self.center_x + self.width / 2)), self.top_left_y)

        self.bottom_left = (int((self.center_x - self.width / 2)), self.bottom_right_y)
        self.bottom_right = (self.bottom_right_x, self.bottom_right_y)

        self.confidence_score = confidence_score
        self.class_name = class_name

        self.inside_boundary = False

    def update_box_boundary_status(self, boundary: Boundary):
        self.inside_boundary = boundary.is_bounding_box_inside_boundary(
            top_left=self.top_left,
            top_right=self.top_right,
            bottom_left=self.bottom_left,
            bottom_right=self.bottom_right
        )

    def overlay_box(self, frame):

        COLOR_RED = (255, 0, 0)
        COLOR_GREEN = (0, 255, 0)
        COLOR_WHITE = (255, 255, 255)

        box_color = COLOR_RED if self.inside_boundary else COLOR_GREEN

        # Draw the rectangle around the object
        cv2.rectangle(frame, self.top_left, self.bottom_right, box_color, 2)

        # Annotate with class name and confidence
        thickness = 2
        label = f"{self.class_name}: {self.confidence_score:.2f}"
        text_position = (self.top_left_x, self.top_left_y - 10)
        cv2.putText(frame, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, thickness)

        return frame

class BoundingBoxes:
    def __init__(self, inference_result, class_names):
        boxes = inference_result[0].boxes.xywh  # (center_x, center_y, width, height)
        confs = inference_result[0].boxes.conf  # Confidence scores
        class_ids = inference_result[0].boxes.cls  # Class IDs

        self.boxes = [
            BoundingBox(box, conf, class_names[int(class_id)]) 
            for box, conf, class_id in zip(boxes, confs, class_ids)
        ]

    def overlay_all_boxes(self, frame):
        for box in self.boxes:
            box.overlay_box(frame)
        return frame
    
    def update_box_boundary_status(self, boundary: Boundary):
        for box in self.boxes:
            box.update_box_boundary_status(boundary)

    def get_closest_box_distance_to_boundary(self, boundary: Boundary):
        closest_box_distance = boundary.top_height # higher distance == closer

        for box in self.boxes:
            if box.inside_boundary:
                if box.bottom_height > closest_box_distance:
                    closest_box_distance = box.bottom_height

        if closest_box_distance == boundary.top_height:
            closest_box_distance = None

        return closest_box_distance

class Inference:
    def __init__(self, model, class_names = ['motorcycle', 'pedestrian', 'vehicle']):
        self.model = model
        self.class_names = class_names

    @classmethod
    def from_path(cls, weights_path):
        return cls(YOLO(weights_path))

    def __call__(self, frame: np.ndarray) -> BoundingBoxes:
        return BoundingBoxes(self.model(frame), self.class_names)


class DrivingAssistant:
    def __init__(self, inference: Inference, boundary: Boundary):
        self.inference = inference
        self.boundary = boundary

    def __call__(self, frame: np.ndarray):
        bounding_boxes = self.inference(frame)
        bounding_boxes.update_box_boundary_status(self.boundary)

        closest_box_distance = bounding_boxes.get_closest_box_distance_to_boundary(self.boundary)

        frame = self.boundary.overlay_boundary(frame, closest_box_distance)
        frame = bounding_boxes.overlay_all_boxes(frame)
        

        return frame        

class VideoLoader:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.height, self.width = self.get_frame_info()

    @classmethod
    def from_path(cls, video_path: str):
        return cls(cv2.VideoCapture(video_path))

    def get_frame_info(self):
        ret, first_frame = self.cap.read()
        if ret:
            height, width, _ = first_frame.shape
            return (height, width)
        else:
            print("Failed to read the video")
            self.cap.release()

    def run(self, process_fn: Callable[[np.ndarray], np.ndarray] = None):
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if not ret:
                break  # Exit loop if no frame is captured

            if process_fn:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = process_fn(frame)

            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow('Driving Assistant Window', frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        self.cap.release()
        cv2.destroyAllWindows()

