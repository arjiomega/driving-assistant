import numpy as np
import cv2

POINT = tuple[int, int]
VERTICES = list[POINT, POINT, POINT, POINT] # ORDERED
COLOR_FORMAT =  tuple[int, int, int]

class Boundary:
    
    def __init__(self, top_left: POINT, top_right: POINT, bottom_left: POINT, bottom_right: POINT, frame_center: POINT):
        self.frame_center = frame_center

        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right

        self.vertices = self.get_vertices(top_left, top_right, bottom_left, bottom_right)

        self.top_height = top_left[1]
        self.bottom_height = bottom_left[1]

        self.occupied_height = top_left[1]

    @classmethod
    def from_frame_shape(cls, height, width):

        center_x = width // 2
        top_bottom_ratio = 0.3

        bottom_height = height-height//7
        top_height = height-400

        return cls(
            top_left=(center_x - int((center_x // 2) * top_bottom_ratio), top_height),
            top_right=(center_x + int((center_x // 2) * top_bottom_ratio), top_height),
            bottom_left=(center_x - center_x // 2, bottom_height),
            bottom_right=(center_x + center_x // 2, bottom_height),
            frame_center=(width // 2, height // 2)
        )

    def get_vertices(self, top_left: POINT, top_right: POINT, bottom_left: POINT, bottom_right: POINT):
        """returns an ordered points"""
        return [top_left, top_right, bottom_right, bottom_left]

    def _draw_boundary(self, frame: np.ndarray, vertices: VERTICES, color: COLOR_FORMAT, draw_edge=False):
        vertices = np.array([vertices], dtype=np.int32)
        cv2.fillPoly(frame, vertices, color)
        if draw_edge:
            cv2.polylines(frame, vertices, isClosed=True, color=(255, 255, 255), thickness=2)

    def _draw_occupied(self, frame: np.ndarray, occupied_height: int):
        RED_COLOR = (255, 0, 0)
        dx_top = self.frame_center[0] - self.top_left[0]
        dx_bottom = self.frame_center[0] - self.bottom_left[0]
        dx_segment = int(
            (
                (dx_top - dx_bottom) / (self.top_height - self.bottom_height)
            ) * (occupied_height - self.bottom_height) + dx_bottom
        )
        occupied_vertices = self.get_vertices(
            top_left=self.top_left, 
            top_right=self.top_right,
            bottom_left=(self.frame_center[0] - dx_segment, occupied_height),
            bottom_right=(self.frame_center[0] + dx_segment, occupied_height)
        )

        self._draw_boundary(frame=frame, vertices=occupied_vertices, color=RED_COLOR)

    def overlay_boundary(self, frame: np.ndarray, occupied_height=None, color: COLOR_FORMAT=(204, 255, 255), alpha=0.5):
        overlay = frame.copy()

        # draw base boundary
        self._draw_boundary(overlay, self.vertices, color)

        # draw occupied space in boundary
        if occupied_height:
            self._draw_occupied(overlay, occupied_height)

        # Add transparency
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame
    
    def _is_point_inside_boundary(self, point):
        return cv2.pointPolygonTest(np.array(self.vertices, dtype=np.int32), tuple(point), False) >= 0

    def is_bounding_box_inside_boundary(self, top_left: POINT, top_right: POINT, bottom_left: POINT, bottom_right: POINT):
        
        bounding_box_vertices = self.get_vertices(top_left, top_right, bottom_left, bottom_right)

        for point in bounding_box_vertices:
            if self._is_point_inside_boundary(point):
                return True
        return False