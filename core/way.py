from enum import Enum


class Way:
    class PointType(Enum):
        STEP = 0,
        STEP_ROTATE = 1

    class Point:
        def __init__(self, x, y, point_type, weight):
            self.x = x
            self.y = y
            self.type = point_type
            self.weight = weight

    def __init__(self):
        self._points = []
        self._STEP_POINT_WEIGHT = 1.
        self._STEP_ROTATE_POINT_WEIGHT = 1.

    def add_point(self, cords: list or tuple, point_type: PointType):
        x, y = cords
        weight = self._STEP_POINT_WEIGHT if point_type is self.PointType.STEP else self._STEP_ROTATE_POINT_WEIGHT
        point = self.Point(x=x, y=y, point_type=point_type, weight=weight)
        self._points.append(point)

    def get_metrics(self):
        total_weights = 0
        for point in self._points:
            total_weights += point.weight
        return total_weights
