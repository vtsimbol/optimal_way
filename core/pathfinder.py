import numpy as np


class Pathfinder:
    def __init__(self, mask: np.ndarray):
        self._mask = mask
        self._found_ways = []

    def __call__(self, start_point, finish_point):
        self._found_ways = []
        self._find_way(start_point, finish_point, self._mask.copy(), [])

        number_of_points = [len(w) for w in self._found_ways]
        if len(number_of_points) == 0:
            return None

        best_way_idx = np.argmin(number_of_points)
        best_way = self._found_ways[best_way_idx]
        return best_way

    @staticmethod
    def _get_neighboring_points(point, mask: np.ndarray):
        xc, yc = point
        available_pts = ~mask
        if np.sum(available_pts) == 0:
            return None

        x0, y0 = np.maximum(xc - 1, 0), np.maximum(yc - 1, 0)
        x1, y1 = np.minimum(xc + 1, mask.shape[0] - 1), np.minimum(yc + 1, mask.shape[1] - 1)
        roi = available_pts[x0:x1 + 1, y0:y1 + 1]

        x_indices, y_indices = np.where(roi)
        points = [(x + x0, y + y0) for x, y in zip(x_indices, y_indices)]
        return points

    def _find_way(self, start_point, finish_point, mask, way: list):
        way.append(start_point)

        if (np.abs(start_point[0] - finish_point[0]) == 1 and np.abs(start_point[1] - finish_point[1]) == 0) or \
                (np.abs(start_point[0] - finish_point[0]) == 0 and np.abs(start_point[1] - finish_point[1]) == 1):
            self._found_ways.append(way)
            return

        x0, y0 = start_point
        mask[x0, y0] = True

        available_points = self._get_neighboring_points(start_point, mask)
        if available_points is None:
            return

        for point in available_points:
            x, y = point

            if not ((np.abs(x - x0) == 1 and np.abs(y - y0) == 0) or
                    (np.abs(x - x0) == 0 and np.abs(y - y0) == 1)):
                continue

            mask[x, y] = True
            self._find_way(point, finish_point, mask.copy(), way.copy())


if __name__ == '__main__':
    stacks_mask = np.asarray([
       [True,  True,  True,  True,  True,  True,  True,  True,  True],
       [True, False, False, False, False, False, False, False,  True],
       [False, False,  True,  True, False,  True,  True, False,  True],
       [True, False,  True,  True, False,  True,  True, False,  True],
       [True, False,  True,  True, False,  True,  True, False,  True],
       [True, False,  True,  True, False,  True,  True, False,  True],
       [True, False,  True,  True, False,  True,  True, False,  True],
       [True, False, False, False, False, False, False, False,  True],
       [True,  True,  True,  True,  True,  True,  True,  True,  True]
    ], dtype=bool)

    point1, point2 = (2, 0), (3, 3)

    pathfinder = Pathfinder(mask=stacks_mask)
    print(pathfinder(point1, point2))
