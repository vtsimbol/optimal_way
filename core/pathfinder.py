import numpy as np


class Pathfinder:
    def __init__(self, stacks_mask: np.ndarray):
        self._stacks_mask = stacks_mask
        self._found_ways = []

    def start(self, point1, point2):
        # x0, y0 = point1
        # x1, y1 = point2
        # if self._stacks_mask[x0, y0] == 1 or self._stacks_mask[x1, y1] == 1:
        #     raise ValueError('Incorrect point')

        self._found_ways = []
        previous_steps_mask = self._stacks_mask.copy()
        self._find_way(point1, point2, previous_steps_mask, [])

        number_of_points = [len(w) for w in self._found_ways]
        best_way_idx = np.argmin(number_of_points)
        best_way = self._found_ways[best_way_idx]
        return self._found_ways, best_way

    def _get_neighboring_points(self, point, used_points_mask: np.ndarray):
        xc, yc = point
        available_pts = ~np.logical_or(used_points_mask, self._stacks_mask)
        if np.sum(available_pts) == 0:
            return None

        x0, y0 = np.maximum(xc - 1, 0), np.maximum(yc - 1, 0)
        x1, y1 = np.minimum(xc + 1, self._stacks_mask.shape[0] - 1), np.minimum(yc + 1, self._stacks_mask.shape[1] - 1)
        roi = available_pts[x0:x1 + 1, y0:y1 + 1]

        x_indices, y_indices = np.where(roi)
        points = [(x + x0, y + y0) for x, y in zip(x_indices, y_indices)]
        return points

    @staticmethod
    def _get_next_step(point, direction, short_way=True):
        x0, y0 = point
        if short_way:
            if np.abs(direction[0]) > np.abs(direction[1]):
                if direction[0] < 0:
                    # ←←←
                    x1, y1 = x0 - 1, y0
                else:
                    # →→→
                    x1, y1 = x0 + 1, y0
            else:
                if direction[1] < 0:
                    # ↑↑↑
                    x1, y1 = x0, y0 - 1
                else:
                    # ↓↓↓
                    x1, y1 = x0, y0 + 1
        else:
            if np.abs(direction[0]) < np.abs(direction[1]):
                if direction[0] < 0:
                    # ←←←
                    x1, y1 = x0 - 1, y0
                else:
                    # →→→
                    x1, y1 = x0 + 1, y0
            else:
                if direction[1] < 0:
                    # ↑↑↑
                    x1, y1 = x0, y0 - 1
                else:
                    # ↓↓↓
                    x1, y1 = x0, y0 + 1
        return x1, y1

    def _find_way(self, start_point, finish_point, previous_steps_mask, way: list):
        way.append(start_point)

        if start_point[0] == finish_point[0] and start_point[1] == finish_point[1]:
            self._found_ways.append(way)
            return

        x0, y0 = start_point
        previous_steps_mask[x0, y0] = True

        available_points = self._get_neighboring_points(start_point, previous_steps_mask)
        if available_points is None:
            return

        for point in available_points:
            x, y = point

            if not ((np.abs(x - x0) == 1 and np.abs(y - y0) == 0) or
                    (np.abs(x - x0) == 0 and np.abs(y - y0) == 1)):
                continue

            previous_steps_mask[x, y] = True
            self._find_way(point, finish_point, previous_steps_mask.copy(), way.copy())


if __name__ == '__main__':
    stacks_mask = np.asarray([
        [False, False, False, False, False],
        [False, False, True, False, False],
        [False, False, True, False, False],
        [False, False, False, False, False],
        [False, True, False, True, False],
        [False, True, False, True, False],
        [False, False, False, False, False]
    ], dtype=bool)

    point1, point2 = (0, 0), (6, 4)

    pathfinder = Pathfinder(stacks_mask=stacks_mask)
    pathfinder.start(point1, point2)
