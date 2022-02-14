import numpy as np

from way import PointType, Way


class Pathfinder:
    def __init__(self, stacks_mask: np.ndarray):
        self._stacks_mask = stacks_mask

    @staticmethod
    def _get_direction(point1, point2):
        x0, y0 = point1
        x1, y1 = point2
        direct = np.asarray([x1 - x0, y1 - y0], dtype=np.float)
        direct /= np.linalg.norm(direct)
        return direct

    def _get_neighboring_points(self, point, used_points_mask: np.ndarray):
        xc, yc = point
        available_pts = ~np.logical_or(used_points_mask, self._stacks_mask)
        if np.sum(available_pts) == 0:
            return None

        x0, y0 = np.maximum(xc - 1, 0), np.maximum(yc - 1, 0)
        x1, y1 = np.minimum(xc + 1, stacks_mask.shape[0] - 1), np.minimum(yc + 1, stacks_mask.shape[1] - 1)
        roi = available_pts[x0:x1 + 1, y0:y1 + 1]

        x_indices, y_indices = np.where(roi)
        points = [(x + x0, y + y0) for x, y in zip(x_indices, y_indices) if not (x + x0 == xc and y + y0 == yc)]
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

    def find_way(self, point1, point2):
        x0, y0 = point1
        x1, y1 = point2
        if self._stacks_mask[x0, y0] == 1 or self._stacks_mask[x1, y1] == 1:
            raise ValueError('Incorrect point')

        used_points_mask = self._stacks_mask.copy()

        way = Way()
        while True:
            if point1[0] == point2[0] and point1[1] == point2[1]:
                print('Finish!')
                break

            direction = self._get_direction(point1, point2)
            available_points = self._get_neighboring_points(point1, used_points_mask)
            if available_points is None:
                raise RuntimeError('Points not found')

            x, y = self._get_next_step(point1, direction, short_way=True)
            if (x, y) in available_points:
                print(f'({x}, {y})')
                way.add_point(cords=(x, y), point_type=PointType.STEP)

                x0, y0 = point1
                used_points_mask[x0, y0] = True
                point1 = (x, y)
                continue

            x, y = self._get_next_step(point1, direction, short_way=False)
            if (x, y) in available_points:
                print(f'({x}, {y})')
                way.add_point(cords=(x, y), point_type=PointType.STEP)

                x0, y0 = point1
                used_points_mask[x0, y0] = True
                point1 = (x, y)
                continue

            raise RuntimeError('LOL!')


if __name__ == '__main__':
    stacks_mask = np.asarray([
        [False, False, False, False, False],
        [False, True, True, True, False],
        [False, True, False, True, False],
        [False, False, False, False, False],
        [False, False, True, True, True]
    ], dtype=bool)

    point1, point2 = (4, 0), (0, 4)

    pathfinder = Pathfinder(stacks_mask=stacks_mask)
    pathfinder.find_way(point2, point1)
