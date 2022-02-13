import numpy as np


class Pathfinder:
    def __init__(self, stacks_mask: np.ndarray):
        self._stacks_mask = stacks_mask

    def find_way(self, point1, point2):
        x0, y0 = point1
        x1, y1 = point2
        if self._stacks_mask[x0, y0] == 1 or self._stacks_mask[x1, y1] == 1:
            raise ValueError('Incorrect point')
