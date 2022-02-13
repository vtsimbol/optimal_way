import argparse
import json

import numpy as np

from utils import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True, help='The path to the warehouse annotation')
args = parser.parse_args()


class WayFinder:
    def __init__(self, anno_path: str):
        warehouse_anno = self._load(anno_path)
        warehouse_data = self._init_warehouse(warehouse_anno)

        self._visualizer = Visualizer(grid_step=10)
        self._stacks = warehouse_data['stacks_mask']
        self._item_points = warehouse_data['item_points']
        self._entry_point = warehouse_data['entry_point']
        self._exit_point = warehouse_data['exit_point']

    @staticmethod
    def _load(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def _init_warehouse(data):
        grid_size = data['grid_size']
        entry_point = np.asarray(data['entry_point'], dtype=int)
        exit_point = np.asarray(data['exit_point'], dtype=int)
        stacks_mask = np.asarray(data['stacks_mask'], dtype=bool)
        item_points = np.asarray(data['item_points'], dtype=int)

        stacks_x, stacks_y = np.where(stacks_mask)
        stacks_min_x, stacks_min_y = np.min(stacks_x), np.min(stacks_y)
        stacks_max_x, stacks_max_y = np.max(stacks_x), np.max(stacks_y)

        points_min_x = np.minimum(entry_point[0], exit_point[0])
        points_max_x = np.maximum(entry_point[0], exit_point[0])
        points_min_y = np.minimum(entry_point[1], exit_point[1])
        points_max_y = np.maximum(entry_point[1], exit_point[1])

        x_min, x_max = np.minimum(points_min_x, stacks_min_x), np.maximum(points_max_x, stacks_max_x)
        y_min, y_max = np.minimum(points_min_y, stacks_min_y), np.maximum(points_max_y, stacks_max_y)
        if x_min > 0:
            x_min -= 1
        if y_min > 0:
            y_min -= 1
        if x_max < grid_size:
            x_max += 1
        if y_max < grid_size:
            y_max += 1

        entry_point[0] -= x_min
        entry_point[1] -= y_min
        exit_point[0] -= x_min
        exit_point[1] -= y_min
        for i in range(len(item_points)):
            item_points[i][0] -= x_min
            item_points[i][1] -= y_min

        return {
            'grid_size': [x_max - x_min + 1, y_max - y_min + 1],
            'stacks_mask': stacks_mask[x_min:x_max, y_min:y_max],
            'entry_point': entry_point,
            'exit_point': exit_point,
            'item_points': item_points
        }

    def __call__(self):
        pass


if __name__ == '__main__':
    finder = WayFinder(anno_path=args.input_file)
    finder()
