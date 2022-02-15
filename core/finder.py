from datetime import datetime
from itertools import permutations, combinations
import json
from multiprocessing import Process, Pipe
import os
import signal
import sys
import time

import numpy as np

from .pathfinder import Pathfinder
from utils import Visualizer


class WayFinder:
    def __init__(self, anno_path: str):
        warehouse_anno = self._load(anno_path)
        warehouse_data = self._init_warehouse(warehouse_anno)

        self._visualizer = Visualizer(grid_step=10)
        self._stacks = warehouse_data['stacks_mask']
        self._walls = warehouse_data['walls_mask']
        self._item_points = warehouse_data['item_points']
        self._entry_point = warehouse_data['entry_point']
        self._exit_point = warehouse_data['exit_point']

        self._pathfinder_mask = np.logical_or(self._stacks, self._walls)
        self._PATHFINDER_WORKER_TIMEOUT = 15 * 60

        # import cv2
        # img = self._visualizer.draw_warehouse(warehouse_data)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)

    @staticmethod
    def _load(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    @staticmethod
    def _init_warehouse(data):
        def get_mask_roi(mask):
            x_cords, y_cords = np.where(mask)
            x0, y0 = np.min(x_cords), np.min(y_cords)
            x1, y1 = np.max(x_cords), np.max(y_cords)
            return [x0, y0, x1, y1]

        def get_items_roi(points):
            points = np.asarray(points.copy())
            x0, y0 = np.min(points[:, 0]), np.min(points[:, 1])
            x1, y1 = np.max(points[:, 0]), np.max(points[:, 1])
            return [x0, y0, x1, y1]

        entry_point = np.asarray(data['entry_point'], dtype=int)
        exit_point = np.asarray(data['exit_point'], dtype=int)
        stacks_mask = np.asarray(data['stacks_mask'], dtype=bool)
        walls_mask = np.asarray(data['walls_mask'], dtype=bool)
        item_points = np.asarray(data['item_points'], dtype=int)

        stacks_roi = get_mask_roi(stacks_mask)
        walls_roi = get_mask_roi(walls_mask)
        if stacks_roi[0] <= walls_roi[0] or stacks_roi[1] <= walls_roi[1] or \
                stacks_roi[2] >= walls_roi[2] or stacks_roi[3] >= walls_roi[3]:
            raise RuntimeError('Incorrect warehouse annotation: stacks outside the walls')

        if not walls_roi[0] <= entry_point[0] <= walls_roi[2] or not walls_roi[1] <= entry_point[1] <= walls_roi[3]:
            raise RuntimeError('Incorrect warehouse annotation: entry point outside the walls')

        if not walls_roi[0] <= exit_point[0] <= walls_roi[2] or not walls_roi[1] <= exit_point[1] <= walls_roi[3]:
            raise RuntimeError('Incorrect warehouse annotation: entry point outside the walls')

        items_roi = get_items_roi(item_points)
        if items_roi[0] < stacks_roi[0] or items_roi[1] < stacks_roi[1] or \
                items_roi[2] > stacks_roi[2] or items_roi[3] > stacks_roi[3]:
            raise RuntimeError('Incorrect warehouse annotation: item points outside the stacks')

        x0, y0, x1, y1 = walls_roi

        entry_point[0] -= x0
        entry_point[1] -= y0
        exit_point[0] -= x0
        exit_point[1] -= y0
        for i in range(len(item_points)):
            item_points[i][0] -= x0
            item_points[i][1] -= y0

        return {
            'grid_size': [x1 - x0 + 1, y1 - y0 + 1],
            'stacks_mask': stacks_mask[x0:x1 + 1, y0:y1 + 1],
            'walls_mask': walls_mask[x0:x1 + 1, y0:y1 + 1],
            'entry_point': entry_point,
            'exit_point': exit_point,
            'item_points': item_points
        }

    def _start_pathfinder_worker(self, start_point: list or tuple, finish_point: list or tuple):
        def pathfinder_worker(point1, point2: list or tuple, mask: np.ndarray, pipe):
            try:
                pathfinder = Pathfinder(mask=mask.copy())
                way = pathfinder(point1.copy(), point2.copy())

                if way is None:
                    pipe.send('Error: Way not found')
                else:
                    pipe.send(way)
            except Exception as e:
                pipe.send(f'Error: {e}')

            while True:
                time.sleep(15)

        parent_pipe, child_pipe = Pipe()
        process = Process(target=pathfinder_worker,
                          args=(start_point, finish_point, self._pathfinder_mask, child_pipe,))
        process.start()
        timestamp = datetime.utcnow()
        return process, parent_pipe, timestamp

    def _poll_worker(self, worker_process, worker_pipe, start_timestamp):
        def stop_process(process):
            if 'linux' in sys.platform:
                os.kill(process.pid, signal.SIGKILL)
            else:
                process.terminate()
            process.join()

        if worker_pipe.poll():
            data = worker_pipe.recv()
            if isinstance(data, str):
                stop_process(worker_process)
                return False, data
            elif isinstance(data, list):
                stop_process(worker_process)
                return True, data

        timestamp = datetime.utcnow()
        if (timestamp - start_timestamp).seconds > self._PATHFINDER_WORKER_TIMEOUT:
            stop_process(worker_process)
            return False, 'Operation timeout'

        return None, None

    def _get_results_from_workers(self, processes, pipes, timestamps):
        num_workers = len(processes)
        ways = [None for _ in range(num_workers)]
        statuses = [False for _ in range(num_workers)]

        while True:
            time.sleep(1)
            for i in range(num_workers):
                if statuses[i]:
                    continue

                ret, data = self._poll_worker(processes[i], pipes[i], timestamps[i])
                if ret is not None:
                    if ret is False:
                        raise RuntimeError('Way not found!')

                    ways[i] = data
                    statuses[i] = True

            if np.sum(statuses) == num_workers:
                break

        return ways

    @staticmethod
    def _check_way(way):
        if way is None or (isinstance(way, list) and len(way) == 0):
            raise RuntimeError('Way not found!')

    def _get_ways_from_one_point_to_others(self, main_point, another_points, reverse: bool):
        processes, pipes, timestamps = [], [], []
        for point in another_points:
            if not reverse:
                process, pipe, timestamp = self._start_pathfinder_worker(main_point, point)
            else:
                process, pipe, timestamp = self._start_pathfinder_worker(point, main_point)

            processes.append(process)
            pipes.append(pipe)
            timestamps.append(timestamp)

        ways = self._get_results_from_workers(processes, pipes, timestamps)

        result = {}
        for i, way in enumerate(ways):
            self._check_way(way)
            result[f'{i}'] = way
        return result

    def _get_ways_between_points(self, points):
        processes, pipes, timestamps = [], [], []
        point_indices = []
        for pts_idx in combinations(list(range(len(points))), 2):
            p1_idx, p2_idx = pts_idx[0], pts_idx[1]
            process, pipe, timestamp = self._start_pathfinder_worker(points[p1_idx], points[p2_idx])

            processes.append(process)
            pipes.append(pipe)
            timestamps.append(timestamp)
            point_indices.append((p1_idx, p2_idx))

        ways = self._get_results_from_workers(processes, pipes, timestamps)

        result = {}
        for pts_idx, way in zip(point_indices, ways):
            self._check_way(way)
            min_idx, max_idx = np.min(pts_idx), np.max(pts_idx)
            result[f'{min_idx}{max_idx}'] = way
            result[f'{max_idx}{min_idx}'] = way

        return result

    def __call__(self, show=True):
        print('Finding the best way from the entry point to each item')
        ways_from_entry_to_item = self._get_ways_from_one_point_to_others(main_point=self._entry_point,
                                                                          another_points=self._item_points,
                                                                          reverse=False)

        print('Finding the best way from the exit point to each item')
        if self._entry_point[0] == self._exit_point[0] and self._entry_point[1] == self._exit_point[1]:
            ways_from_item_to_exit = ways_from_entry_to_item.copy()
        else:
            ways_from_item_to_exit = self._get_ways_from_one_point_to_others(main_point=self._exit_point,
                                                                             another_points=self._item_points,
                                                                             reverse=True)

        print('Finding the best way between items')
        ways_between_items = self._get_ways_between_points(self._item_points)

        print('Getting best way')
        number_of_items = len(self._item_points)
        point_indices = list(range(number_of_items))
        full_ways = []
        metrics = []
        for pts_idx in permutations(point_indices, number_of_items):
            full_ways.append(ways_from_entry_to_item[f'{pts_idx[0]}'])

            for i in range(number_of_items - 1):
                start_point_idx = pts_idx[i]
                finish_point_idx = pts_idx[i + 1]
                full_ways.append(ways_between_items[f'{start_point_idx}{finish_point_idx}'])

            full_ways.append(ways_from_item_to_exit[f'{pts_idx[-1]}'])
            total_points_in_way = 0
            for way in full_ways:
                total_points_in_way += len(way)
            metrics.append(total_points_in_way)

        best_way_idx = np.argmin(metrics)
        best_way = full_ways[best_way_idx]
        print(best_way)
        return best_way
