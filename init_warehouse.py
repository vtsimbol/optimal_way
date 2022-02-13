import argparse
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=int, default=50)
parser.add_argument('--img_size', type=int, default=1000)
parser.add_argument('--output_dir', type=str, default=None, help='The path to the directory for saving schemas')
parser.add_argument('--input_file', type=str, default=None, help='The path to the saved warehouse schema')
args = parser.parse_args()


class WarehouseAnnotator:
    class Mode(Enum):
        DRAW_STACK = 'draw stack'
        CLEAR_STACK = 'clear stack'
        SET_ENTRY = 'set entry'
        SET_EXIT = 'set exit'
        SET_ITEMS = 'set items'

    def __init__(self, grid_size: int, img_size: int, save_dir: str, anno_path: str or None):
        if img_size // grid_size == 0:
            raise ValueError('Please, increase image size or decrease grid size')

        self._grid_size = grid_size
        self._img_size = img_size
        self._save_dir = save_dir

        self._grid_step = img_size // grid_size
        self._mask = np.zeros((self._grid_size, self._grid_size), dtype=bool)
        self._entry_point = None
        self._exit_point = None
        self._items_points = []

        if anno_path is not None:
            self._load_anno(file_path=anno_path)

        self._current_mode = self.Mode.DRAW_STACK

        self._last_mouse_point = []
        self._current_mouse_position = None
        self._grid_color = (160, 160, 160)
        self._stack_color = (235, 125, 52)
        self._items_color = (0, 0, 0)
        self._entry_color = (51, 255, 0)
        self._exit_color = (0, 64, 255)
        self._cursor_color = (52, 103, 235)
        self._info_text_color = (222, 252, 255)

        self._select_mode_definitions = {
            self.Mode.DRAW_STACK: '1',
            self.Mode.CLEAR_STACK: '2',
            self.Mode.SET_ENTRY: '3',
            self.Mode.SET_EXIT: '4',
            self.Mode.SET_ITEMS: '5'
        }

        self._logger = logging.getLogger('WarehouseAnnotator')

    def _load_anno(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            anno = json.load(file)

        self._grid_size = anno['grid_size']
        self._entry_point = anno['entry_point']
        self._exit_point = anno['exit_point']
        self._mask = np.asarray(anno['mask'], dtype=bool)
        self._items_points = anno['item_points']

    def start(self):
        self._worker()

    def _mouse_events(self, event, x, y, flags, param):
        if x < 0 or x > self._img_size or y < 0 or y > self._img_size:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._current_mode in [self.Mode.DRAW_STACK, self.Mode.CLEAR_STACK]:
                if len(self._last_mouse_point) >= 2:
                    self._last_mouse_point = []
                self._last_mouse_point.append((x, y))
                self._change_mask()
            elif self._current_mode in [self.Mode.SET_ENTRY, self.Mode.SET_EXIT]:
                self._set_entry_or_exit((x, y))
            elif self._current_mode is self.Mode.SET_ITEMS:
                self._set_items((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            self._current_mouse_position = (x, y)

    def _draw_background_image(self):
        dx, dy = self._img_size + 5, 20
        img = np.ones((self._img_size, self._img_size + 300, 3), dtype=np.uint8)
        img[:, :self._img_size] = 255

        cv2.putText(img, f'--- Select mode ---', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._info_text_color, 1)
        dy += 20

        for mode, char in self._select_mode_definitions.items():
            text = f'{mode.value} - {char}'
            cv2.putText(img, text, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._info_text_color, 1)
            dy += 20

        action_lines = ['', '--- Actions ---', 'save - s', 'exit - q']
        for line in action_lines:
            cv2.putText(img, line, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._info_text_color, 1)
            dy += 20

        return img

    def _draw_info(self, img):
        if self._current_mode is None:
            return

        dx, dy = self._img_size + 5, 240
        lines = ['--- Current mode ---',  f'{self._current_mode.value}']
        for line in lines:
            cv2.putText(img, line, (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._stack_color, 1)
            dy += 20

        dy += 50
        if self._current_mode in [self.Mode.DRAW_STACK, self.Mode.CLEAR_STACK]:
            cv2.putText(img, 'Stacks:', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._stack_color, 1)
            dy += 20
            if len(self._last_mouse_point) == 0:
                cv2.putText(img, 'Select start point', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._stack_color, 1)
            elif len(self._last_mouse_point) == 1:
                cv2.putText(img, 'Select stop point', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._stack_color, 1)

        dy += 50
        if self._entry_point is None:
            cv2.putText(img, 'Entry point:', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._entry_color, 1)
            dy += 20
            cv2.putText(img, 'Set entry point', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._entry_color, 1)

        dy += 50
        if self._exit_point is None:
            cv2.putText(img, 'Exit:', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._exit_color, 1)
            dy += 20
            cv2.putText(img, 'Set exit point', (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._exit_color, 1)

    def _draw_grid(self, img):
        offset = self._grid_step
        for _ in range(self._grid_size):
            cv2.line(img, (offset, 0), (offset, self._img_size), self._grid_color, 1)
            cv2.line(img, (0, offset), (self._img_size, offset), self._grid_color, 1)
            offset += self._grid_step

    def _draw_mouse_position(self, img):
        if self._current_mouse_position is None:
            return img

        x, y = self._current_mouse_position
        grid_row, grid_column = y // self._grid_step, x // self._grid_step
        half_grid_step = self._grid_step // 2

        img = cv2.line(img,
                       pt1=(grid_column * self._grid_step + half_grid_step, 0),
                       pt2=(grid_column * self._grid_step + half_grid_step, self._img_size),
                       color=self._cursor_color, thickness=1)
        img = cv2.line(img,
                       pt1=(0, grid_row * self._grid_step + half_grid_step),
                       pt2=(self._img_size, grid_row * self._grid_step + half_grid_step),
                       color=self._cursor_color, thickness=1)

        if self._current_mode is self.Mode.DRAW_STACK and len(self._last_mouse_point) > 0:
            grid_column0 = self._last_mouse_point[0][0] // self._grid_step
            grid_row0 = self._last_mouse_point[0][1] // self._grid_step
            text = f'({np.abs(grid_column - grid_column0)}, {np.abs(grid_row - grid_row0)})'
            cv2.putText(img, text, (x + self._grid_step, y + self._grid_step), cv2.FONT_HERSHEY_SIMPLEX, 1, self._cursor_color, 1)

        return img

    def _set_new_stack(self, x0, y0, x1, y1):
        temp_mask = np.array(self._mask.copy(), dtype=int)

        if x0 == x1 and y0 == y1:
            self._last_mouse_point = []
            return

        if x0 != x1 and y1 != y0:
            self._last_mouse_point = []
            return

        if x0 == x1:
            temp_mask[x0, y0:y1] += 1
        else:
            temp_mask[x0:x1, y0] += 1

        if np.max(temp_mask) > 1:
            self._last_mouse_point = []
            return

        if x0 == x1:
            self._mask[x0, y0:y1 + 1] = ~self._mask[x0, y0:y1 + 1]
        else:
            self._mask[x0:x1 + 1, y0] = ~self._mask[x0:x1 + 1, y0]

        self._last_mouse_point = []

    def _clear_stacks(self, x0, y0, x1, y1):
        roi = np.zeros((2, 2), dtype=int)
        self._mask[x0:x1 + 1, y0:y1 + 1] = False

    def _change_mask(self):
        if len(self._last_mouse_point) == 2:
            x0 = np.minimum(self._last_mouse_point[0][0], self._last_mouse_point[1][0])
            y0 = np.minimum(self._last_mouse_point[0][1], self._last_mouse_point[1][1])
            x1 = np.maximum(self._last_mouse_point[0][0], self._last_mouse_point[1][0])
            y1 = np.maximum(self._last_mouse_point[0][1], self._last_mouse_point[1][1])

            x0, y0 = x0 // self._grid_step, y0 // self._grid_step
            x1, y1 = x1 // self._grid_step, y1 // self._grid_step

            if self._current_mode is self.Mode.DRAW_STACK:
                self._set_new_stack(x0, y0, x1, y1)
            elif self._current_mode is self.Mode.CLEAR_STACK:
                self._clear_stacks(x0, y0, x1, y1)

    def _set_entry_or_exit(self, point):
        if len(point) == 0 or self._current_mode not in [self.Mode.SET_ENTRY, self.Mode.SET_EXIT]:
            raise RuntimeError('Impossible!')

        x, y = point[0] // self._grid_step, point[1] // self._grid_step
        if self._current_mode is self.Mode.SET_ENTRY:
            self._entry_point = (x, y)
        else:
            self._exit_point = (x, y)

    def _set_items(self, point):
        if len(point) == 0 or self._current_mode != self.Mode.SET_ITEMS:
            raise RuntimeError('Impossible!')

        x, y = point[0] // self._grid_step, point[1] // self._grid_step
        if self._mask[x, y] != 1:
            self._logger.error('The item can only be placed on the stack')
            return

        for point_idx, point in enumerate(self._items_points):
            if point[0] == x and point[1] == y:
                del self._items_points[point_idx]
                return

        self._items_points.append((x, y))

    def _draw_annotations(self, img):
        def draw_polygon(mask_cords: list or tuple, color: tuple):
            c, r = mask_cords
            x0, x1 = c * self._grid_step, (c + 1) * self._grid_step
            y0, y1 = r * self._grid_step, (r + 1) * self._grid_step
            pts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=int)
            cv2.fillPoly(img, pts=[pts], color=color)

        columns, rows = np.where(self._mask)
        for r, c in zip(rows, columns):
            draw_polygon((c, r), self._stack_color)

        for point in self._items_points:
            draw_polygon(point, self._items_color)

        if self._entry_point is not None:
            c, r = self._entry_point
            radius = self._grid_step // 2
            x, y = c * self._grid_step + radius, r * self._grid_step + radius
            cv2.circle(img, (x, y), radius, self._entry_color, 2)

        if self._exit_point is not None:
            c, r = self._exit_point
            x0, x1 = c * self._grid_step, (c + 1) * self._grid_step
            y0, y1 = r * self._grid_step, (r + 1) * self._grid_step
            cv2.line(img, (x0, y0), (x1, y1), self._exit_color, 2)
            cv2.line(img, (x1, y0), (x0, y1), self._exit_color, 2)

    def _save(self):
        if self._entry_point is None:
            self._logger.error('Please, set the entry to the warehouse')
            return False

        if self._exit_point is None:
            self._logger.error('Please, set the exit from the warehouse')
            return False

        if np.sum(self._mask) == 0:
            self._logger.error('Please, set the stacks in the warehouse')
            return False

        save_path = os.path.join(self._save_dir, f'{datetime.now().strftime("%d%m%Y_%H%M%S")}.json')
        data = {
            'grid_size': int(self._grid_size),
            'entry_point': list(self._entry_point),
            'exit_point': list(self._exit_point),
            'mask': [[int(v) for v in m] for m in self._mask],
            'item_points': list(self._items_points)
        }
        with open(save_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)

        return True

    def _worker(self):
        background_img = self._draw_background_image()

        cv2.namedWindow('annotation')
        cv2.setMouseCallback(('annotation'), self._mouse_events)

        while True:
            img = self._draw_mouse_position(background_img.copy())
            self._draw_annotations(img)
            self._draw_grid(img)
            self._draw_info(img)

            cv2.imshow('annotation', img)
            key = cv2.waitKey(1)

            if key == ord('q'):
                status = self._save()
                if not status:
                    continue
                cv2.destroyAllWindows()
                exit()
            if key == ord('s'):
                self._save()
            elif key == ord(self._select_mode_definitions[self.Mode.DRAW_STACK]):
                self._current_mode = self.Mode.DRAW_STACK
                self._last_mouse_point = []
            elif key == ord(self._select_mode_definitions[self.Mode.CLEAR_STACK]):
                self._current_mode = self.Mode.CLEAR_STACK
                self._last_mouse_point = []
            elif key == ord(self._select_mode_definitions[self.Mode.SET_ENTRY]):
                self._current_mode = self.Mode.SET_ENTRY
                self._last_mouse_point = []
            elif key == ord(self._select_mode_definitions[self.Mode.SET_EXIT]):
                self._current_mode = self.Mode.SET_EXIT
                self._last_mouse_point = []
            elif key == ord(self._select_mode_definitions[self.Mode.SET_ITEMS]):
                self._current_mode = self.Mode.SET_ITEMS
                self._last_mouse_point = []


if __name__ == '__main__':
    output_dir = args.output_dir if args.output_dir is not None else str(Path(__file__).parent)
    annotator = WarehouseAnnotator(grid_size=args.grid_size,
                                   img_size=args.img_size,
                                   save_dir=output_dir,
                                   anno_path=args.input_file)
    annotator.start()
