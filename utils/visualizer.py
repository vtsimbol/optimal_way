import cv2
import numpy as np


class Visualizer:
    def __init__(self, grid_step=10):
        self._grid_step = grid_step

        self._grid_color = (160, 160, 160)
        self._stack_color = (235, 125, 52)
        self._items_color = (0, 0, 0)
        self._entry_color = (51, 255, 0)
        self._exit_color = (0, 64, 255)
        self._walls_mask = (64, 64, 64)

    def show(self, data, way=None):
        def draw_polygon(stacks_mask_cords: list or tuple, polygon_color: tuple):
            c, r = stacks_mask_cords
            x0, x1 = c * self._grid_step, (c + 1) * self._grid_step
            y0, y1 = r * self._grid_step, (r + 1) * self._grid_step
            pts = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=int)
            cv2.fillPoly(img, pts=[pts], color=polygon_color)

        grid_size = data['grid_size']
        stacks_mask = data['stacks_mask']
        walls_mask = data['walls_mask']
        entry_point = data['entry_point']
        exit_point = data['exit_point']
        items_points = data['item_points']

        w, h = grid_size[0] * self._grid_step, grid_size[1] * self._grid_step
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # draw stacks and walls
        columns, rows = np.where(stacks_mask)
        for r, c in zip(rows, columns):
            draw_polygon((c, r), self._stack_color)
        columns, rows = np.where(walls_mask)
        for r, c in zip(rows, columns):
            draw_polygon((c, r), self._walls_mask)

        # draw items
        for point in items_points:
            draw_polygon(point, self._items_color)

        # draw entry point
        c, r = entry_point
        radius = self._grid_step // 2
        x, y = c * self._grid_step + radius, r * self._grid_step + radius
        cv2.circle(img, (x, y), radius, self._entry_color, 2)

        # draw exit point
        c, r = exit_point
        x0, x1 = c * self._grid_step, (c + 1) * self._grid_step
        y0, y1 = r * self._grid_step, (r + 1) * self._grid_step
        cv2.line(img, (x0, y0), (x1, y1), self._exit_color, 2)
        cv2.line(img, (x1, y0), (x0, y1), self._exit_color, 2)

        # draw grid
        for i in range(1, grid_size[1]):
            cv2.line(img, (0, self._grid_step * i), (w, self._grid_step * i), self._grid_color, 1)
        for i in range(1, grid_size[0]):
            cv2.line(img, (self._grid_step * i, 0), (self._grid_step * i, h), self._grid_color, 1)

        if way is not None:
            color = (0, 0, 255)
            total_points = len(way)
            color_step = 510 // total_points

            for point in way:
                draw_polygon(point, color)

                b, g, r = color
                if g != 255:
                    g = int(np.minimum(255, g + color_step))

                else:
                    r = int(np.maximum(0, r - color_step))
                color = (b, g, r)

                cv2.imshow('test', img)
                cv2.waitKey(0)

        return img
