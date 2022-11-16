from glob import glob

import pyvista as pv
import numpy as np
import cv2

DATASET_DIR = "F:/nuScenes/sweeps/"

def open_lidar(file_name: str, min_dist: float = 2):

    assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

    scan = np.fromfile(file_name, dtype=np.float32)
    points = scan.reshape((-1, 5)).T

    x_filt = np.abs(points[0, :]) < min_dist
    y_filt = np.abs(points[1, :]) < min_dist
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]

    return points.T

if __name__ == "__main__":

    files = glob(DATASET_DIR + "LIDAR_TOP/*.bin")

    for file_path in files:
        points = open_lidar(file_path)

        BLOC_SIZE = 0.5 # 25 centimetre

        DISTANCE_X = 40
        DISTANCE_Y = 40

        print("Taille de la map :", DISTANCE_X*BLOC_SIZE, "x", DISTANCE_Y*BLOC_SIZE, "m")

        grid = np.zeros((DISTANCE_Y * 2, DISTANCE_X * 2))

        x_axis = (points[:, 0] // BLOC_SIZE).astype(int) + DISTANCE_X
        x_selection = (x_axis >= 0) & (x_axis < DISTANCE_X * 2)
        points = points[x_selection]
        x_axis = x_axis[x_selection]
        
        y_axis = (points[:, 1] // BLOC_SIZE).astype(int) + DISTANCE_Y
        y_selection = (y_axis >= 0) & (y_axis < DISTANCE_Y * 2)
        points = points[y_selection]
        x_axis = x_axis[y_selection]
        y_axis = y_axis[y_selection]

        axis = np.array((y_axis, x_axis)).T
        index, counts = np.unique(axis, axis=0, return_counts=True)

        y_axis, x_axis = index.T

        grid[y_axis, x_axis] = counts

        grid = grid.astype(float) / np.max(grid) * 255.0

        cv2.imshow("Test", np.kron(grid, np.ones((300//DISTANCE_X, 300//DISTANCE_Y)))[::-1])
        cv2.waitKey(1)