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

        BLOC_SIZE = 0.5 # 50 centimetres
        BLOC_SIZE_Z = 1 # 1 mètre

        DISTANCE_X = 25
        DISTANCE_Y = 25
        DISTANCE_Z = 5

        print("Taille de la map :", DISTANCE_X*BLOC_SIZE, "x", DISTANCE_Y*BLOC_SIZE, "x", BLOC_SIZE_Z*DISTANCE_Z, "m")

        grid = np.zeros((DISTANCE_Y * 2, DISTANCE_X * 2, DISTANCE_Z))

        x_axis = (points[:, 0] / BLOC_SIZE + DISTANCE_X).astype(int)
        x_selection = (x_axis >= 0) & (x_axis < DISTANCE_X * 2)
        points = points[x_selection]
        x_axis = x_axis[x_selection]
        
        y_axis = (points[:, 1] / BLOC_SIZE + DISTANCE_Y).astype(int)
        y_selection = (y_axis >= 0) & (y_axis < DISTANCE_Y * 2)
        points = points[y_selection]
        x_axis = x_axis[y_selection]
        y_axis = y_axis[y_selection]

        z_axis = (points[:, 2] / BLOC_SIZE_Z + 1).astype(int)
        z_selection = (z_axis >= 0) & (z_axis < DISTANCE_Z - 1)
        points = points[z_selection]
        x_axis = x_axis[z_selection]
        y_axis = y_axis[z_selection]
        z_axis = z_axis[z_selection]

        grid[y_axis, x_axis, z_axis] = 1

        x, y, z = np.where(grid > 0)

        points = np.array(list(zip(x, y, z)))

        pl = pv.Plotter()

        for p in list(zip(x, y, z)):
            bounds = (p[0] - BLOC_SIZE, p[0] + BLOC_SIZE,
                    p[1] - BLOC_SIZE, p[1] + BLOC_SIZE,
                    p[2], p[2] + BLOC_SIZE_Z)

            pl.add_mesh(pv.Box(bounds=bounds))
        #pl.add_points(points, render_points_as_spheres=True, point_size=10)
        pl.show()