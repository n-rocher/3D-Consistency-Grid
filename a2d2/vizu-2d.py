import os
import utm
import cv2
import json
import matplotlib.pyplot as plt

import numpy as np
import pyvista as pv
from glob import glob
import numpy.linalg as la

DATASET_DIR = "E:/A2D2/"
LIDAR_SENSORS = ["cam_front_center", "cam_front_left", "cam_front_right", "cam_rear_center", "cam_side_left", "cam_side_right"]

EPSILON = 1.0e-10  # norm should not be small


def Rx(theta):
    return np.matrix([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])

def Ry(theta):
    return np.matrix([[np.cos(theta), 0, np.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]])

def Rz(theta):
    return np.matrix([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


if __name__ == "__main__":

    f = open(DATASET_DIR + "/cams_lidars.json")
    CONFIG = json.load(f)
    f.close()

    sequence_paths = [os.path.dirname(path) for path in glob(DATASET_DIR + "*/")]

    for sequence_path in sequence_paths:

        sequence_path = os.path.basename(sequence_path)
        sequence_name = sequence_path.replace("_", "")

        f = open(DATASET_DIR + "/" + sequence_name + "_bus_signals.json")
        BUS_SIGNAL = json.load(f)
        f.close()

        lidar_paths = glob(DATASET_DIR + "/" + sequence_path + "/lidar_360/*.npz")
        lidar_index = list(map(lambda x: int(os.path.basename(x)[:-4]), lidar_paths))

        roll_angle = list(map(lambda x: x[1], BUS_SIGNAL["roll_angle"]["values"]))
        pitch_angle = list(map(lambda x: x[1], BUS_SIGNAL["pitch_angle"]["values"]))

        default = 0
        lidar = 5000
        space = 1
        for index in range(default, default + lidar * space, space):

            point2d = None
            points = None
            grid = None

            index = lidar_index[index]

            roll_angle = BUS_SIGNAL['roll_angle']["values"][index][1]
            pitch_angle = BUS_SIGNAL['pitch_angle']["values"][index][1]

            grid = np.zeros((1000, 1000), dtype=np.uint8)

            points = np.load(lidar_paths[index])["points"]
            pointstr = np.ones((len(points), 4))
            pointstr[:, :3] = points
            points = pointstr
            #points = points @ Rz(pitch_angle)
            # points = points.T

            x = points[:, 0]
            y = points[:, 1]

            point2d = np.copy(points[:, :2])
            point2d = point2d * 10
            point2d = point2d.astype(int)
            point2d = point2d + 500

            point2d = point2d[point2d[:, 0] < 1000]
            point2d = point2d[point2d[:, 1] < 1000]

            grid[point2d[:, 0], point2d[:, 1]] = 1

            cv2.imshow("points", grid * 255)
            cv2.waitKey(5)
