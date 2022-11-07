from glob import glob

import cv2
import json
import math
import pandas
import numpy as np
import pyvista as pv

from rich import print
from pyquaternion import Quaternion

DATASET_DIR = "F:/nuScenes/"
SAMPLE_DATA = DATASET_DIR + "v1.0-trainval/sample_data.json"
EGO_POSE = DATASET_DIR + "v1.0-trainval/ego_pose.json"


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def open_lidar(file_name: str, min_dist: float = 2):

    assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

    scan = np.fromfile(file_name, dtype=np.float32)
    points = scan.reshape((-1, 5)).T

    x_filt = np.abs(points[0, :]) < min_dist
    y_filt = np.abs(points[1, :]) < min_dist
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]

    return points.T

def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


if __name__ == "__main__":

    # Available files
    lidar_files = glob(DATASET_DIR + "sweeps/LIDAR_TOP/*.bin")
    file_name_list = list(map(lambda x: x[len(DATASET_DIR):].replace("/", "").replace("\\", ""), lidar_files))

    # Database
    database_file = open(SAMPLE_DATA, 'r')
    database_file = json.load(database_file)
    DATABASE = {data["filename"].replace("/", "").replace("\\", ""): data for data in database_file if "filename" in data and "LIDAR" in data["filename"]}

    # Egoposition
    ego_pose_file = open(EGO_POSE, 'r')
    ego_pose_file = json.load(ego_pose_file)
    EGOPOSE_DATABASE = {data["token"]: data for data in ego_pose_file}

    print(len(ego_pose_file))

    print("Il y a ", len(lidar_files), "fichiers")

    default = 0
    lidar = 5000
    space = 1
    for index in range(default, default + lidar * space, space):

        point2d = None
        points = None
        grid = None

        pc = open_lidar(lidar_files[index])
        points = pc[:, :4]
        points[:, 3] = 1

        egopose = EGOPOSE_DATABASE[DATABASE[file_name_list[index]]["ego_pose_token"]]

        aa = quaternion_yaw(Quaternion(egopose["rotation"]))
        R_z = np.array([[math.cos(0 + aa), -math.sin(0 + aa), 0, 0],
                        [math.sin(0 + aa), math.cos(0 + aa), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ])


        points = R_z @ points.T
        points = points.T



        grid = np.zeros((1000, 1000), dtype=np.uint8)

        x = points[:, 0]
        y = points[:, 1]

        point2d = np.copy(points[:, :2])
        point2d = point2d * 10
        point2d = point2d.astype(int)
        point2d = point2d + 500

        point2d[point2d[:, 0] >= 1000] = 999
        point2d[point2d[:, 1] >= 1000] = 999

        grid[point2d[:, 0], point2d[:, 1]] = 1

        cv2.imshow("points", grid * 255)
        cv2.waitKey(5)
