import os
import utm
import json

import numpy as np
import pyvista as pv
from glob import glob
import numpy.linalg as la

DATASET_DIR = "E:/A2D2/"
LIDAR_SENSORS = ["cam_front_center", "cam_front_left", "cam_front_right", "cam_rear_center", "cam_side_left", "cam_side_right"]

EPSILON = 1.0e-10 # norm should not be small



def Rx(theta):
  return np.matrix([[ 1, 0           , 0        ,0   ],
                   [ 0, np.cos(theta),-np.sin(theta), 0],
                   [ 0, np.sin(theta), np.cos(theta), 0],
                   [ 0, 0, 0, 1]])
  
def Ry(theta):
  return np.matrix([[ np.cos(theta), 0, np.sin(theta), 0 ],
                   [ 0           , 1, 0          , 0 ],
                   [-np.sin(theta), 0, np.cos(theta), 0],
                   [ 0, 0, 0, 1]])
  
def Rz(theta):
  return np.matrix([[ np.cos(theta), -np.sin(theta), 0, 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 , 0 ],
                   [ 0           , 0            , 1 , 0],
                   [ 0, 0, 0, 1]])

if __name__ == "__main__":

    f = open(DATASET_DIR + "/cams_lidars.json")
    CONFIG = json.load(f)
    f.close()

    sequence_paths = [ os.path.dirname(path) for path in glob(DATASET_DIR + "*/")]

    for sequence_path in sequence_paths:

        sequence_path = os.path.basename(sequence_path)
        sequence_name = sequence_path.replace("_", "")
    
        f = open(DATASET_DIR + "/" + sequence_name + "_bus_signals.json")
        BUS_SIGNAL = json.load(f)
        f.close()
      
        print(BUS_SIGNAL.keys())

        lidar_paths = sorted(glob(DATASET_DIR + "/" + sequence_path + "/lidar_360/*.npz"))
        lidar_index = list(map(lambda x : int(os.path.basename(x)[:-4]), lidar_paths))

        latitude_degree = list(map(lambda x : x[1], BUS_SIGNAL["latitude_degree"]["values"]))
        longitude_degree = list(map(lambda x : x[1], BUS_SIGNAL["longitude_degree"]["values"]))
        GPS_DATA = np.asarray(list(map(lambda x : utm.from_latlon(x[0], x[1])[:2], zip(latitude_degree, longitude_degree))))

    

        print(BUS_SIGNAL["roll_angle"]["unit"])

        roll_angle = list(map(lambda x : x[1], BUS_SIGNAL["roll_angle"]["values"]))
        pitch_angle = list(map(lambda x : x[1], BUS_SIGNAL["pitch_angle"]["values"]))

        print(np.max(roll_angle), np.min(roll_angle), np.radians(np.max(roll_angle)), np.radians(np.min(roll_angle)))
        print(np.max(pitch_angle), np.min(pitch_angle))

        origin = None

        plotter = pv.Plotter()
        for index in lidar_index[0:150:10]:
            output_points = None

            roll_angle = BUS_SIGNAL['roll_angle']["values"][index][1]
            pitch_angle = BUS_SIGNAL['pitch_angle']["values"][index][1]

            # print(roll_angle, pitch_angle)

            gps_info = GPS_DATA[index]

            if origin is None:
                origin = gps_info


            translation = np.eye(4)
            translation[:2, 3] = gps_info - origin

            points = np.load(lidar_paths[index])["points"]

            point_tr = np.ones((len(points), 4))
            point_tr[:, :3] = points

            # translation
            point_tr =   Rz(np.radians(90 + roll_angle)) @ translation @ point_tr.T
            point_tr = point_tr.T

            # point_tr = point_tr.T
            # point_tr = point_tr.T

            # plotter.add_points(points[:, :3].astype(np.float32), point_size=2, render_points_as_spheres=True, scalars=[[255, 0, 0] ]* len(points), rgb=True)
            plotter.add_points(point_tr[:, :3].astype(np.float32), point_size=1, render_points_as_spheres=True, scalars=[[0, 255, 0] ]* len(points), rgb=True)
                    
        plotter.show()
        exit()
