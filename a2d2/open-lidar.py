import os
from glob import glob
import json

import pyvista as pv
import numpy as np
import numpy.linalg as la

DATASET_DIR = "E:/A2D2/"
LIDAR_SENSORS = ["cam_front_center", "cam_front_left", "cam_front_right", "cam_rear_center", "cam_side_left", "cam_side_right"]

EPSILON = 1.0e-10 # norm should not be small

def get_origin_of_a_view(view):
    return view['origin']

def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']
     
    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)
    
    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")
        
    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm
    
    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
 
    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)
    
    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)
    
    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")
        
    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm
    
    return x_axis, y_axis, z_axis

def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)
    
    # get origin 
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)
    
    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis
    
    # origin
    transform_to_global[0:3, 3] = origin
    
    return transform_to_global

def get_transform_from_global(view):
   # get transform to global
   transform_to_global = get_transform_to_global(view)
   trans = np.eye(4)
   rot = np.transpose(transform_to_global[0:3, 0:3])
   trans[0:3, 0:3] = rot
   trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])
    
   return trans

def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))
    
    return transform


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

        lidar_paths = dict()
        lidar_names = dict()
        LIDAR = dict()

        for lidar_sensor in LIDAR_SENSORS:
            lidar_paths[lidar_sensor] = glob(DATASET_DIR + "/" + sequence_path + "/lidar/" + lidar_sensor + "/*.npz")
            lidar_names[lidar_sensor] = [os.path.basename(path)[:-4][-9:] for path in lidar_paths[lidar_sensor]]
            for name, path in zip(lidar_names[lidar_sensor], lidar_paths[lidar_sensor]):
                if name in LIDAR:
                    LIDAR[name][lidar_sensor] = path
                else: 
                    LIDAR[name] = { lidar_sensor: path }

        SYNC_NAMES = sorted(list(set.intersection(*map(set, lidar_names.values()))))

        for sync_name in SYNC_NAMES:

            lidar_data = LIDAR[sync_name]
            output_points = None

            for lidar_sensor in LIDAR_SENSORS: 
                cam_front_center = np.load(lidar_data[lidar_sensor])

                points = cam_front_center["pcloud_points"]

                src_view = CONFIG['cameras'][lidar_sensor[4:]]['view']

                vehicle_view = CONFIG['vehicle']['view']

                trans = transform_from_to(src_view, vehicle_view)
                points_hom = np.ones((points.shape[0], 4))
                points_hom[:, 0:3] = points
                points_trans = (np.dot(trans, points_hom.T)).T 
                
                points = points_trans[:,0:3]

                if output_points is None:
                    output_points = points
                else:
                    output_points = np.concatenate((output_points, points))

            output_points = output_points.astype(np.float16)

            print(sync_name, output_points.shape, output_points.dtype)

            np.savez_compressed(DATASET_DIR + sequence_path + "/lidar_360/" + sync_name + ".npz", points=output_points)

            # pdata = pv.PolyData(output_points.astype(np.float32))
            # pdata.plot(cmap='jet')
