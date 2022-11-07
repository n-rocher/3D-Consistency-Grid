from glob import glob

import pyvista as pv
import numpy as np

DATASET_DIR = "U:/nuScenes/sweeps/"

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

    print("Il y a ", len(files), "fichiers")

    pc = open_lidar(files[0])

    pdata = pv.PolyData(pc[:, :3])

    # create many spheres from the point cloud
    sphere = pv.Sphere(radius=0.02, phi_resolution=10, theta_resolution=10)
    pc = pdata.glyph(scale=False, geom=sphere, orient=False)
    pc.plot(cmap='jet')
