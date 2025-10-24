#!/usr/bin/env python3

'''
Read img from ~./data_source/Basler_calib_data
Export 
'''

import cv2 as cv
import numpy as np
import glob

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from example_interfaces.srv import Trigger


class CalibCamera(Node):
    def __init__(self):
        super().__init__('calib_camera')

        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object point for inner corner, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = glob.glob('/home/thinh/ros2_ws/src/camera_worker_pkg/data_source/Basler_calib_data/*.png')
        for fname in images:
            img = cv.imread(fname, cv.IMREAD_GRAYSCALE)

            # Find the checkerboard corners
            ret, corners = cv.findChessboardCorners(img, (8, 6), None)


            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(img, corners, (5, 5), (-1, -1), criteria=criteria)
                imgpoints.append(corners2)

        print(len(imgpoints))
        print(len(objpoints))
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

        np.savez("/home/thinh/ros2_ws/src/camera_worker_pkg/data_source/Basler_calib_data/basler_camera_params.npz", 
                 camera_matrix=mtx, 
                 dist_coeffs=dist,
                 rvecs=rvecs,
                 tvecs=tvecs)

def main(args=None):
    rclpy.init(args=args)
    my_node = CalibCamera()
    rclpy.shutdown()

if __name__ == '__main__':
    main()