import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class CameraCalibration:
    """Class for camera calibration
    """
    def __init__(self):
        self.images = [] # array of all calibration images' names
        self.objpoints = []  # array to store all images' object points
        self.imgpoints = []  # array to store all images' found corners

    def camera_calibration_main(self):
        self.open_images()
        self.find_chessboard_corners()
        self.get_undistort_images()

    def open_images(self):
        """Read in all calibration images
        """
        self.images = glob.glob('camera_cal/calibration*.jpg')

    def find_chessboard_corners(self):
        """Find chessboard corners from each image
        """
        # Prepare object points
        objpoints = np.zeros((6 * 9, 3), np.float32)
        objpoints[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        for filename in self.images:
            # Read image and convert to gray scale
            img = mpimg.imread(filename)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Use openCV to find chess board corners
            ret, corners = cv2.findChessboardCorners(gray_img, (9, 6), None)

            # If corners found, add object points, image points to lists
            if ret:
                self.imgpoints.append(corners)
                self.objpoints.append(objpoints)

    def get_undistort_images(self):
        """Undistort images if corners found
        """
        # Create a folder to save images
        undistort_directory = 'undistorted_images\\'
        if not os.path.exists(undistort_directory):
            os.makedirs(undistort_directory)

        # Undistort each calibration image
        for filename in self.images:
            img = mpimg.imread(filename)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                                               img.shape[1::-1], None, None)
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
            # save undistored_img to undistorted_images folder
            pure_filename = filename[filename.rfind('\\')+1:]
            undistort_filename = ''.join([undistort_directory, pure_filename])
            plt.imsave(undistort_filename, undistorted_img)


def main():
    camera_cali = CameraCalibration()
    camera_cali.camera_calibration_main()


if __name__ == '__main__':
    main()