import os
import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# inner size of chessboard
width = 9
height = 6
square_size = 0.025  # 0.025 meters

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
objp = np.zeros((height * width, 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

objp = objp * square_size  # Create real world coords. Use your metric.

# Arrays to store object points and image points from all the images.


img_width = 640
img_height = 480
image_size = (img_width, img_height)

image_dir = "dataset/pairs/"

number_of_images = 50

def get_image_points(image_dir, number_of_images):    # string left or right side
    objpoints = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []  # 2d points in image plane.
    
    for i in range(1, number_of_images):
        # read image
        right_img = cv2.imread(image_dir + "right_%02d.png" % i)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        left_img = cv2.imread(image_dir + "left_%02d.png" % i)
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray,
            (width, height),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray,
            (width, height),
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        Y, X, channels = right_img.shape

        # skip images where the corners of the chessboard are too close to the edges of the image
        if ret_right  and ret_left:
            minRx_left = corners_left[:, :, 0].min()
            maxRx_left = corners_left[:, :, 0].max()
            minRy_left = corners_left[:, :, 1].min()
            maxRy_left = corners_left[:, :, 1].max()

            border_threshold_x = X / 12
            border_threshold_y = Y / 12

            x_thresh_bad = False
            if minRx_left < border_threshold_x:
                x_thresh_bad = True

            y_thresh_bad = False
            if minRy_left < border_threshold_y:
                y_thresh_bad = True

            if (y_thresh_bad == True) or (x_thresh_bad == True):
                continue

            minRx_right = corners_right[:, :, 0].min()
            maxRx_right = corners_right[:, :, 0].max()
            minRy_right = corners_right[:, :, 1].min()
            maxRy_right = corners_right[:, :, 1].max()

            x_thresh_bad = False
            if minRx_right < border_threshold_x:
                x_thresh_bad = True

            y_thresh_bad = False
            if minRy_right < border_threshold_y:
                y_thresh_bad = True

            if (y_thresh_bad == True) or (x_thresh_bad == True):
                continue


        if ret_right and ret_left:
            
            objpoints.append(objp)

            # improving the location of points (sub-pixel)
            corners2 = cv2.cornerSubPix(left_gray, corners_left, (3, 3), (-1, -1), criteria)
            imgpoints_left.append(corners2)


            # improving the location of points (sub-pixel)
            corners3 = cv2.cornerSubPix(right_gray, corners_right, (3, 3), (-1, -1), criteria)
            imgpoints_right.append(corners3)   


            # Draw and display the corners
            # Show the image to see if pattern is found ! imshow function.
            cv2.drawChessboardCorners(left_img, (width, height), corners2, ret_left)
            cv2.namedWindow("Chessboard corners", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Chessboard corners", 750, 750)
            cv2.imshow("Chessboard corners", left_img)
            cv2.waitKey(10)
        else:
            print("Chessboard couldn't detected. Image pair: ", i)
            continue

    cv2.destroyAllWindows()

    return objpoints, imgpoints_left, imgpoints_right


def get_K_D(objpoints, imgpoints, image_size):
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    ret, K, D, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    )

    return K, D


objpoints, left_imgpoints, right_imgpoints = get_image_points(image_dir, number_of_images)

K_left, D_left = get_K_D(objpoints, left_imgpoints, image_size)
K_right, D_right = get_K_D(objpoints, right_imgpoints, image_size)

imgpointsLeft = np.asarray(left_imgpoints, dtype=np.float64)
imgpointsRight = np.asarray(right_imgpoints, dtype=np.float64)
objpoints = np.asarray(objpoints, dtype=np.float64)



(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
        objpoints, imgpointsLeft, imgpointsRight,
        K_left, D_left,
        K_right, D_right,
        image_size, None, None,
        cv2.CALIB_FIX_INTRINSIC, 
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))

R2 = np.zeros([3,3])
P1 = np.zeros([3,4])
P2 = np.zeros([3,4])
Q = np.zeros([4,4])

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection, dispartityToDepthMap) = cv2.fisheye.stereoRectify(
        K_left, D_left,
        K_right, D_right,
        image_size, 
        rotationMatrix, translationVector,
        0, R2, P1, P2, Q,
        cv2.CALIB_ZERO_DISPARITY, (0,0) , 0, 0)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
        K_left, D_left, leftRectification,
        leftProjection, image_size, cv2.CV_16SC2)
        
map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
        K_right, D_right, rightRectification,
        rightProjection, image_size, cv2.CV_16SC2)

img_l = cv2.imread(image_dir + "left_01.png")
img_r = cv2.imread(image_dir + "right_01.png")

dst_L = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)


if __name__ == "__main__":

    N, XX, YY = dst_L.shape[::-1] # RGB image size

    visRectify = np.zeros((YY, XX*2, N), np.uint8) # create a new image with a new size (height, 2*width)
    visRectify[:,0:XX:,:] = dst_L      # left image assignment
    visRectify[:,XX:XX*2:,:] = dst_R   # right image assignment

    # draw horizontal lines
    for y in range(0,YY,10):
        cv2.line(visRectify, (0,y), (XX*2,y), (255,0,0))

    N, XX, YY = dst_L.shape[::-1] # RGB image size

    normal = np.zeros((YY, XX*2, N), np.uint8) # create a new image with a new size (height, 2*width)
    normal[:,0:XX:,:] = img_l      # left image assignment
    normal[:,XX:XX*2:,:] = img_r   # right image assignment


    cv2.imshow('visRectify',visRectify)  # display image with lines
    cv2.imshow('normal',normal)  # display image with lines
    cv2.waitKey(0)
