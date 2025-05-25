import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ex6_2 import map1_left, map2_left, map1_right, map2_right

# Arrays to store object points and image points from all the images.
I_width = 640
I_height = 480

I = cv2.imread("dataset/example/example1.png")

N, XX, YY = I.shape[::-1]
left_I = np.zeros((YY, int(XX / 2), N), np.uint8)
right_I = np.copy(left_I)
left_I = I[:, 0 : int(XX / 2) :, :]
right_I = I[:, int(XX / 2) : XX :, :]

left_I = cv2.resize(left_I, (I_width, I_height))
right_I = cv2.resize(right_I, (I_width, I_height))

left_rectified = cv2.remap(left_I, map1_left, map2_left, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_I, map1_right, map2_right, cv2.INTER_LINEAR)

left_rect_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
right_rect_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
left_gray = cv2.cvtColor(left_I, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_I, cv2.COLOR_BGR2GRAY)

block_matcher = cv2.StereoBM_create(numDisparities=96, blockSize=15)
sgm_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=96, blockSize=15)

disparity_bm = block_matcher.compute(left_rect_gray, right_rect_gray)
disparity_sgm = sgm_matcher.compute(left_rect_gray, right_rect_gray)
disparity_bm_unrec = block_matcher.compute(left_gray, right_gray)
disparity_sgm_unrec = sgm_matcher.compute(left_gray, right_gray)


disparity_bm_norm = cv2.normalize(
    disparity_bm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
)
disparity_sgm_norm = cv2.normalize(
    disparity_sgm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
)

disparity_bm_unrec_norm = cv2.normalize(
    disparity_bm_unrec,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U,
)
disparity_sgm_unrec_norm = cv2.normalize(
    disparity_sgm_unrec,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX,
    dtype=cv2.CV_8U,
)

heatmap_bm = cv2.applyColorMap(disparity_bm_norm, cv2.COLORMAP_HOT)
heatmap_sgm = cv2.applyColorMap(disparity_sgm_norm, cv2.COLORMAP_HOT)

heatmap_bm_unrec = cv2.applyColorMap(disparity_bm_unrec_norm, cv2.COLORMAP_HOT)
heatmap_sgm_unrec = cv2.applyColorMap(disparity_sgm_unrec_norm, cv2.COLORMAP_HOT)


plt.figure(figsize=(18, 10))

plt.subplot(4, 2, 1)
plt.title("left_I")
plt.imshow(left_I)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 2)
plt.title("right_I")
plt.imshow(right_I)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 3)
plt.title("left_rectified")
plt.imshow(left_rectified)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 4)
plt.title("right_rectified")
plt.imshow(right_rectified)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 5)
plt.title("heatmap_bm")
plt.imshow(heatmap_bm)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 6)
plt.title("heatmap_sgm")
plt.imshow(heatmap_sgm)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 7)
plt.title("heatmap_bm_unrec")
plt.imshow(heatmap_bm_unrec)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.subplot(4, 2, 8)
plt.title("heatmap_sgm_unrec")
plt.imshow(heatmap_sgm_unrec)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()
