import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    img_left = cv2.imread("left_panorama.jpg")
    img_right = cv2.imread("right_panorama.jpg")
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    siftobject = cv2.SIFT_create()

    kp_left, desc_left = siftobject.detectAndCompute(img_left_gray, None)
    kp_img_left = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 0, 255))

    kp_right, desc_right = siftobject.detectAndCompute(img_right_gray, None)
    kp_img_right = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 0, 255))
    
    plt.subplot(1, 2, 1)
    plt.imshow(kp_img_left, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(kp_img_right, cmap='gray')
    plt.axis('off')
    plt.show()


    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc_left, desc_right)

    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img_left,kp_left,img_right,kp_right,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.show()

    ptsA = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
    width = img_left.shape[1] + img_right.shape[1]
    height = img_left.shape[0] + img_right.shape[0]
    result = cv2.warpPerspective(img_left, H, (width, height))
    result[0:img_right.shape[0], 0:img_right.shape[1]] = img_right
    
    res = np.where(result != 0)
    x, y = res[1], res[0]
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    result_cropped = result[ymin:ymax, xmin:xmax]
    plt.imshow(result_cropped, cmap='gray')
    plt.axis('off')
    plt.show()
