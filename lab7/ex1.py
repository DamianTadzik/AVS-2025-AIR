# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import maximum_filter


# def Harris(image, sobel_size, gauss_size):
#     x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=sobel_size)
#     y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=sobel_size)

#     xx = cv2.GaussianBlur(x * x, (gauss_size, gauss_size), 0)
#     yy = cv2.GaussianBlur(y * y, (gauss_size, gauss_size), 0)
#     xy = cv2.GaussianBlur(x * y, (gauss_size, gauss_size), 0)

#     K = 0.05
#     det = xx * yy - xy ** 2
#     trace = xx + yy
#     H = det - K * trace ** 2

#     H = cv2.normalize(
#         H, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
#     )
#     return H


# def find_max(image, size, threshold):
#     data_max = maximum_filter(image, size)
#     maxima = image == data_max
#     diff = image > threshold
#     maxima[diff == 0] = 0
#     return np.nonzero(maxima)


# def draw_marks(image, coordinates):
#     plt.figure()
#     plt.imshow(image)
#     for coord in zip(*coordinates):
#         plt.plot(coord[1], coord[0], "*", color="r")  # Swap x and y coordinates for plotting
#     plt.show()


# if __name__ == "__main__":
#     sobel_filter = 5
#     gauss_filter = 7
#     threshold = 0.44

#     img_1 = cv2.imread("fontanna1.jpg")
#     img_2 = cv2.imread("fontanna2.jpg")
#     img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY).astype(np.float32)
#     img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY).astype(np.float32)

#     H1 = Harris(img_1_gray, sobel_filter, gauss_filter)
#     H2 = Harris(img_2_gray, sobel_filter, gauss_filter)

#     corners1 = find_max(H1, sobel_filter, threshold)
#     corners2 = find_max(H2, sobel_filter, threshold)
#     draw_marks(img_1, corners1)
#     draw_marks(img_2, corners2)


#     img_3 = cv2.imread("budynek1.jpg")
#     img_4 = cv2.imread("budynek2.jpg")
#     img_3_gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY).astype(np.float32)
#     img_4_gray = cv2.cvtColor(img_4, cv2.COLOR_BGR2GRAY).astype(np.float32)

#     H3 = Harris(img_3_gray, sobel_filter, gauss_filter)
#     H4 = Harris(img_4_gray, sobel_filter, gauss_filter)

#     corners3 = find_max(H3, sobel_filter, threshold)
#     corners4 = find_max(H4, sobel_filter, threshold)
#     draw_marks(img_3, corners3)
#     draw_marks(img_4, corners4)
   

import cv2
import scipy.ndimage as filters
import numpy as np
from matplotlib import pyplot as plt

def read_gray(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    return img

def H(img, mask_size):
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=mask_size)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=mask_size)
    Ix = cv2.GaussianBlur(sobelx * sobelx, (mask_size, mask_size), 0)
    Iy = cv2.GaussianBlur(sobely * sobely, (mask_size, mask_size), 0)
    Ixy = cv2.GaussianBlur(sobelx * sobely, (mask_size, mask_size), 0)
    det = Ix * Iy - Ixy * Ixy
    trace = Ix + Iy
    H = det - 0.05 * trace * trace
    return normalize(H)

def find_max(image, size, threshold):  # size - maximum filter mask size
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def plot_points(img, points):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.plot(points[1], points[0], '*')
    plt.show()

if __name__ == "__main__":
    fontanna1 = read_gray('fontanna1.jpg')
    fontanna2 = read_gray('fontanna2.jpg')
    fontanna1_max = find_max(H(fontanna1, 7), 7, 0.4)
    fontanna2_max = find_max(H(fontanna2, 7), 7, 0.4)

    plot_points(fontanna1, fontanna1_max)
    plot_points(fontanna2, fontanna2_max)

    # Budynek
    budynek1 = read_gray('budynek1.jpg')
    budynek2 = read_gray('budynek2.jpg')
    budynek1_max = find_max(H(budynek1, 7), 7, 0.5)
    budynek2_max = find_max(H(budynek2, 7), 7, 0.5)
    
    plot_points(budynek1, budynek1_max)
    plot_points(budynek2, budynek2_max)
