import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

img = cv2.imread("trybik.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

_, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

img_bin = cv2.bitwise_not(img_bin)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

# cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Resized_Window1", 1000, 1000)
# cv2.imshow("Resized_Window1", img_bin)
# cv2.waitKey(0)

# _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Resized_Window1", 1000, 1000)
# cv2.imshow("Resized_Window1", img_bin)
# cv2.waitKey(0)

contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
# cv2.namedWindow("Resized_Window1", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Resized_Window1", 1000, 1000)
# cv2.imshow("Resized_Window1", img)
# cv2.waitKey(0)


sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient_amplitude = np.sqrt(sobelx ** 2 + sobely ** 2)
gradient_amplitude = gradient_amplitude / np.amax(gradient_amplitude)
gradient_orientation = np.rad2deg(np.arctan2(sobely, sobelx))


moments = cv2.moments(img_bin)
center_x = int(moments["m10"] / moments["m00"])
center_y = int(moments["m01"] / moments["m00"])
reference_point = (center_x, center_y)

Rtable = [[] for _ in range(360)]

for point in contours[0]:
    x, y = point[0]
    vector = (x - reference_point[0], y - reference_point[1])
    distance = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    angle = np.arctan2(vector[1], vector[0])
    grad_orientation = (int(gradient_orientation[y, x]) + 360) % 360
    Rtable[grad_orientation].append((distance, angle))


target_img = cv2.imread("trybiki2.jpg")
target_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)

sobelx_target = cv2.Sobel(target_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely_target = cv2.Sobel(target_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient_magnitude_target = np.sqrt(sobelx_target ** 2 + sobely_target ** 2)
gradient_magnitude_target = gradient_magnitude_target / np.amax(
    gradient_magnitude_target
)
gradient_orientation_target = (np.rad2deg(np.arctan2(sobely_target, sobelx_target)) + 360) % 360


hough_space = np.zeros(target_gray.shape)
# threshold = 0.5
for y in range(target_gray.shape[0]):
    for x in range(target_gray.shape[1]):
        if gradient_magnitude_target[y, x] > 0.5:
            orientation = (int(gradient_orientation_target[y, x]) + 360) % 360
            for r, fi in Rtable[orientation]:
                x1 = int(x - r * np.cos((fi)))
                y1 = int(y - r * np.sin((fi)))
                if 0 <= x1 < target_gray.shape[1] and 0 <= y1 < target_gray.shape[0]:
                    hough_space[y1, x1] += 1


max_coords = np.where(hough_space == hough_space.max())
print(max_coords)

for y, x in zip(max_coords[0], max_coords[1]):
    plt.imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    plt.plot([x], [y], '*', color='r')
plt.show()

for peak_y, peak_x in zip(*max_coords):
    cv2.circle(target_img, (peak_x, peak_y), 2, (0, 0, 255), -1)
    for point in contours[0]:
        px, py = point[0]
        x_offset = px - center_x
        y_offset = py - center_y
        new_x = peak_x + x_offset
        new_y = peak_y + y_offset
        if 0 <= new_x < target_img.shape[1] and 0 <= new_y < target_img.shape[0]:
            target_img[new_y, new_x] = [0, 255, 0]
 

cv2.namedWindow("Pattern Contour", cv2.WINDOW_NORMAL)
cv2.imshow("Pattern Contour", img)

cv2.namedWindow("Hough Space", cv2.WINDOW_NORMAL)
cv2.imshow("Hough Space", (hough_space * 255).astype(np.uint8))

cv2.namedWindow("Detected Position", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Position", target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()