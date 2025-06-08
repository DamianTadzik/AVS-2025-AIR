import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming
from ex1 import H, read_gray, plot_points
from pm import plot_matches
from scipy.ndimage import maximum_filter

def keep_top_n_keypoints(matrix, N):
    # Spłaszczamy do 1D
    flat = matrix.flatten()
    # Znajdujemy próg — N-ta największa wartość
    if N >= len(flat):
        return matrix  # nic nie zmieniamy
    threshold = np.partition(flat, -N)[-N]
    # Tworzymy nową macierz, w której wartości poniżej progu zerujemy
    result = np.where(matrix >= threshold, matrix, 0)
    return result

# def non_maximal_suppresion(keypoints):
#     print(f"{keypoints=}")
#     keypoints = keep_top_n_keypoints(keypoints, 400)
#     keypoints_supressed = np.zeros(keypoints.shape, dtype=np.float32)
#     for i in range(1, keypoints.shape[0] - 1):
#         for j in range(1, keypoints.shape[1] - 1):
#             roi = keypoints[i - 1:i + 2, j - 1:j + 2]
#             if roi[1, 1] == 0:
#                 continue
#             ind = np.unravel_index(np.argmax(roi, axis=None), roi.shape)
#             maximum_value = np.amax(roi)
#             keypoints_supressed[i - 1 + ind[0], j - 1 + ind[1]] = maximum_value
#     return keypoints_supressed

def non_maximal_suppresion(keypoints, N=1000):
    # keypoints = sort_keypoints(keypoints)
    # 1. Keep only top N keypoints
    keypoints = keep_top_n_keypoints(keypoints, N)
    # 2. Apply maximum filter
    max_filtered = maximum_filter(keypoints, size=3)
    # 3. Suppress non-maxima
    keypoints_suppressed = np.where((keypoints == max_filtered) & (keypoints > 0), keypoints, 0)
    # 4. Count non-zero keypoints
    kp_cnt = np.count_nonzero(keypoints_suppressed)
    return keypoints_suppressed, kp_cnt



# def get_keypoints_with_31_neighbors(keypoints): 
#     result = []
#     for i in range(31, keypoints.shape[0] - 31):
#         for j in range(31, keypoints.shape[1] - 31):
#             if keypoints[i, j] == 0:
#                 continue
#             result.append(((i, j), keypoints[i, j]))
#     return result

def get_keypoints_with_31_neighbors(keypoints, margin=16):
    output = []
    for i in range(margin, keypoints.shape[0] - margin):  # y
        for j in range(margin, keypoints.shape[1] - margin):  # x
            if keypoints[i, j] > 0:
                output.append(((j, i), keypoints[i, j], None, 0))  # (x=j, y=i)
    return output


def sort_keypoints(keypoints):
    return sorted(keypoints, key=lambda x: x[1], reverse=False)

def add_centroid_and_orientation(image, keypoints):
    image = image.astype("float32")
    all_keypoints_info = []
    for keypoint in keypoints:
        m00 = 0
        m10 = 0
        m01 = 0
        m11 = 0
        index = keypoint[0]
        for i in range(-3, 3):
            for j in range(-3, 3):
                if round(np.sqrt(i ** 2 + j ** 2)) > 3:
                    continue
                m00 += image[index[0] + i, index[1] + j]
                m10 += (index[0] + i) * image[index[0] + i, index[1] + j]
                m01 += (index[1] + j) * image[index[0] + i, index[1] + j]
                m11 += (index[0] + i) * (index[1] + j) * image[index[0] + i, index[1] + j]
        x = m10 / m00
        y = m01 / m00
        c = (x, y)
        theta = np.arctan2(m01, m10)
        all_keypoints_info.append((keypoint[0], keypoint[1], c, theta))
    return all_keypoints_info

# def add_centroid_and_orientation(image, keypoints):
#     patch_radius = 15
#     output = []
#     for kp in keypoints:
#         x, y = kp[0]
#         if y - patch_radius < 0 or y + patch_radius >= image.shape[0] or \
#            x - patch_radius < 0 or x + patch_radius >= image.shape[1]:
#             continue
#         patch = image[y - patch_radius:y + patch_radius + 1, x - patch_radius:x + patch_radius + 1]
#         cy, cx = np.mgrid[-patch_radius:patch_radius+1, -patch_radius:patch_radius+1]
#         m01 = np.sum(cy * patch)
#         m10 = np.sum(cx * patch)
#         theta = np.arctan2(m01, m10)
#         output.append((kp[0], kp[1], (m10, m01), theta))
#     return output

def draw_keypoints_with_orientation(image, keypoints_info, scale=20):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for kp in keypoints_info:
        (y, x), _, _, theta = kp
        pt1 = (int(x), int(y))
        pt2 = (int(x + scale * np.cos(theta)), int(y + scale * np.sin(theta)))
        cv2.arrowedLine(image_rgb, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
        cv2.circle(image_rgb, pt1, 1, (0, 0, 255), -1)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

# def brief_descriptor(image, keypoints):
#     positions = np.loadtxt("orb_descriptor_positions.txt").astype(np.int8)
#     positions_0 = positions[:, :2]
#     positions_1 = positions[:, 2:]
#     image = cv2.GaussianBlur(image, (5, 5), 0)

#     descriptor = np.zeros((len(keypoints), 256), dtype=np.uint16)
#     for i, keypoint in enumerate(keypoints):
#         x, y = keypoint[0]
#         harris_value = keypoint[1]
#         c = keypoint[2]
#         theta = keypoint[3]
#         sin_theta = np.sin(theta)
#         cos_theta = np.cos(theta)

#         # Wykonanie testu binarnego dla par punktów
#         for j in range(256):
#             position_row_0 = positions_0[j, 0]
#             position_col_0 = positions_0[j, 1]
#             position_row_1 = positions_1[j, 0]
#             position_col_1 = positions_1[j, 1]
#             position_row_0_rotated = int(np.round(position_row_0 * sin_theta + position_col_0 * cos_theta))
#             position_col_0_rotated = int(np.round(position_col_0 * cos_theta - position_row_0 * sin_theta))
#             position_row_1_rotated = int(np.round(position_row_1 * sin_theta + position_col_1 * cos_theta))
#             position_col_1_rotated = int(np.round(position_col_1 * cos_theta - position_row_1 * sin_theta))

#             # Check bounds to prevent IndexError
#             if not (0 <= y + position_row_0_rotated < image.shape[0] and
#                     0 <= x + position_col_0_rotated < image.shape[1] and
#                     0 <= y + position_row_1_rotated < image.shape[0] and
#                     0 <= x + position_col_1_rotated < image.shape[1]):
#                 continue

            
#             if image[y + position_row_0_rotated, x + position_col_0_rotated] < image[y + position_row_1_rotated, x + position_col_1_rotated]:
#                 descriptor[i, j] = 1

#     return keypoints, descriptor

def brief_descriptor(image, keypoints):
    positions = np.loadtxt("orb_descriptor_positions.txt").astype(np.int8)
    positions_0 = positions[:, :2]
    positions_1 = positions[:, 2:]
    image = cv2.GaussianBlur(image, (5, 5), 0)

    descriptor = np.zeros((len(keypoints), 256), dtype=np.uint8)

    valid_keypoints = []

    for i, keypoint in enumerate(keypoints):
        x, y = keypoint[0]
        theta = keypoint[3]

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        valid = True
        desc = np.zeros(256, dtype=np.uint8)

        for j in range(256):
            r0, c0 = positions_0[j]
            r1, c1 = positions_1[j]

            # Obrót punktów
            pr0 = int(np.round(r0 * cos_theta - c0 * sin_theta))
            pc0 = int(np.round(r0 * sin_theta + c0 * cos_theta))
            pr1 = int(np.round(r1 * cos_theta - c1 * sin_theta))
            pc1 = int(np.round(r1 * sin_theta + c1 * cos_theta))

            y0 = y + pr0
            x0 = x + pc0
            y1 = y + pr1
            x1 = x + pc1

            # Sprawdzenie granic
            if not (0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1] and
                    0 <= y1 < image.shape[0] and 0 <= x1 < image.shape[1]):
                valid = False
                break

            if image[y0, x0] < image[y1, x1]:
                desc[j] = 1

        if valid:
            valid_keypoints.append(keypoint)
            descriptor[len(valid_keypoints)-1] = desc

    descriptor = descriptor[:len(valid_keypoints)]
    return valid_keypoints, descriptor

# def get_hamming_distance(arr1, arr2):
#     return hamming(arr1, arr2) #* len(arr1)

def get_hamming_distance(arr1, arr2):
    return np.count_nonzero(arr1 != arr2)

# def fast_detector_harris(image):
#     threshold = 40
#     n = 3
#     k = 0.05
#     window_size = 32
#     mask_size = 7

#     keypoints = np.zeros(image.shape, dtype=np.float32)
#     harris_matrix = H(image, mask_size)

#     image = image.astype("float32")
#     for i in range(n, image.shape[0] - n):
#         for j in range(n, image.shape[1] - n):
#             center_pixel = image[i, j]
#             consecutive_pixels = []  

#             for dx, dy in [(0, n), (n, 0), (0, -n), (-n, 0), (n, n), (-n, -n), (n, -n), (-n, n)]:
#                 if abs(int(center_pixel) - image[i + dy, j + dx]) > threshold:
#                     consecutive_pixels.append(True)
#                 else:
#                     consecutive_pixels.append(False)

#             if sum(consecutive_pixels) >= 7:
#                 keypoints[i, j] = harris_matrix[i, j]

#     return keypoints

def fast_detector_harris(image):
    threshold = 44
    k = 0.05
    mask_size = 7

    keypoints = np.zeros(image.shape, dtype=np.float32)
    harris_matrix = H(image, mask_size)
    image = image.astype("float32")

    offsets = [
        (0, -3),  (1, -3),  (2, -2),  (3, -1),
        (3, 0),   (3, 1),   (2, 2),   (1, 3),
        (0, 3),   (-1, 3),  (-2, 2),  (-3, 1),
        (-3, 0),  (-3, -1), (-2, -2), (-1, -3)
    ]

    keypoints_counter = 0
    height, width = image.shape
    for i in range(3, height - 3):
        for j in range(3, width - 3):
            center = image[i, j]
            brighter = 0
            darker = 0
            for dx, dy in offsets:
                val = image[i + dy, j + dx]
                if val > center + threshold:
                    brighter += 1
                elif val < center - threshold:
                    darker += 1

            if brighter >= 12 or darker >= 12:
                keypoints[i, j] = harris_matrix[i, j]
                keypoints_counter += 1

    return keypoints, keypoints_counter


def orb(image):
    # FAST detector and the harris measure
    keypoints, kp_cnt = fast_detector_harris(image)
    print(f"liczba kp z fast harrisa {kp_cnt=}")

    supressed_keypoints, kp_cnt = non_maximal_suppresion(keypoints)
    print(f"liczba kp po suppresion {kp_cnt=}")

    keypoints_with_31_neighbors = get_keypoints_with_31_neighbors(supressed_keypoints)
    keypoints_with_31_neighbors.sort(key=lambda x: x[1], reverse=True)
    keypoints = keypoints_with_31_neighbors # [:200]
    fontanna1_max = [[point[0][0] for point in keypoints], [point[0][1] for point in keypoints]]
    # plot_points(image, fontanna1_max)

    keypoints_with_centroid_and_orientation = add_centroid_and_orientation(image, keypoints)
    draw_keypoints_with_orientation(image, keypoints_with_centroid_and_orientation)

    return brief_descriptor(image, keypoints_with_centroid_and_orientation)


def get_matches(descriptor1, descriptor2, ratio_thresh=0.85):
    matches = []
    keypoints1, descriptors1 = descriptor1
    keypoints2, descriptors2 = descriptor2

    for i, desc1 in enumerate(descriptors1):
        distances = []
        for j, desc2 in enumerate(descriptors2):
            distance = get_hamming_distance(desc1, desc2)
            distances.append((distance, keypoints2[j][0]))  # [0] bo (kp, descriptor)

        # Sortuj po dystansie rosnąco
        distances.sort(key=lambda x: x[0], reverse=False)

        # Ratio test — tylko jeśli mamy min. 2 kandydatów
        if len(distances) >= 2:
            d1, d2 = distances[0][0], distances[1][0]
            if d1 / d2 < ratio_thresh:
                matches.append((d1, keypoints1[i][0], distances[0][1]))

    # Posortuj wszystkie dopasowania wg jakości (dystans rosnąco)
    matches.sort(key=lambda x: x[0])
    return matches[:20], matches


if __name__ == "__main__":

    fontanna1 = read_gray('fontanna1.jpg')
    fontanna2 = read_gray('fontanna2.jpg')

    fontanna1_orb = orb(fontanna1)
    fontanna2_orb = orb(fontanna2)

    matches, all_matches = get_matches(fontanna1_orb, fontanna2_orb)
    plot_matches(fontanna1, fontanna2, matches, all_matches)
    plt.show()

    # oznacz punkty charakterystyczne na finałowym obrazku
