import cv2
import numpy as np
from scipy.ndimage import convolve1d

def compute_gradients(image):
    image = np.int32(image)
    h, w, _ = image.shape

    magnitude = np.zeros((h, w, 3))
    orientation = np.zeros((h, w, 3))

    for ch in range(3):
        dx = convolve1d(image[:, :, ch], [-1, 0, 1], axis=1, mode='nearest')
        dy = convolve1d(image[:, :, ch], [-1, 0, 1], axis=0, mode='nearest')
        magnitude[:, :, ch] = np.hypot(dx, dy)
        orientation[:, :, ch] = (np.rad2deg(np.arctan2(dy, dx)) % 180)

    max_idx = np.argmax(magnitude, axis=2)
    max_magnitude = np.take_along_axis(magnitude, max_idx[:, :, np.newaxis], axis=2).squeeze()
    max_orientation = np.take_along_axis(orientation, max_idx[:, :, np.newaxis], axis=2).squeeze()

    return max_magnitude, max_orientation

def compute_cell_histograms(magnitude, orientation, cell_size=8, bins=9):
    h, w = magnitude.shape
    n_cells_x = w // cell_size
    n_cells_y = h // cell_size
    hist_tensor = np.zeros((n_cells_y, n_cells_x, bins))
    bin_width = 180 / bins

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            mag_patch = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            ori_patch = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            for m, o in zip(mag_patch.ravel(), ori_patch.ravel()):
                bin_idx = int(o // bin_width) % bins
                hist_tensor[i, j, bin_idx] += m

    return hist_tensor

def normalize_blocks(hist_tensor, block_size=2, eps=1e-6):
    n_cells_y, n_cells_x, bins = hist_tensor.shape
    blocks_y = n_cells_y - block_size + 1
    blocks_x = n_cells_x - block_size + 1
    block_features = []

    for i in range(blocks_y):
        for j in range(blocks_x):
            block = hist_tensor[i:i+block_size, j:j+block_size, :].flatten()
            norm = np.linalg.norm(block, ord=2) + eps
            block_features.append(block / norm)

    return np.concatenate(block_features)


# === Load and process the image using cv2 ===
image_path = r'pedestrians/pos/per00001.ppm'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# === HOG processing ===
mag, ori = compute_gradients(image)
cell_hists = compute_cell_histograms(mag, ori, cell_size=8, bins=9)
hog_descriptor = normalize_blocks(cell_hists, block_size=2)

print("HOG feature vector length:", len(hog_descriptor))


import numpy as np
import math

def compute_histograms_with_interpolation(magnitude, orientation, cell_size=8, bins=9):
    h, w = magnitude.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    bin_width = 180 / bins  # 20 degrees per bin

    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            mag_patch = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            ori_patch = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            for m, o in zip(mag_patch.ravel(), ori_patch.ravel()):
                # Ensure angle in [0,180)
                if o < 0:
                    o += 180
                elif o >= 180:
                    o -= 180

                # Find bin index
                bin_idx = int(o // bin_width)
                # Middle of the bin
                bin_center = bin_idx * bin_width + bin_width / 2

                # Distance to bin center
                dist = o - bin_center

                # Determine the neighboring bins for interpolation (with wrap-around)
                if dist < 0:
                    lower_bin = (bin_idx - 1) % bins
                    upper_bin = bin_idx
                    weight_upper = 1 + dist / bin_width  # dist negative, so less than 1
                    weight_lower = 1 - weight_upper
                else:
                    lower_bin = bin_idx
                    upper_bin = (bin_idx + 1) % bins
                    weight_lower = 1 - dist / bin_width
                    weight_upper = 1 - weight_lower

                # Add weighted magnitude to histogram bins
                histograms[i, j, lower_bin] += m * weight_lower
                histograms[i, j, upper_bin] += m * weight_upper

    return histograms


def normalize_blocks(histograms, eps=1e-5):
    n_cells_y, n_cells_x, bins = histograms.shape
    block_size = 2  # 2x2 cells per block
    feature_vector = []

    for i in range(n_cells_y - 1):
        for j in range(n_cells_x - 1):
            block = np.concatenate((
                histograms[i, j, :],
                histograms[i, j + 1, :],
                histograms[i + 1, j, :],
                histograms[i + 1, j + 1, :]
            ))
            norm = np.linalg.norm(block) + eps
            normalized_block = block / norm
            feature_vector.append(normalized_block)

    return np.concatenate(feature_vector)


import scipy.ndimage
import matplotlib.pyplot as plt

def HOGpicture(w, bs=8):  # w = histograms, bs = cell size
    bim1 = np.zeros((bs, bs))
    bim1[bs//2:bs//2+1, :] = 1
    bim = np.zeros(bim1.shape + (9,))
    for i in range(9):
        bim[:, :, i] = scipy.ndimage.rotate(bim1, -i*20, reshape=False, order=0) / 255

    Y, X, Z = w.shape
    w[w < 0] = 0
    im = np.zeros((bs * Y, bs * X))
    for i in range(Y):
        iisl = i * bs
        iisu = (i + 1) * bs
        for j in range(X):
            jjsl = j * bs
            jjsu = (j + 1) * bs
            for k in range(9):
                im[iisl:iisu, jjsl:jjsu] += bim[:, :, k] * w[i, j, k]
    plt.imshow(im, cmap='gray')
    plt.title('Histogram Gradients Visualization')
    plt.axis('off')
    plt.show()


# 1. Compute max magnitude and orientation per pixel (as done before)
mag, ori = compute_gradients(image)  # your existing gradient function

# 2. Compute histograms with interpolation
hist = compute_histograms_with_interpolation(mag, ori, cell_size=8, bins=9)

# 3. Visualize (optional)
HOGpicture(hist, bs=8)

# 4. Normalize blocks and get feature vector
hog_features = normalize_blocks(hist)

print("Length of final HOG descriptor:", len(hog_features))  # should be 3780 for 64x128 image

def hog(image):
    mag, ori = compute_gradients(image)
    hist = compute_histograms_with_interpolation(mag, ori, cell_size=8, bins=9)
    hog_vector = normalize_blocks(hist)
    return hog_vector
