import cv2
import numpy as np
from scipy.ndimage import convolve1d

N=4

def compute_gradients(image):
    # Convert image to int32 to avoid overflow
    image = np.int32(image)
    h, w, _ = image.shape

    # Initialize gradient magnitude and orientation arrays
    magnitude = np.zeros((h, w, 3))
    orientation = np.zeros((h, w, 3))

    for ch in range(3):  # iterate over RGB channels
        # Compute horizontal and vertical gradients
        dx = convolve1d(image[:, :, ch], [-1, 0, 1], axis=1, mode='nearest')
        dy = convolve1d(image[:, :, ch], [-1, 0, 1], axis=0, mode='nearest')
        # Compute gradient magnitude and orientation
        magnitude[:, :, ch] = np.hypot(dx, dy)
        orientation[:, :, ch] = (np.rad2deg(np.arctan2(dy, dx)) % 180)

    # Select channel with strongest gradient per pixel
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

    for i in range(n_cells_y):  # iterate over cell rows
        for j in range(n_cells_x):  # iterate over cell cols
            # Extract magnitude and orientation patches
            mag_patch = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            ori_patch = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            # Accumulate gradients into histogram bins
            for m, o in zip(mag_patch.ravel(), ori_patch.ravel()):
                bin_idx = int(o // bin_width) % bins
                hist_tensor[i, j, bin_idx] += m

    return hist_tensor

def normalize_blocks(hist_tensor, block_size=2, eps=1e-6):
    n_cells_y, n_cells_x, bins = hist_tensor.shape
    blocks_y = n_cells_y - block_size + 1
    blocks_x = n_cells_x - block_size + 1
    block_features = []

    for i in range(blocks_y):  # iterate over blocks vertically
        for j in range(blocks_x):  # iterate over blocks horizontally
            # Flatten block and normalize using L2 norm
            block = hist_tensor[i:i+block_size, j:j+block_size, :].flatten()
            norm = np.linalg.norm(block, ord=2) + eps
            block_features.append(block / norm)

    return np.concatenate(block_features)

def compute_histograms_with_interpolation(magnitude, orientation, cell_size=8, bins=9):
    h, w = magnitude.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    bin_width = 180 / bins  # 20 degrees per bin

    histograms = np.zeros((n_cells_y, n_cells_x, bins))

    for i in range(n_cells_y):  # iterate over cells vertically
        for j in range(n_cells_x):  # iterate over cells horizontally
            mag_patch = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            ori_patch = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]

            for m, o in zip(mag_patch.ravel(), ori_patch.ravel()):
                # Clamp orientation to [0, 180)
                if o < 0:
                    o += 180
                elif o >= 180:
                    o -= 180

                bin_idx = int(o // bin_width)
                bin_center = bin_idx * bin_width + bin_width / 2
                dist = o - bin_center

                # Interpolate between neighboring bins
                if dist < 0:
                    lower_bin = (bin_idx - 1) % bins
                    upper_bin = bin_idx
                    weight_upper = 1 + dist / bin_width
                    weight_lower = 1 - weight_upper
                else:
                    lower_bin = bin_idx
                    upper_bin = (bin_idx + 1) % bins
                    weight_lower = 1 - dist / bin_width
                    weight_upper = 1 - weight_lower

                # Distribute magnitude between bins
                histograms[i, j, lower_bin] += m * weight_lower
                histograms[i, j, upper_bin] += m * weight_upper

    return histograms

def normalize_blocks(histograms, eps=1e-6):
    n_cells_y, n_cells_x, bins = histograms.shape
    block_size = 2  # 2x2 cells per block
    feature_vector = []

    for i in range(n_cells_y - 1):  # slide window vertically
        for j in range(n_cells_x - 1):  # slide window horizontally
            # Concatenate histograms from 2x2 block
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

def HOGpicture(w, bs=8, title='Histogram Gradients Visualization'):
    # Generate simple line filters for orientation visualization
    bim1 = np.zeros((bs, bs))
    bim1[bs//2:bs//2+1, :] = 1
    bim = np.zeros(bim1.shape + (9,))
    for i in range(9):  # rotate filter for each bin
        bim[:, :, i] = scipy.ndimage.rotate(bim1, -i*20, reshape=False, order=0) / 255

    Y, X, Z = w.shape
    w[w < 0] = 0
    im = np.zeros((bs * Y, bs * X))
    for i in range(Y):  # iterate over cell rows
        iisl = i * bs
        iisu = (i + 1) * bs
        for j in range(X):  # iterate over cell cols
            jjsl = j * bs
            jjsu = (j + 1) * bs
            for k in range(9):  # overlay line filters for each bin
                im[iisl:iisu, jjsl:jjsu] += bim[:, :, k] * w[i, j, k]
    # Display visualization
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, N*64, N*128)
    cv2.imshow(title, cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    # plt.imshow(im, cmap='gray')
    # plt.title(title)
    # plt.axis('off')
    # plt.show()

def hog(image):
    # Full HOG pipeline: gradient → histogram → normalization
    mag, ori = compute_gradients(image)
    hist = compute_histograms_with_interpolation(mag, ori, cell_size=8, bins=9)
    hog_vector = normalize_blocks(hist)
    return hog_vector

if __name__ == '__main__':
    # Load and display the input image
    # image_path = r'pedestrians/pos/per00001.ppm'
    image_path = r'pedestrians/pos/per00060.ppm'
    # image_path = r'pedestrians/pos/per00003.ppm'
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")
    cv2.namedWindow("Original Image (BGR)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Image (BGR)", N*64, N*128)
    cv2.imshow("Original Image (BGR)", image_bgr)
    # Convert BGR to RGB for further processing
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"[INFO] image size {image_bgr.shape=}")

    # Compute gradients
    magnitude, orientation = compute_gradients(image_rgb)
    print(magnitude.shape)
    # Normalize magnitude to [0,255] for visualization
    norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Show gradient magnitude images
    cv2.namedWindow("Gradient Magnitude", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gradient Magnitude", N*64, N*128)
    cv2.imshow("Gradient Magnitude", norm_magnitude)
    cv2.namedWindow("Gradient Orientation (scaled to 0-255)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gradient Orientation (scaled to 0-255)", N*64, N*128)
    cv2.imshow("Gradient Orientation (scaled to 0-255)", (orientation / 180.0 * 255.0).astype(np.uint8))

    HEAD = 10
    # HOG histogram calculation (no interpolation)
    cell_histograms = compute_cell_histograms(magnitude[:, :], orientation[:, :], cell_size=8, bins=9)
    hog_vector = normalize_blocks(cell_histograms)
    print("[INFO] HOG feature vector length (no interp):", len(hog_vector))
    print(f"[INFO] HOG first {HEAD=} elements: {hog_vector[:HEAD]=}")

    # HOG with interpolation (recommended way)
    hist_interp = compute_histograms_with_interpolation(magnitude[:, :], orientation[:, :], cell_size=8, bins=9)
    hog_features = normalize_blocks(hist_interp)
    print("[INFO] HOG feature vector length (with interp):", len(hog_features))  # expected: 3780 for 64x128
    print(f"[INFO] HOG first {HEAD=} elements: {hog_features[:HEAD]=}")

    # Show HOG visualization
    print("[INFO] Displaying HOG visualization...")
    HOGpicture(cell_histograms, bs=8, title='no-interp')
    HOGpicture(hist_interp, bs=8, title='interp')


    # Wait before closing windows
    print("[INFO] Press any key to close image windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
