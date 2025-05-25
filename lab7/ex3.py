import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming
from ex1 import H, read_gray, plot_points
from pm import plot_matches
from scipy.ndimage import maximum_filter


def keep_top_n_keypoints(matrix, N):
    flat = matrix.flatten()
    if N >= len(flat):
        return matrix
    threshold = np.partition(flat, -N)[-N]
    return np.where(matrix >= threshold, matrix, 0)


def non_maximal_suppression(keypoints, N=1000):
    keypoints = keep_top_n_keypoints(keypoints, N)
    max_f = maximum_filter(keypoints, size=3)
    suppressed = np.where((keypoints == max_f) & (keypoints > 0), keypoints, 0)
    kp_cnt = int(np.count_nonzero(suppressed))
    return suppressed, kp_cnt


def get_keypoints_with_31_neighbors(keypoints, margin=16):
    pts = []
    h, w = keypoints.shape
    for y in range(margin, h - margin):
        for x in range(margin, w - margin):
            if keypoints[y, x] > 0:
                pts.append(((x, y), keypoints[y, x], None, 0))
    return pts


def add_centroid_and_orientation(image, keypoints):
    img = image.astype(np.float32)
    out = []
    for (x, y), response, _, _ in keypoints:
        # patch radius 3 for moment
        m00 = m10 = m01 = 0.0
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if dx*dx + dy*dy > 9:
                    continue
                px = x + dx
                py = y + dy
                val = img[py, px]
                m00 += val
                m10 += dx * val
                m01 += dy * val
        if m00 <= 0:
            continue
        cx = m10 / m00
        cy = m01 / m00
        theta = np.arctan2(m01, m10)
        out.append(((x, y), response, (cx, cy), theta))
    return out


def draw_keypoints_with_orientation(image, keypoints_info, scale=15):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for (x, y), _, _, theta in keypoints_info:
        x0, y0 = int(x), int(y)
        x1 = int(x + scale * np.cos(theta))
        y1 = int(y + scale * np.sin(theta))
        cv2.arrowedLine(vis, (x0, y0), (x1, y1), (0,255,0), 1, tipLength=0.3)
        cv2.circle(vis, (x0, y0), 2, (0,0,255), -1)
    plt.imshow(vis)
    plt.axis('off')
    plt.show()


def brief_descriptor(image, keypoints):
    positions = np.loadtxt("orb_descriptor_positions.txt").astype(np.int8)
    p0 = positions[:, :2]
    p1 = positions[:, 2:]
    img = cv2.GaussianBlur(image, (5,5), 0)
    descs = []
    valid_kp = []
    for (x, y), response, _, theta in keypoints:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        desc = np.zeros(256, dtype=np.uint8)
        ok = True
        for j in range(256):
            r0, c0 = p0[j]
            r1, c1 = p1[j]
            dx0 = int(np.round(c0 * cos_t - r0 * sin_t))
            dy0 = int(np.round(c0 * sin_t + r0 * cos_t))
            dx1 = int(np.round(c1 * cos_t - r1 * sin_t))
            dy1 = int(np.round(c1 * sin_t + r1 * cos_t))
            x0, y0 = x + dx0, y + dy0
            x1, y1 = x + dx1, y + dy1
            if not (0 <= x0 < image.shape[1] and 0 <= y0 < image.shape[0] and
                    0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]):
                ok = False
                break
            if img[y0, x0] < img[y1, x1]:
                desc[j] = 1
        if ok:
            valid_kp.append(((x,y), response, None, theta))
            descs.append(desc)
    if len(descs) == 0:
        return [], np.array([])
    return valid_kp, np.array(descs)


def get_hamming_distance(a, b):
    return int(np.count_nonzero(a != b))


def fast_detector_harris(image):
    threshold = 44 if image.dtype == np.uint8 else 0.15
    mask_size = 7
    hmat = H(image, mask_size)
    img = image.astype(np.float32)
    offsets = [
        (0,-3),(1,-3),(2,-2),(3,-1),(3,0),(3,1),(2,2),(1,3),
        (0,3),(-1,3),(-2,2),(-3,1),(-3,0),(-3,-1),(-2,-2),(-1,-3)
    ]
    kp = np.zeros(img.shape, dtype=np.float32)
    cnt = 0
    h, w = img.shape
    for y in range(3, h-3):
        for x in range(3, w-3):
            c = img[y,x]
            br = sum(1 for dx,dy in offsets if img[y+dy,x+dx] > c+threshold)
            dr = sum(1 for dx,dy in offsets if img[y+dy,x+dx] < c-threshold)
            if br >= 12 or dr >= 12:
                kp[y,x] = hmat[y,x]
                cnt += 1
    return kp, cnt


def orb(image):
    kp_map, cnt = fast_detector_harris(image)
    print(f"FAST Harris kp count: {cnt}")
    sup_kp, cnt2 = non_maximal_suppression(kp_map)
    print(f"After NMS kp count: {cnt2}")
    pts = get_keypoints_with_31_neighbors(sup_kp)
    pts.sort(key=lambda x: x[1], reverse=True)
    pts = pts[:200]
    orient_kp = add_centroid_and_orientation(image, pts)
    draw_keypoints_with_orientation(image, orient_kp)
    return brief_descriptor(image, orient_kp)


def get_matches(descriptor1, descriptor2, ratio_thresh=0.75):
    k1, d1 = descriptor1
    k2, d2 = descriptor2
    matches = []

    for i, desc1 in enumerate(d1):
        distances = []
        for j, desc2 in enumerate(d2):
            distances.append((get_hamming_distance(desc1, desc2), k2[j][0]))
        distances.sort(key=lambda x: x[0])

        if len(distances) >= 2:
            d1_, d2_ = distances[0][0], distances[1][0]
            if d1_ / d2_ < ratio_thresh:
                x1, y1 = k1[i][0]
                x2, y2 = distances[0][1]
                # tutaj swapujemy:
                matches.append((d1_, (y1, x1), (y2, x2)))

    matches.sort(key=lambda x: x[0])
    return matches[:20], matches


if __name__ == "__main__":
    # img1 = read_gray('budynek1.jpg')
    # img2 = read_gray('budynek2.jpg')
    
    # img1 = read_gray('fontanna1.jpg')
    # img2 = read_gray('fontanna2.jpg')

    # img1 = read_gray('left_panorama.jpg')
    # img2 = read_gray('right_panorama.jpg')
            
    img1 = read_gray('eiffel1.jpg')
    img2 = read_gray('eiffel2.jpg')

    desc1 = orb(img1)
    desc2 = orb(img2)
    m20, mall = get_matches(desc1, desc2)
    plot_matches(img1, img2, m20, mall)
    plt.show()