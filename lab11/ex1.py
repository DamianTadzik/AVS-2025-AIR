import cv2
import numpy as np

THRESHOLD = 36  # fixed binarization threshold
MIN_AREA = 500   # minimum area to consider object
ASPECT_RATIO_MIN = 1.6  # minimum height-to-width ratio to consider it vertical

def merge_rectangles(rects, y_threshold=20):
    """
    Heuristic to merge vertically aligned rectangles that are likely part of the same silhouette.
    Rectangles are merged if one's bottom is close to the other's top and they are horizontally aligned.
    """
    merged = []
    used = [False] * len(rects)

    for i in range(len(rects)):
        if used[i]:
            continue
        x1, y1, w1, h1 = rects[i]
        rx1, ry1, rw, rh = x1, y1, w1, h1
        used[i] = True
        for j in range(i+1, len(rects)):
            if used[j]:
                continue
            x2, y2, w2, h2 = rects[j]
            if abs((y2) - (y1 + h1)) < y_threshold and abs(x2 - x1) < max(w1, w2):  # vertically close and aligned
                rx1 = min(rx1, x2)
                ry1 = min(ry1, y2)
                rw = max(rx1 + rw, x2 + w2) - rx1
                rh = max(ry1 + rh, y2 + h2) - ry1
                used[j] = True
        merged.append((rx1, ry1, rw, rh))
    return merged

cap = cv2.VideoCapture('vid1_IR.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binarization with a fixed threshold (experimentally chosen)
    _, bin_img = cv2.threshold(G, THRESHOLD, 255, cv2.THRESH_BINARY)

    # Median filtering to remove salt-and-pepper noise
    bin_img = cv2.medianBlur(bin_img, 5)

    # Morphological closing to connect broken parts
    kernel = np.ones((5, 5), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    # Label connected components and get stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    rectangles = []
    for i in range(1, num_labels):  # skip the background (label 0)
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        if h / w < ASPECT_RATIO_MIN:
            continue
        rectangles.append((x, y, w, h))

    # Attempt to merge split silhouettes
    merged_rects = merge_rectangles(rectangles)

    # Draw the merged rectangles
    for (x, y, w, h) in merged_rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Processed', frame)
    cv2.imshow('Binary', bin_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
