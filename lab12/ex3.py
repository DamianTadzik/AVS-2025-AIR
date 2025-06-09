import cv2
import numpy as np
from ex1 import hog
from sklearn import svm
import joblib  # for loading saved model

# Load pretrained SVM model
clf = joblib.load('svm_hog_model.joblib')

# Parameters
window_size = (64, 128)
step_size = 16  # pixels to move window on x and y
threshold = 0.5  # decision function threshold to consider positive detection

def sliding_window(image, step_size, window_size):
    """Generator for sliding window over the image."""
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_pedestrians(image, clf, window_size=(64, 128), step_size=16):
    detections = []

    for (x, y, window) in sliding_window(image, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue  # skip incomplete windows at edges

        feat = hog(window)
        feat = feat.reshape(1, -1)
        pred = clf.decision_function(feat)  # distance to hyperplane
        if pred >= threshold:
            detections.append((x, y, window_size[0], window_size[1], pred))

    return detections

if __name__ == '__main__':
    # Load your test image here; replace filename if needed
    image_path = 'testImage1.png'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image {image_path}")
        exit(-1)

    # Convert BGR to RGB for HOG
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections = detect_pedestrians(image_rgb, clf, window_size, step_size)

    # Draw rectangles on original BGR image for visualization
    for (x, y, w, h, score) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"{score[0]:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Pedestrian Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
