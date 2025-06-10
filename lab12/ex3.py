import cv2
import numpy as np
from ex1 import hog
from sklearn import svm
import joblib  # for loading saved model

# Load pretrained SVM model
clf = joblib.load('svm_hog_model_cross_validated.joblib')

# Parameters
window_size = (64, 128)
step_size = 16  # pixels to move window on x and y
threshold = -.2  # decision function threshold to consider positive detection

def sliding_window(image, step_size, window_size):
    """Generator for sliding window over the image."""
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_pedestrians(image, clf, window_size=(64, 128), step_size=16, show_windows=False):
    detections = []
    
    heatmap = []
    xset = set()
    yset = set()

    for (x, y, window) in sliding_window(image, step_size, window_size):
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            continue

        feat = hog(window)
        feat = feat.reshape(1, -1)
        pred = clf.decision_function(feat)

        heatmap.append(pred)
        xset.add(x)
        yset.add(y)

        if show_windows:
            win_vis = window.copy()
            prob_txt = f"{pred[0]:.2f}"
            cv2.putText(win_vis, prob_txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
            cv2.imshow("Sliding Window", cv2.cvtColor(win_vis, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                break

        if pred >= threshold:
            detections.append((x, y, window_size[0], window_size[1], pred))

    heatmap = np.array(heatmap)
    print(heatmap)
    heatmap = heatmap.reshape(len(yset), len(xset))
    print(heatmap)
    print(heatmap.shape)

    return detections, heatmap

def show_heatmap_on_image(image_window, image, heatmap):
    # Normalizacja heatmapy do 0-255
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)

    # Powiększamy heatmapę do rozmiaru obrazu
    heatmap_resized = cv2.resize(heatmap_uint8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Kolorowa mapa
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    # Nałożenie heatmapy na obraz (60% obraz + 40% heatmapa)
    overlay = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    cv2.imshow(image_window, overlay)



if __name__ == '__main__':
    for image_path in ['testImage1.png', 'testImage2.png', 'testImage3.png', 'testImage4.png']:
        # image_path = 'testImage1.png'
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            exit(-1)

        # Convert BGR to RGB for HOG
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections, heatmap = detect_pedestrians(image_rgb, clf, window_size, step_size, show_windows=False)
        print(detections)

        # Draw rectangles on original BGR image for visualization
        for (x, y, w, h, score) in detections:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{score[0]:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(f"Pedestrian Detection {image_path}", image)

        show_heatmap_on_image(f"Score Heatmap {image_path}", image, heatmap)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
