import cv2
import numpy as np
import matplotlib.pyplot as plt

i_values_mean = []
P_values_mean = []
R_values_mean = []
F1_values_mean = []

i_values_median = []
P_values_median = []
R_values_median = []
F1_values_median = []

def calculate_and_store_metrics(I_pred, I_gt, frame_id, i_values, P_values, R_values, F1_values):
    """
    Computes TP, TN, FP, FN, Precision, Recall, and F1-score.
    Prints results and appends them to given lists.
    
    Parameters:
    - I_pred: Binary predicted image (foreground = 255, background = 0)
    - I_gt: Binary ground truth image (foreground = 255, background = 0)
    - frame_id: Current frame ID
    - i_values: List to store frame indices
    - P_values: List to store Precision values
    - R_values: List to store Recall values
    - F1_values: List to store F1-score values
    """
    TP = np.sum((I_pred == 255) & (I_gt == 255))  # True Positives
    TN = np.sum((I_pred == 0) & (I_gt == 0))      # True Negatives
    FP = np.sum((I_pred == 255) & (I_gt == 0))    # False Positives
    FN = np.sum((I_pred == 0) & (I_gt == 255))    # False Negatives

    P = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
    R = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0  # F1-score

    # # Print results
    # print(f"Frame {frame_id}: P={P:.3f}  R={R:.3f}  F1={F1:.3f}")

    # Append to lists
    i_values.append(frame_id)
    P_values.append(P)
    R_values.append(R)
    F1_values.append(F1)

def plot_metrics(i_values, P_values, R_values, F1_values, title="NO-TITLE"):
    """
    Plots Precision, Recall, and F1-score over frame indices.
    
    Parameters:
    - i_values: List of frame indices
    - P_values: List of Precision values
    - R_values: List of Recall values
    - F1_values: List of F1-score values
    """
    plt.figure(figsize=(10, 6),)
    plt.plot(i_values, P_values, label="Precision (P)", marker='o')
    plt.plot(i_values, R_values, label="Recall (R)", marker='o')
    plt.plot(i_values, F1_values, label="F1 Score", marker='o')
    plt.title(title)

    plt.xlabel("Frame ID")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    # plt.show()



background = cv2.imread('pedestrians_empty_bg.jpg')
background_G = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
print(f"{background_G.shape=}")
background_G = cv2.resize(background_G, (360, 240))
print(f"{background_G.shape=}")
XX = background_G.shape[0]
YY = background_G.shape[1]



BG_mean = np.zeros((XX, YY), 'float64')
def approx_mean(I, alpha):
    global BG_mean
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('uint8')

    IG_MEAN = (abs(IG.astype('int') - BG_mean.astype('int') + 255) / 2).astype('uint8')

    IG_MEAN = cv2.medianBlur(IG_MEAN, 3)

    IG_MEAN_BIN = (255*(IG_MEAN < 148)).astype('uint8')
    
    cv2.imshow("BG_mean", BG_mean.astype('uint8'))
    cv2.imshow("IG_MEAN", IG_MEAN.astype('uint8'))
    cv2.imshow("IG_MEAN_BIN", IG_MEAN_BIN.astype('uint8'))
    
    BG_mean[:, :] = (alpha * IG.astype('float64') + (1 - alpha) * BG_mean[:, :]) / (1+alpha)
    return IG_MEAN_BIN


BG_median = np.zeros((XX, YY), np.uint8)
def approx_median(I):
    global BG_median
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('uint8')
    
    IG_MEDIAN = ((IG.astype('int') - BG_median.astype('int') + 255) / 2).astype('uint8')

    IG_MEDIAN = cv2.medianBlur(IG_MEDIAN, 3)

    IG_MEDIAN_BIN = (255*((IG_MEDIAN > 168) | (IG_MEDIAN < 92))).astype('uint8')

    cv2.imshow("BG_median", BG_median.astype('uint8'))
    cv2.imshow("IG_MEDIAN", IG_MEDIAN.astype('uint8'))
    cv2.imshow("IG_MEDIAN_BIN", IG_MEDIAN_BIN.astype('uint8'))

    BG_median += (BG_median < IG).astype('uint8')
    BG_median -= (BG_median > IG).astype('uint8')
    return IG_MEDIAN_BIN


DATASET = 'pedestrian'
# DATASET = 'highway'
DATASET = 'office'

for i in range (1 ,1100):
    I = cv2.imread('../lab2/' + DATASET + '/input/in%06d.jpg' % i )
    cv2.imshow("IG", cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('uint8'))

    I_mean = approx_mean(I, .01)
    I_median = approx_median(I)


    I_groundtruth = cv2.imread('../lab2/' + DATASET + '/groundtruth/gt%06d.png' % i )
    I_groundtruth = cv2.cvtColor(I_groundtruth, cv2.COLOR_BGR2GRAY)

    I_groundtruth = 255*((I_groundtruth == 255)) # (I_groundtruth == 170))
    cv2.imshow ("I_groundtruth", I_groundtruth.astype('uint8'))

    calculate_and_store_metrics(I_mean, I_groundtruth, i, i_values_mean, P_values_mean, R_values_mean, F1_values_mean)
    calculate_and_store_metrics(I_median, I_groundtruth, i, i_values_median, P_values_median, R_values_median, F1_values_median)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

plot_metrics(i_values_mean, P_values_mean, R_values_mean, F1_values_mean, "MEAN")
plot_metrics(i_values_median, P_values_median, R_values_median, F1_values_median, "MEDIAN")
plt.show()
