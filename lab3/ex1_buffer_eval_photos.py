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




iN = 0
N = 60
background = cv2.imread('pedestrians_empty_bg.jpg')
background = cv2.resize(background, (360, 240))
background_G = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
print(f"{background_G.shape}")
XX = background_G.shape[0]
YY = background_G.shape[1]

BUF = np.zeros((XX, YY, N), np.uint8)
BUF[:, :, iN] = background_G
iN = iN + 1 

def my_process(I):
    global iN
    global BUF  

    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype('uint8')

    BUF[:, :, iN] = IG
    iN = iN + 1 
    if iN == N:
        iN = 0

    BUF_mean = np.mean(BUF, axis=2).astype('uint8')
    BUF_median = np.median(BUF, axis=2).astype('uint8')

    IG_MEAN = (abs(IG.astype('int') - BUF_mean.astype('int') + 0) / 2).astype('uint8')
    IG_MEDIAN = (abs(IG.astype('int') - BUF_median.astype('int') + 0) / 2).astype('uint8')

    # dodane później ale oddaje gites
    IG_MEAN = cv2.medianBlur(IG_MEAN, 3)
    IG_MEDIAN = cv2.medianBlur(IG_MEDIAN, 3)

    IG_MEAN_BIN = 255*(IG_MEAN > 12).astype('uint8')
    IG_MEDIAN_BIN = 255*(IG_MEDIAN > 12).astype('uint8')

    cv2.imshow("IG", IG)
    cv2.imshow("BUF_mean", BUF_mean)
    cv2.imshow("BUF_median", BUF_median)
    cv2.imshow("IG_MEAN", IG_MEAN)
    cv2.imshow("IG_MEDIAN", IG_MEDIAN)
    cv2.imshow("IG_MEAN_BIN", IG_MEAN_BIN)
    cv2.imshow("IG_MEDIAN_BIN", IG_MEDIAN_BIN)

    return (IG_MEAN_BIN, IG_MEDIAN_BIN)


DATASET = 'pedestrian'
# DATASET = 'highway'
# DATASET = 'office'

for i in range (300 ,1100):
    I = cv2.imread('../lab2/' + DATASET + '/input/in%06d.jpg' % i )

    (I_mean, I_median) = my_process(I)

    I_groundtruth = cv2.imread('../lab2/' + DATASET + '/groundtruth/gt%06d.png' % i )
    I_groundtruth = cv2.cvtColor(I_groundtruth, cv2.COLOR_BGR2GRAY)

    I_groundtruth = 255*((I_groundtruth == 255))
    # I_groundtruth = 255* ((I_groundtruth == 255) | (I_groundtruth == 170))
    cv2.imshow ("I_groundtruth", I_groundtruth.astype('uint8'))

    calculate_and_store_metrics(I_mean, I_groundtruth, i, i_values_mean, P_values_mean, R_values_mean, F1_values_mean)

    calculate_and_store_metrics(I_median, I_groundtruth, i, i_values_median, P_values_median, R_values_median, F1_values_median)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


plot_metrics(i_values_mean, P_values_mean, R_values_mean, F1_values_mean, "MEAN")

plot_metrics(i_values_median, P_values_median, R_values_median, F1_values_median, "MEDIAN")

plt.show()

# ## Source: https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
# # Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture('pedestrians_input.mp4')

# # Check if camera opened successfully
# if (cap.isOpened()== False):
#     print("Error opening video file")

# # Read until video is completed
# while(cap.isOpened()):
    
# # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:
#     # Display the resulting frame
#         cv2.imshow('Frame', frame)

#         my_process(frame)
        
#     # Press Q on keyboard to exit
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

# # Break the loop
#     else:
#         break

# # When everything done, release
# # the video capture object
# cap.release()

# # Closes all the frames
# cv2.destroyAllWindows()
