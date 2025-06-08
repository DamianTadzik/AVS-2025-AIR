import cv2
import numpy as np
import matplotlib.pyplot as plt

i_values = []
P_values = []
R_values = []
F1_values = []


DATASET = 'pedestrian'
# DATASET = 'highway'
# DATASET = 'office'

I0 = cv2.imread(DATASET + '/input/in000300.jpg')
I0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
I0 = I0.astype('int')

PREV_I = I0

for i in range (300 ,1100):
    I = cv2.imread(DATASET + '/input/in%06d.jpg' % i )
    IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    IG = IG.astype('int')


    IG_sub = abs(IG - PREV_I) / 2

    IG_sub_bin = 255*(IG_sub > 12)  

    IG_sub_bin = cv2.medianBlur(IG_sub_bin.astype('uint8'), 3)

    IG_sub_bin_eroded = cv2.erode(IG_sub_bin.astype('uint8'), np.ones((3, 3), np.uint8), iterations=1)
    # IG_sub_bin_eroded = cv2.erode(IG_sub_bin_eroded.astype('uint8'), np.ones((1, 1), np.uint8), iterations=1)

    IG_sub_bin_eroded_dilated = cv2.dilate(IG_sub_bin_eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=4)

    ## Labelling 
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(IG_sub_bin_eroded_dilated)
    I_VIS = I.astype('uint8')
    if (stats.shape[0] > 1):
        tab = stats[1:, 4]
        pi = np.argmax(tab)
        pi = pi + 1

        cv2.rectangle(I_VIS, (stats[pi,0], stats[pi, 1]), (stats[pi, 0] + stats[pi, 2], stats[pi, 1] + stats[pi, 3]), (255,0,0), 2)
        cv2.putText(I_VIS, "%f" %stats[pi, 4], (stats[pi, 0], stats[pi, 1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        cv2.putText(I_VIS, "%d" %pi, (int(centroids[pi, 0]), int(centroids[pi, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0))


    ## Showing images
    cv2.imshow ("IG", IG.astype('uint8'))
    # cv2.imshow ("IG_blurred", IG_blurred.astype('uint8'))

    cv2.imshow ("IG_sub", IG_sub.astype('uint8'))
    cv2.imshow ("IG_sub_bin", IG_sub_bin.astype('uint8'))

    cv2.imshow ("IG_sub_bin_eroded", IG_sub_bin_eroded.astype('uint8'))
    cv2.imshow ("IG_sub_bin_eroded_dilated", IG_sub_bin_eroded_dilated.astype('uint8'))

    cv2.imshow ("I_VIS", I_VIS)


    ## EVAL 
    I_groundtruth = cv2.imread(DATASET + '/groundtruth/gt%06d.png' % i )
    I_groundtruth = cv2.cvtColor(I_groundtruth, cv2.COLOR_BGR2GRAY)

    # U MNIE 
    ## 0 - background
    ## 255 - foreground 

    # U NICH
    ## 0 - background
    ## 50 - shadow
    ## 85 - beyond area of interest
    ## 170 - unknow motion
    ## 255 - foreground
    # ALE SPROWADZAM TO DO 
    ## [0 170] - background
    ## 255 - foreground

    # I_groundtruth = 255* ((I_groundtruth == 255))
    I_groundtruth = 255* ((I_groundtruth == 255) | (I_groundtruth == 170))
    cv2.imshow ("I_groundtruth", I_groundtruth.astype('uint8'))

    # TP foreground pixel is detected as foreground
    # TN background pixel is detected as foreground
    # FP foreground pixel is detected as foreground
    # FN foreground pixel is detected as foreground

    TP = np.sum((IG_sub_bin_eroded_dilated == 255) & (I_groundtruth == 255))  # True positives
    TN = np.sum((IG_sub_bin_eroded_dilated == 0) & (I_groundtruth == 0))      # True negatives
    FP = np.sum((IG_sub_bin_eroded_dilated == 0) & (I_groundtruth == 255))    # False positives
    FN = np.sum((IG_sub_bin_eroded_dilated == 255) & (I_groundtruth == 0))    # False negatives

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2*P*R / (P + R)

    print(f"P={P:.3f}\t\tR={R:.3f}\t\tF1={F1:.3f}")

    i_values.append(i)
    P_values.append(P)
    R_values.append(R)
    F1_values.append(F1)

    cv2.waitKey (16)
    PREV_I = IG


plt.figure(figsize=(10, 6))

plt.plot(i_values, P_values, label="Precision (P)", marker='o')
plt.plot(i_values, R_values, label="Recall (R)", marker='o')
plt.plot(i_values, F1_values, label="F1 Score", marker='o')

plt.xlabel("Frame id")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()
    
