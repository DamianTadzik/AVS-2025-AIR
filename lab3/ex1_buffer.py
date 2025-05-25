import cv2
import numpy as np

i_mean = []
P_mean = []
R_mean = []
F1_mean = []

i_median = []
P_median = []
R_median = []
F1_median = []


iN = 0
N = 60
background = cv2.imread('pedestrians_empty_bg.jpg')
background_G = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
# print(f"{background_G.shape}")
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

    


## Source: https://www.geeksforgeeks.org/python-play-a-video-using-opencv/
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('pedestrians_input.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
    
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
    # Display the resulting frame
        cv2.imshow('Frame', frame)

        my_process(frame)
        
    # Press Q on keyboard to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
