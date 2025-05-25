import cv2
import numpy as np


prev = cv2.imread("../lab2/highway/input/in000300.jpg")
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)


feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
p0 = cv2.goodFeaturesToTrack(prev, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(prev)


for i in range(1, 1100):
    curr = cv2.imread('../lab2/highway/input/in%06d.jpg' % i)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)

    if p1 is not None:
        st_flattened = st.flatten()
        p1_flattened = p1.reshape(-1, 2)
        p0_flattened = p0.reshape(-1, 2)
        good_new = p1_flattened[st_flattened == 1]
        good_old = p0_flattened[st_flattened == 1]

        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (88, 25, 155), 1)
            curr = cv2.circle(curr, (int(a), int(b)), 2, (88, 25, 155), -1)

        # img = np.copy(curr)
        img = cv2.add(curr, mask)
        cv2.namedWindow("Lucas-Kanade", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Lucas-Kanade", 1000, 1000)
        cv2.imshow("Lucas-Kanade", img)
        cv2.waitKey(1)

    prev = curr.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
