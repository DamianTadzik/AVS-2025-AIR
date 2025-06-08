import cv2
import numpy as np

def of(I, J, W2=3, dY=3, dX=3):

    u = np.zeros((I.shape[0], I.shape[1]), dtype=np.float32)
    v = np.zeros((I.shape[0], I.shape[1]), dtype=np.float32)

    for j in range(W2, I.shape[0] - W2):
        for i in range(W2, I.shape[1] - W2):
            IO = np.float32(I[j - W2 : j + W2 + 1, i - W2 : i + W2 + 1])

            min_distance = float("inf")
            min_x, min_y = 0, 0

            for dj in range(-dY, dY + 1):
                for di in range(-dX, dX + 1):
                    if (
                        j + dj >= W2
                        and j + dj < I.shape[0] - W2
                        and i + di >= W2
                        and i + di < I.shape[1] - W2
                    ):
                        JO = np.float32(
                            J[
                                j + dj - W2 : j + dj + W2 + 1,
                                i + di - W2 : i + di + W2 + 1,
                            ]
                        )
                        distance = np.sqrt(np.sum(np.square(JO - IO)))
                        if distance < min_distance:
                            min_distance = distance
                            min_x, min_y = di, dj

            u[j, i] = min_x
            v[j, i] = min_y

    return u, v


def vis_flow(u, v, YX, name):
    magnitude, angle = cv2.cartToPolar(u, v)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    angle = (angle * 90 / np.pi).astype(np.uint8)

    optical_flow_hsv = np.zeros((YX[0], YX[1], 3), dtype=np.uint8)
    optical_flow_hsv[..., 0] = angle
    optical_flow_hsv[..., 1] = magnitude
    optical_flow_hsv[..., 2] = 255
    optical_flow_bgr = cv2.cvtColor(optical_flow_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow(name, optical_flow_bgr)
    
    while True:
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    I = cv2.imread("I.jpg")
    J = cv2.imread("J.jpg")

    I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY).astype('uint8')
    J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY).astype('uint8')

    print(f"{I.shape[0]=}")

    u, v = of(I, J)
    vis_flow(u, v, I.shape, "Optical Flow")
