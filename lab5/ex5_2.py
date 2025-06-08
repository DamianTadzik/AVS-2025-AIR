import cv2
import numpy as np
from task4_1 import of, vis_flow


def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k - 1], (0, 0), fx=0.5, fy=0.5))
    return images


def optical_flow_multiscale(I_org, J_org, max_scale=3, W2=3, dY=3, dX=3):
    IP = pyramid(I_org, max_scale)
    JP = pyramid(J_org, max_scale)

    u_or, v_or = of(I_org, J_org)

    u_total = np.zeros_like(u_or)
    v_total = np.zeros_like(v_or)

    I = IP[-1]

    for scale in range(max_scale - 1, -1, -1):
        J = JP[scale]

        u, v = of(I, J, W2, dY, dX)
        vis_flow(u, v, I.shape, "Optical Flow")
        
        
        I_new = np.copy(I)
        if scale > 0:
            for j in range(I_new.shape[0]):
                for i in range(I_new.shape[1]):
                    if 0 <= j + v[j,i] < I_new.shape[0] and 0 <= i + u[j, i] < I_new.shape[1]:
                        I_new[(j + v[j,i]).astype('int'), (i + u[j, i]).astype('int')] = I[j, i]

            I = cv2.resize(I_new, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            
        u_total += cv2.resize(u * (2 ** scale), (0, 0), fx=2 ** scale, fy=2 ** scale, interpolation=cv2.INTER_LINEAR)
        v_total += cv2.resize(v * (2 ** scale), (0, 0), fx=2 ** scale, fy=2 ** scale, interpolation=cv2.INTER_LINEAR)

    vis_flow(u_total, v_total, I_org.shape, "Optical Flow")


I = cv2.imread("I.jpg")
J = cv2.imread("J.jpg")
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
J = cv2.cvtColor(J, cv2.COLOR_RGB2GRAY)

optical_flow_multiscale(I, J)
