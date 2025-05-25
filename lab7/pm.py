import numpy as np
import matplotlib.pyplot as plt

def appendimages(im1,im2):    
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    
    return np.concatenate((im1,im2), axis=1)    
    
def plot_matches(im1,im2,matches,unfiltered_matches=False):
    colors=['r','g','b','c','m','y']
    im3 = appendimages(im1,im2)
    
    plt.figure()
    plt.imshow(im3, cmap='gray')
    
    cols1 = im1.shape[1]
    for i, m in enumerate(matches):
        plt.plot([m[1][1],m[2][1]+cols1],[m[1][0],m[2][0]],colors[i%6], linewidth=0.5)
        
    for match in matches:
        _, a, b = match
        print(f"{a=} {b=}")
        plt.scatter([a[1], b[1]+cols1], [a[0], b[0]], color='red', s=4)

    if unfiltered_matches:
        for match in unfiltered_matches:
            _, a, b = match
            plt.scatter([a[1], b[1]+cols1], [a[0], b[0]], color='yellow', s=1)
    plt.axis('off') 
