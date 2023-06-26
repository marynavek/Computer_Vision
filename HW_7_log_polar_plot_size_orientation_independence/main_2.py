#!/usr/bin/env python
from PIL import Image
from math import *
import numpy as np
from scipy.ndimage.interpolation import geometric_transform
import cv2
from matplotlib import pyplot as plt


def topolar(img, order=1):
    """
    Transform img to its polar coordinate representation.

    order: int, default 1
        Specify the spline interpolation order. 
        High orders may be slow for large images.
    """
    # max_radius is the length of the diagonal 
    # from a corner to the mid-point of img.
    max_radius = 0.5*np.linalg.norm( img.shape )

    def transform(coords):
        # Put coord[1] in the interval, [-pi, pi]
        theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

        # Then map it to the interval [0, max_radius].
        #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
        radius = max_radius * coords[0] / img.shape[0]

        i = 0.5*img.shape[0] - radius*np.sin(theta)
        j = radius*np.cos(theta) + 0.5*img.shape[1]
        return i,j

    polar = geometric_transform(img, transform, order=order)

    rads = max_radius * np.linspace(0,1,img.shape[0])
    angs = np.linspace(0, 2*np.pi, img.shape[1])

    return polar, (rads, angs)

if __name__ == "__main__":
    print("Hello")
    image_path = "/Users/marynavek/Projects/ComputerVision/natual_im_1.jpg"
        
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (256,256))
    radius = np.linspace(0, 1, 100)
    angle = np.linspace(0, 2*np.pi, radius.size)
    r_grid, a_grid = np.meshgrid(radius, angle)

    def polar_to_cartesian(data):
        new = np.zeros_like(data) * np.nan
        x = np.linspace(-1, 1, new.shape[1])
        y = np.linspace(-1, 1, new.shape[0])
        for i in range(new.shape[0]):
            for j in range(new.shape[1]):
                x0, y0 = x[j], y[i]
                r, a = np.sqrt(x0**2 + y0**2), np.arctan2(y0, x0)
                data_i = np.argmin(np.abs(a_grid[:, 0] - a))
                data_j = np.argmin(np.abs(r_grid[0, :] - r))
                val = data[data_i, data_j]

                if r <= 1:
                    new[i, j] = val

        return new

    new = polar_to_cartesian(image)

    # pol, (rads,angs) = topolar(image)

    plt.imshow(image, cmap='gray')
    plt.show()

    plt.imshow(new, cmap='gray', interpolation='bicubic')
    plt.show()

# img = chelsea()[...,0] / 255.
# pol, (rads,angs) = topolar(img)

# fig,ax = plt.subplots(2,1,figsize=(6,8))

# ax[0].imshow(img, cmap=plt.cm.gray, interpolation='bicubic')

# ax[1].imshow(pol, cmap=plt.cm.gray, interpolation='bicubic')

# ax[1].set_ylabel("Radius in pixels")
# ax[1].set_yticks(range(0, img.shape[0]+1, 50))
# ax[1].set_yticklabels(rads[::50].round().astype(int))

# ax[1].set_xlabel("Angle in degrees")
# ax[1].set_xticks(range(0, img.shape[1]+1, 50))
# ax[1].set_xticklabels((angs[::50]*180/3.14159).round().astype(int))

# plt.show()