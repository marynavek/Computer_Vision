import math
import cv2
from matplotlib import pyplot as plt
from numpy import zeros
import numpy as np


def find_polar_for_point(point, point_of_view):
    r = math.log(math.sqrt(math.pow((point[0]-point_of_view[0]),2) + math.pow((point[1]-point_of_view[1]),2)))
    if (point[0]-point_of_view[0]) == 0:
        angle = math.atan(point[1]-point_of_view[1])
        
    else:
        angle = math.atan((point[1]-point_of_view[1])/(point[0]-point_of_view[0]))
    new_point = [r, angle]
    return new_point

def polar_to_cartesian(point):
    x = math.exp(point[0])*math.cos(point[1])
    y = math.exp(point[0])*math.sin(point[1])
    return [int(x), int(y)]

def apply_log_polar(image, point_of_view):
    new_coordinates = np.zeros(image.shape)
    for col in range(image.shape[0]):
        for row in range(image.shape[1]):
            point = [col, row]
            polar_point = find_polar_for_point(point, point_of_view)
            print(polar_point)
            print(new_coordinates.shape)
            cart = polar_to_cartesian(polar_point)
            new_coordinates[cart[0]][cart[1]] = image[col][row]

    return new_coordinates

if __name__ == "__main__":
    print("Hello")
    image_path = "/Users/marynavek/Projects/ComputerVision/natual_im_1.jpg"
        
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (256,256))
    # print(image[0])

    polar_im = apply_log_polar(image, [150,150])

    # plt.imshow(image, cmap='gray')
    # plt.show()

    # trasform = logpolar_naive(image, 0, 0, p_n=None, t_n=None)
    # plt.imshow(trasform, cmap='gray')
    # plt.show()