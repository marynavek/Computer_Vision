import numpy as np
import cv2
from matplotlib import pyplot as plt


def edge_detector_vertical_off_on():
    filter = np.array([
        [-2,-2,2,2],
        [-2,-2,2,2],
        [-2,-2,2,2],
        [-2,-2,2,2]
    ])
    return filter

def edge_detector_vertical_on_off():
    filter = np.array([
        [2,2,-2,-2],
        [2,2,-2,-2],
        [2,2,-2,-2],
        [2,2,-2,-2]
    ])
    return filter

def edge_detector_horizontal_off_on():
    filter = np.array([
        [-2,-2,-2,-2],
        [-2,-2,-2,-2],
        [2,2,2,2],
        [2,2,2,2]
    ])
    return filter

def edge_detector_horizontal_on_off():
    filter = np.array([
        [2,2,2,2],
        [2,2,2,2],
        [-2,-2,-2,-2],
        [-2,-2,-2,-2]
    ])
    return filter


def edge_detector_vertical_on_off_on():
    filter = np.array([
        [2,-2,-2,2],
        [2,-2,-2,2],
        [2,-2,-2,2],
        [2,-2,-2,2]
    ])
    return filter

def edge_detector_horizontal_on_off_on():
    filter = np.array([
        [2,2,2,2],
        [-2,-2,-2,-2],
        [-2,-2,-2,-2],
        [2,2,2,2]
    ])
    return filter

def edge_detector_vertical_off_on_off():
    filter = np.array([
        [-2,2,2,-2],
        [-2,2,2,-2],
        [-2,2,2,-2],
        [-2,2,2,-2]
    ])
    return filter

def edge_detector_horizontal_off_on_off():
    filter = np.array([
        [-2,-2,-2,-2],
        [2,2,2,2],
        [2,2,2,2],
        [-2,-2,-2,-2]
    ])
    return filter

def edge_detector_diagonal_1():
    filter = np.array([
        [2,-2,-4,-6],
        [2,2,-2,-4],
        [2,2,2,-2],
        [2,2,2,2]
    ])
    return filter

def edge_detector_diagonal_2():
    filter = np.array([
        [2,2,2,2],
        [-2,2,2,2],
        [-4,-2,2,2],
        [-6,-4,-2,2]
    ])
    return filter


def edge_detector_kirsh():
    filter = np.array([
        [5,-3,-3],
        [5,0,-3],
        [5,-3,-3]
    ])
    return filter

def edge_detector_prewitt():
    filter = np.array([
        [1,1,-1],
        [1,-2,-1],
        [1,1,-1]
    ])
    return filter

def apply_filter(image, filter, threshold):
    x,y  = image.shape
    filtered = np.zeros(shape=(x,y))
    for i in range(x - 2):
        for j in range(y - 2):
            gx = np.sum(np.multiply(filter, image[i:i + 4, j:j + 4])) 
            # gy = np.sum(np.multiply(Gy, image[i:i + 3, j:j + 3])) 
            filtered[i + 1, j + 1] = np.sqrt(gx ** 2) 
    
    for i in range(x):
        for j in range(y):
            if filtered[i,j]>255*threshold:
                filtered[i,j]=255
            else:
                filtered[i,j]=0

    return filtered

def kirsch_filter(image, threshold):
    x,y = image.shape
    list=[]
    kirsch = np.zeros((x,y))
    for i in range(2,x-1):
        for j in range(2,y-1):
            d1 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d2 = np.square((-3) * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d3 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d4 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d5 = np.square(5 * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d6 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d7 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] - 
                  3 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d8 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            
            list=[d1, d2, d3, d4, d5, d6, d7, d8]
            kirsch[i,j]= int(np.sqrt(max(list)))
                         
    for i in range(x):
        for j in range(y):
            if kirsch[i,j]>255*threshold:
                kirsch[i,j]=255
            else:
                kirsch[i,j]=0
    return kirsch

def threshold_filter_results(filtered_image, threshold):
    x,y = filtered_image.shape
    thresholded_filter_image = np.zeros(filtered_image.shape)
    for i in range(x):
        for j in range(y):
            if filtered_image[i,j]>255*threshold:
                thresholded_filter_image[i,j]=255
            else:
                thresholded_filter_image[i,j]=0
    return thresholded_filter_image

if __name__ == "__main__":
    print("HELLO")
    # natual_im_1.jpg
    # synthetic_im_3.jpeg
    # reduced_version.jpg
    image_path = "/Users/marynavek/Projects/ComputerVision/natural_scene.png"
        
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (256,256))

    plt.imshow(image, cmap='gray')
    plt.show()

    filter_1 = edge_detector_vertical_off_on()
    filter_2 = edge_detector_vertical_on_off()
    filter_3 = edge_detector_vertical_off_on_off()
    filter_4 = edge_detector_vertical_on_off_on()
    filter_5 = edge_detector_horizontal_off_on()
    filter_6 = edge_detector_horizontal_on_off()
    filter_7 = edge_detector_horizontal_off_on_off()
    filter_8 = edge_detector_horizontal_on_off_on()
    filter_9 = edge_detector_diagonal_1()
    
    filtered_image_1 = cv2.filter2D(image, -1, filter_1)
    filtered_image_2 = cv2.filter2D(image, -1, filter_2)
    filtered_image_3 = cv2.filter2D(image, -1, filter_3)
    filtered_image_4 = cv2.filter2D(image, -1, filter_4)
    filtered_image_5 = cv2.filter2D(image, -1, filter_5)
    filtered_image_6 = cv2.filter2D(image, -1, filter_6)
    filtered_image_7 = cv2.filter2D(image, -1, filter_7)
    filtered_image_8 = cv2.filter2D(image, -1, filter_8)
    filtered_image_9 = cv2.filter2D(image, -1, filter_9)

    t_image_1 = threshold_filter_results(filtered_image_1, 0.2)
    t_image_2 = threshold_filter_results(filtered_image_2, 0.2)
    t_image_3 = threshold_filter_results(filtered_image_3, 0.2)
    t_image_4 = threshold_filter_results(filtered_image_4, 0.2)
    t_image_5 = threshold_filter_results(filtered_image_5, 0.2)
    t_image_6 = threshold_filter_results(filtered_image_6, 0.2)
    t_image_7 = threshold_filter_results(filtered_image_7, 0.2)
    t_image_8 = threshold_filter_results(filtered_image_8, 0.2)
    t_image_9 = threshold_filter_results(filtered_image_9, 0.2)

    superimsposed_image = t_image_1 + t_image_2 + t_image_3 + t_image_4 + t_image_5 + t_image_6 + t_image_7 + t_image_8 + t_image_9

    plt.imshow(superimsposed_image, cmap='gray')
    plt.show()

    # t_image_1 = threshold_filter_results(filtered_image_1, 0.2)
    # plt.title("Threshold = 0.2")
    # plt.imshow(t_image_1, cmap='gray')
    # plt.show()

    # t_image_2 = threshold_filter_results(filtered_image_1, 0.4)
    # plt.title("Threshold = 0.4")
    # plt.imshow(t_image_2, cmap='gray')
    # plt.show()
    # t_image_3 = threshold_filter_results(filtered_image_1, 0.6)
    # plt.title("Threshold = 0.6")
    # plt.imshow(t_image_3, cmap='gray')
    # plt.show()
    # t_image_4 = threshold_filter_results(filtered_image_1, 0.8)
    # plt.title("Threshold = 0.8")
    # plt.imshow(t_image_4, cmap='gray')
    # plt.show()

    # filter_2 = edge_detector_vertical_on_off()
    
    # filtered_image_2 = cv2.filter2D(image, -1, filter_2)
    

    # plt.imshow(filtered_image_2, cmap='gray')
    # plt.show()

    # t_image_1 = threshold_filter_results(filtered_image_2, 0.2)
    # plt.title("Threshold = 0.2")
    # plt.imshow(t_image_1, cmap='gray')
    # plt.show()

    # t_image_2 = threshold_filter_results(filtered_image_2, 0.4)
    # plt.title("Threshold = 0.4")
    # plt.imshow(t_image_2, cmap='gray')
    # plt.show()
    # t_image_3 = threshold_filter_results(filtered_image_2, 0.6)
    # plt.title("Threshold = 0.6")
    # plt.imshow(t_image_3, cmap='gray')
    # plt.show()
    # t_image_4 = threshold_filter_results(filtered_image_2, 0.8)
    # plt.title("Threshold = 0.8")
    # plt.imshow(t_image_4, cmap='gray')
    # plt.show()
