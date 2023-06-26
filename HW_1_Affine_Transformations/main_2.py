import math
from matplotlib import pyplot as plt
import numpy as np

def original_points():
    level_1_r_pt = [1,1]
    level_2_r_pt = [1,2]
    level_3_r_pt = [1,3]
    level_4_r_pt = [1,4]
    level_5_r_pt = [1,5]
    level_6_r_pt = [1,6]
    level_7_r_pt = [1,7]
    level_1_l_pt = [2,1]
    level_2_l_pt = [2,2]
    level_3_l_pt = [2,3]
    level_4_l_pt = [2,4]
    level_5_l_pt = [2,5]
    level_6_l_pt = [2,6]
    level_7_l_pt = [2,7]
    return np.array([level_1_r_pt, level_2_r_pt, level_3_r_pt, level_4_r_pt, level_5_r_pt, level_6_r_pt,
                     level_7_r_pt,level_1_l_pt, level_2_l_pt, level_3_l_pt, level_4_l_pt, level_5_l_pt, 
                     level_6_l_pt, level_7_l_pt])

def program_and_plot_ladder(points):

    level_1_r_pt = points[0]
    level_2_r_pt = points[1]
    level_3_r_pt = points[2]
    level_4_r_pt = points[3]
    level_5_r_pt = points[4]
    level_6_r_pt = points[5]
    level_7_r_pt = points[6]
    level_1_l_pt = points[7]
    level_2_l_pt = points[8]
    level_3_l_pt = points[9]
    level_4_l_pt = points[10]
    level_5_l_pt = points[11]
    level_6_l_pt = points[12]
    level_7_l_pt = points[13]

    vertical_lines_r_x = [
        level_1_r_pt[0], 
        level_2_r_pt[0], 
        level_3_r_pt[0],
        level_4_r_pt[0],
        level_5_r_pt[0],
        level_6_r_pt[0],
        level_7_r_pt[0]
    ]

    vertical_lines_l_x = [
        level_1_l_pt[0],
        level_2_l_pt[0],
        level_3_l_pt[0],
        level_4_l_pt[0],
        level_5_l_pt[0],
        level_6_l_pt[0],
        level_7_l_pt[0],
    ]
            
    vertical_lines_r_y = [
        level_1_r_pt[1], 
        level_2_r_pt[1], 
        level_3_r_pt[1],
        level_4_r_pt[1],
        level_5_r_pt[1],
        level_6_r_pt[1],
        level_7_r_pt[1]
    ]

    vertical_lines_l_y = [
        level_1_l_pt[1],
        level_2_l_pt[1],
        level_3_l_pt[1],
        level_4_l_pt[1],
        level_5_l_pt[1],
        level_6_l_pt[1],
        level_7_l_pt[1],
    ]
    
    horizontal_lines_x = [
        level_1_r_pt[0],
        level_1_l_pt[0], 
        level_2_r_pt[0],
        level_2_l_pt[0], 
        level_3_r_pt[0],
        level_3_l_pt[0],
        level_4_r_pt[0],
        level_4_l_pt[0],
        level_5_r_pt[0],
        level_5_l_pt[0],
        level_6_r_pt[0],
        level_6_l_pt[0],
        level_7_r_pt[0],
        level_7_l_pt[0]
    ]

    horizontal_lines_y = [
        level_1_r_pt[1],
        level_1_l_pt[1],
        level_2_r_pt[1],
        level_2_l_pt[1], 
        level_3_r_pt[1],
        level_3_l_pt[1],
        level_4_r_pt[1],
        level_4_l_pt[1],
        level_5_r_pt[1],
        level_5_l_pt[1],
        level_6_r_pt[1],
        level_6_l_pt[1],
        level_7_r_pt[1],
        level_7_l_pt[1]
    ]


    for i in range(0, int(len(vertical_lines_r_x))):
        plt.plot(vertical_lines_r_x[i:i+2], vertical_lines_r_y[i:i+2])

    for i in range(0, int(len(vertical_lines_r_x))):
        plt.plot(vertical_lines_l_x[i:i+2], vertical_lines_l_y[i:i+2])

    for i in range(0, len(horizontal_lines_x), 2):
        plt.plot(horizontal_lines_x[i:i+2], horizontal_lines_y[i:i+2])

    plt.show()


def convert_2d_point_to_homogeneous(point):
    new_point = [0,0,1]
    new_point[0] = point[0]
    new_point[1] = point[1]
    # new_point[2] = weight
    return new_point

def convert_homogeneous_point_to_2d(point):
    new_point = [0,0]
    new_point[0] = point[0]/point[2]
    new_point[1] = point[1]/point[2]

    return new_point

def perspective_transform_matrix(lx, ly):
    matrix = np.array([
        [1,0,0],
        [0,1,0],
        [lx, ly, 1]
    ])
    return matrix

def translation_transform_matrix(tx, ty):
    matrix = np.array([
        [1,0,tx],
        [0,1,ty],
        [0,0,1]
    ])
    return matrix

def rotation_transform_matrix(angle):
    matrix = np.array([
        [math.cos(angle*(math.pi/180)),math.sin(angle*(math.pi/180)),0],
        [-math.sin(angle*(math.pi/180)),math.cos(angle*(math.pi/180)),0],
        [0,0,1]
    ])
    return matrix

def scaling_transform_matrix(sx, sy):
    matrix = np.array([
        [sx, 0,0],
        [0,sy,0],
        [0,0,1]
    ])
    return matrix 

def matrix_vector_multiplication(matrix, vector):
    new_x = matrix[0][0]*vector[0] + matrix[1][0]*vector[1] + matrix[2][0]*vector[2]
    new_y = matrix[0][1]*vector[0] + matrix[1][1]*vector[1] + matrix[2][2]*vector[2]
    new_z = matrix[0][2]*vector[0] + matrix[1][2]*vector[1] + matrix[2][2]*vector[2]

    new_point_vector = [new_x, new_y, new_z]
    return new_point_vector

def matrix_vector_multiplication_perspective(matrix, vector):
    new_x = matrix[0][0]*vector[0] + matrix[0][0]*vector[1] + matrix[0][0]*vector[2]
    new_y = matrix[1][0]*vector[0] + matrix[1][1]*vector[1] + matrix[1][2]*vector[2]
    new_z = matrix[2][0]*vector[0] + matrix[2][1]*vector[1] + matrix[2][2]*vector[2]

    new_point_vector = [new_x, new_y, new_z]
    return new_point_vector

def rotate_ladder(original_points, angle):
    rotation_matrix = rotation_transform_matrix(angle)
    new_points = np.zeros(original_points.shape).astype(np.float64)
    for count, point in enumerate(original_points):
        homogeneous_point = convert_2d_point_to_homogeneous(point)
        converted_point = matrix_vector_multiplication(matrix=rotation_matrix, vector=homogeneous_point)
        back_to_2d_point = convert_homogeneous_point_to_2d(converted_point)
        new_points[count] = back_to_2d_point
    
    return new_points

def scale_ladder(original_points, sx, sy):
    scaling_matrix = scaling_transform_matrix(sx, sy)
    new_points = np.zeros(original_points.shape).astype(np.float64)
    for count, point in enumerate(original_points):
        homogeneous_point = convert_2d_point_to_homogeneous(point)
        converted_point = matrix_vector_multiplication(matrix=scaling_matrix, vector=homogeneous_point)
        back_to_2d_point = convert_homogeneous_point_to_2d(converted_point)
        new_points[count] = back_to_2d_point
    
    return new_points

def translate_ladder(original_points, tx, ty):
    translating_matrix = translation_transform_matrix(tx, ty)
    new_points = np.zeros(original_points.shape).astype(np.float64)
    for count, point in enumerate(original_points):
        homogeneous_point = convert_2d_point_to_homogeneous(point)
        converted_point = matrix_vector_multiplication_perspective(matrix=translating_matrix, vector=homogeneous_point)
        back_to_2d_point = convert_homogeneous_point_to_2d(converted_point)
        new_points[count] = back_to_2d_point
    
    return new_points


def perspective_transform_ladder(original_points, lx, ly):
    perspective_matrix = perspective_transform_matrix(lx, ly)
    new_points = np.zeros(original_points.shape).astype(np.float64)
    for count, point in enumerate(original_points):
        homogeneous_point = convert_2d_point_to_homogeneous(point)
        converted_point = matrix_vector_multiplication_perspective(matrix=perspective_matrix, vector=homogeneous_point)
        back_to_2d_point = convert_homogeneous_point_to_2d(converted_point)
        new_points[count] = back_to_2d_point
    
    return new_points

if __name__ == "__main__":
    print("HELLO")
    points_array_ladder_original = original_points()
    # program_and_plot_ladder(points_array_ladder_original)

    # points_ladder_rotation = rotate_ladder(points_array_ladder_original, 30)
    # program_and_plot_ladder(points_ladder_rotation)

    # points_ladder_scale_2 = scale_ladder(points_array_ladder_original, 2, 5)
    # program_and_plot_ladder(points_ladder_scale_2)

    points_ladder_translate = translate_ladder(points_array_ladder_original, 5, 5)
    program_and_plot_ladder(points_ladder_translate)
    points_ladder_perspective = perspective_transform_ladder(points_array_ladder_original, 0, -5)
    points_ladder_translate = translate_ladder(points_ladder_perspective, 5, 5)
    program_and_plot_ladder(points_ladder_translate)
    # points_ladder_scale_2 = scale_ladder(points_ladder_perspective, 2, 5)
    # program_and_plot_ladder(points_ladder_scale_2)
    # points_ladder_rotation = rotate_ladder(points_ladder_perspective, 30)
    # program_and_plot_ladder(points_ladder_rotation)

    # points_ladder_scale_1 = scale_ladder(points_array_ladder_original, 5, 1)
    # program_and_plot_ladder(points_ladder_scale_1)

    # points_ladder_scale_2 = scale_ladder(points_array_ladder_original, 1, 5)
    # program_and_plot_ladder(points_ladder_scale_2)

    # points_ladder_translate = translate_ladder(points_array_ladder_original, 5, 5)
    # program_and_plot_ladder(points_ladder_translate)

    # points_ladder_perspective = perspective_transform_ladder(points_array_ladder_original, 0, -5)
    # program_and_plot_ladder(points_ladder_perspective)

    # points_ladder_perspective = perspective_transform_ladder(points_array_ladder_original, -5, 0)
    # program_and_plot_ladder(points_ladder_perspective)

    # points_ladder_perspective = perspective_transform_ladder(points_array_ladder_original, 3, 3)
    # program_and_plot_ladder(points_ladder_perspective)
