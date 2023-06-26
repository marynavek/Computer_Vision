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
    new_point = [0,0,1,1]
    new_point[0] = point[0]
    new_point[1] = point[1]
    # new_point[2] = weight
    return new_point

def convert_homogeneous_point_to_2d(point):
    new_point = [0,0]
    new_point[0] = point[0]/point[2]
    new_point[1] = point[1]/point[2]

    return new_point


def create_tranformation_matrix(distance):
    division = -1/distance
    if distance == 0:
        division = -1
    
    matrix = np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,division,division, 0]
    ])
    return matrix

def perspective_transform(original_points):
    Z = 4
    f = 0.02
    new_points = np.zeros(original_points.shape).astype(np.float64)
    # print(original_points)
    for count, point in enumerate(original_points):
        # print(point)
        new_x = (point[0]/(Z*f))
        new_y = (point[1]/(Z*f))
        # print([new_x,new_y])
        new_points[count] = [new_x,new_y]

    return new_points

def get_homogeneous_points(original_points):
    # print(len(original_points))
    new_points =  np.zeros((len(original_points), 3)).astype(np.float64)
    for count, point in enumerate(original_points):
        # print(count)
        homogeneous_point = convert_2d_point_to_homogeneous(point)
        matrix = create_tranformation_matrix(homogeneous_point[1])
        # print(matrix)
        x_h = 0
        y_h = 0
        weight = 0
        for value in matrix[0]:
            x_h += homogeneous_point[0]*value
        for value in matrix[1]:
            y_h += homogeneous_point[1]*value
        for value in matrix[2]:
            weight += homogeneous_point[2]*value
        
        new_point = [x_h, y_h, weight]
        new_points[count] = new_point
    return new_points

def convert_back_to_2d(homogeneous_points):
    new_points =  np.zeros((len(homogeneous_points), 2)).astype(np.float64)
    for count, point in enumerate(homogeneous_points):
        new_point = convert_homogeneous_point_to_2d(point)
        new_points[count] = new_point
    return new_points


def ladder_projection(original_ladder):
    h_points = get_homogeneous_points(original_points=original_ladder)
    final_points = convert_back_to_2d(h_points)
    return final_points



if __name__ == "__main__":
    print("HELLO")
    points_array_ladder_original = original_points()
    projection_of_ladder = ladder_projection(points_array_ladder_original)

    # print(points_array_ladder_original)
    # h_points = get_homogeneous_points(original_points=points_array_ladder_original)
    # # print(h_points)
    # final_points = convert_back_to_2d(h_points)
    # print(final_points)
    program_and_plot_ladder(projection_of_ladder)
    