import math
import numpy as np
import time


def find_angle(p,q, original_matrix):
    if original_matrix[p-1][p-1] == original_matrix[q-1][q-1]:
        angle = (original_matrix[p-1][q-1]/abs(original_matrix[p-1][q-1]))*(math.pi/4)
    else:
        tan_2_angle = (2*original_matrix[p-1][q-1])/(original_matrix[p-1][p-1]-original_matrix[q-1][q-1])
        angle = math.atan(tan_2_angle)/2
    
    return angle

def find_highest_value_above_diagonal(original_matrix):
    m = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    # zero out all values below and ioncluding diagonal
    upper_values = np.triu(original_matrix, 1)
    minimum = 0
    min_row = None
    min_col = None
    for col in range(upper_values.shape[0]):
        for row in range(upper_values.shape[1]):
            if abs(upper_values[col][row]) > minimum:
                minimum = abs(upper_values[col][row])
                min_col = col + 1
                min_row = row + 1
    
    return minimum, min_col, min_row


def jacobi_R_matrix(p, q, original_matrix, angle):
    new_matrix = np.identity(original_matrix.shape[0])
    new_matrix[p-1, p-1] = math.cos(angle)
    new_matrix[p-1, q-1] = math.sin(angle)
    new_matrix[q-1, p-1] = -math.sin(angle)
    new_matrix[q-1, q-1] = math.cos(angle)
    return new_matrix

def obtain_transform_matrix(original_matrix, transform_matrix):
    first_step = np.dot(transform_matrix,original_matrix)
    second_step = np.dot(first_step, np.transpose(transform_matrix))
    return second_step

if __name__ == "__main__":

    A = np.array([
        [1,0,2],
        [0,2,1],
        [2,1,1]
    ])

    # A = np.array([
    #     [1,2,1,2],
    #     [2,2,-1,1],
    #     [1,-1,1,1],
    #     [2,1,1,1]
    # ])

    intermediate_result = A
    count = 0
    start = time.time()
    largest_value_above_diagonal, p, q = find_highest_value_above_diagonal(intermediate_result)
    while largest_value_above_diagonal > 0:
        angle = find_angle(p,q, intermediate_result)
        jacobi_transform = jacobi_R_matrix(p, q, intermediate_result, angle)
        intermediate_result = obtain_transform_matrix(intermediate_result, jacobi_transform)

        print("Transform:")
        print(jacobi_transform)
        print("Intermediate Result:")
        print(intermediate_result)

        for row in range(intermediate_result.shape[0]):
            for col in range(intermediate_result.shape[1]):
                if abs(intermediate_result[row][col]) < 10 ** -3:
                    intermediate_result[row][col] = 0


        print("Intermediate Result Simplified:")
        print(intermediate_result)
        count += 1
        largest_value_above_diagonal, p, q = find_highest_value_above_diagonal(intermediate_result)
    finish_time = time.time()
    print("Final Results:")
    print(intermediate_result)
    print("Total: ",finish_time - start," sec")
    print(f"Total iterations: {count}")