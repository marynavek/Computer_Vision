import math
import numpy as np
import time

def get_x_vector(original_matrix, r):
    transposed_matrix = np.transpose(original_matrix)
    x_vector = transposed_matrix[r]

    return x_vector

def get_y_vector(x_vector, r):
    r = r+2
    total_lenght_y = len(x_vector)-r
    y_vector = np.zeros((total_lenght_y)).astype(np.float16)

    for count, position in enumerate(range(r, len(x_vector))):
        y_vector[count] = x_vector[position]

    return y_vector

def get_y_vector_transpose_multiply(y_vector):
    return np.dot(y_vector, np.transpose(y_vector))

def get_s_value(r,x_vector):
    r = r-1
    y_vector = get_y_vector(x_vector, r)
    
    r = r+1
    x_r_1 = x_vector[r]
    
    y_y_t = get_y_vector_transpose_multiply(y_vector)

    s = math.sqrt(math.pow(x_r_1, 2)+ y_y_t)
    return s

def find_v_r_1(x_r_1, s):
    if x_r_1 < 0:
        return x_r_1 - s
    else:
        return x_r_1 + s
    
def get_alpha(r, x_vector):
    r = r+1

    x_r_1 = x_vector[r]

    s = get_s_value(r,x_vector)

    v_r_1 = find_v_r_1(x_r_1, s)

    alpha_square = 1/(2*s*v_r_1)
    return alpha_square, v_r_1

def get_w_vector(original_matrix, r):
    r = r-1
    x_vector = get_x_vector(original_matrix, r)

    alpha, v_r_1 = get_alpha(r, x_vector)

    y_vector_transpose = get_y_vector(x_vector, r)

    w_vector = np.zeros(original_matrix.shape[1])

    for i in range(0, r+1):
        w_vector[i] = 0

    w_vector[r+1] = v_r_1
    for count, item in enumerate(y_vector_transpose):
        w_vector[r+2+count] = item
    print(w_vector)
    return alpha, w_vector
    
def obtain_transform_matrix(original_matrix, transform_matrix):
    first_step = np.dot(transform_matrix,original_matrix)
    second_step = np.dot(first_step, np.transpose(transform_matrix))
    return second_step

def multiply_matrix_1_d_with_its_transpose(matrix):
    new_matrix = np.zeros((len(matrix), len(matrix)))
    matrix_transpose = np.transpose(matrix)

    for row in range(len(matrix_transpose)):
        for col in range(len(matrix)):
            new_matrix[row][col] = matrix_transpose[row]*matrix[col]

    return new_matrix


if __name__ == "__main__":
    # A = np.array([
    #     [1,2,1,2],
    #     [2,2,-1,1],
    #     [1,-1,1,1],
    #     [2,1,1,1]
    # ])

    A = np.array([
        [1,0,2],
        [0,2,1],
        [2,1,1]
    ])
    

    N = A.shape[0]
    number_of_rotations = int(((N-1)*(N-2))/2)
    intermediate_result = A

    if number_of_rotations != 1:
        number_of_rotations = number_of_rotations-1
    total_rotation = 0
    r_count = 0
    start = time.time()
    while total_rotation < number_of_rotations:
    # for r in range(1,number_of_rotations):
        if r_count == 0:
            r_count = 1
        r = r_count

        alpha_square, w_vector = get_w_vector(intermediate_result, r)
        w_vect_multiplication = multiply_matrix_1_d_with_its_transpose(w_vector)
        transform_matrix = np.identity(intermediate_result.shape[0]) - 2*alpha_square*w_vect_multiplication
        intermediate_result = obtain_transform_matrix(intermediate_result, transform_matrix).astype(np.float16)

        print("Transform:")
        print(transform_matrix)
        print("Intermediate Result:")
        print(intermediate_result)

        for row in range(intermediate_result.shape[0]):
                for col in range(intermediate_result.shape[1]):
                    if abs(intermediate_result[row][col]) < 10 ** -3:
                        intermediate_result[row][col] = 0


        print("Intermediate Result Simplified:")
        print(intermediate_result)

        total_rotation += 1
        r_count +=1
    finish_time = time.time()

    print("Final Results:")
    print(intermediate_result)

    print("Total: ",finish_time - start," sec")