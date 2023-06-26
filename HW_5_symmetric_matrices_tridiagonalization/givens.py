import math
import numpy as np
import time

def givens_R_matrix(r, p, q, original_matrix):
    tan_alpha = original_matrix[r-1][q-1]/original_matrix[r-1][p-1]
    angle = math.atan(tan_alpha)
    new_matrix = np.identity(original_matrix.shape[0])
    new_matrix[p-1, p-1] = math.cos(angle)
    new_matrix[p-1, q-1] = math.sin(angle)
    new_matrix[q-1, p-1] = -math.sin(angle)
    new_matrix[q-1, q-1] = math.cos(angle)
    return new_matrix

def calculate_p_q(r, iteration):
    p = r+1
    q = r+2+iteration
    return p, q

def obtain_transform_matrix(original_matrix, transform_matrix):
    first_step = np.dot(transform_matrix,original_matrix)
    second_step = np.dot(first_step, np.transpose(transform_matrix))
    return second_step

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
    print(number_of_rotations)
    prev_q = 0
    intermediate_result = A
    r_count = 0
    total_rotation = 0
    count_row = 0
    print(number_of_rotations)
    start = time.time()
    while total_rotation < number_of_rotations:
        if r_count == 0:
            r_count = 1
        r = r_count
        p = r_count+1

        while prev_q < N:
            q = r+2+count_row
            prev_q = q
            count_row += 1
            total_rotation += 1
            print(f"r: {r}")
            print(f"p: {p}")
            print(f"q: {q}")
            
            transform_matrix = givens_R_matrix(r, p, q, intermediate_result)
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

        
        prev_q = 0
        count_row = 0
        r_count +=1
    finish_time = time.time()

    print("Final Results:")
    print(intermediate_result)
    print("Total: ",finish_time - start," sec")