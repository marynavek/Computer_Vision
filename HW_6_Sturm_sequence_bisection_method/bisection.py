import math
import numpy as np


def extract_d_and_e_values_from_matrix(original_matrix):
    d_values = np.diag(original_matrix)
    e_values = np.diag(original_matrix, 1)
    return d_values, e_values

def calculate_P_alpha(alpha, d_n, P_n_1, P_n_2, e_value_1):
    first_argument = (d_n - alpha)
    P_n = first_argument*P_n_1 - (e_value_1 ** 2)*P_n_2
    return P_n

def calculate_P_0():
    return 1

def calculate_P_1(alpha, d_n):
    P_1 = d_n-alpha
    return P_1

def calculate_P_values_in_interval(original_matrix, min_interval, max_interval):
    number_of_iterations = len(original_matrix)
    d_values, e_values = extract_d_and_e_values_from_matrix(original_matrix)

    min_P_vals = []

    for i in range(number_of_iterations+1):
        if i == 0:
            P_alpha_n = calculate_P_0()
        elif i == 1:
            P_alpha_n = calculate_P_1(min_interval, d_values[i-1])
        else:
            print()
            P_alpha_n = calculate_P_alpha(min_interval, d_values[i-1], min_P_vals[len(min_P_vals)-1], min_P_vals[len(min_P_vals)-2], e_values[i-2])
        
        min_P_vals.append(P_alpha_n)


    max_P_vals = []
    for i in range(number_of_iterations+1):
        if i == 0:
            P_alpha_n = calculate_P_0()
        elif i == 1:
            P_alpha_n = calculate_P_1(max_interval, d_values[i-1])
        else:
            P_alpha_n = calculate_P_alpha(max_interval, d_values[i-1], max_P_vals[len(max_P_vals)-1], max_P_vals[len(max_P_vals)-2], e_values[i-2])
        
        max_P_vals.append(P_alpha_n)

    return min_P_vals, max_P_vals
#1 - positive, -1 - negative
def convert_P_sequence_to_agrements(P_values_sequence):
    agrements_list = []
    for count, value in enumerate(P_values_sequence):
        if value > 0:
            agrements_list.append(1)
        elif value < 0:
            agrements_list.append(-1)
        else:
            agrements_list.append(agrements_list[count-1])
    return agrements_list


def determine_number_of_agrements(P_values_sequence):
    number_of_agreements = 0
    for count, value in enumerate(P_values_sequence):

        if count == 0:
            continue
        prev_value = P_values_sequence[count-1]

        if value == prev_value:
            number_of_agreements+=1

    return number_of_agreements

def get_total_number_of_eigen_values_in_interval(k1, k2):
    return k1 + k2

def calcualte_number_of_eigen_vals_in_interval(original_matrix, min_value, max_value):
    min_P_vals, max_P_vals = calculate_P_values_in_interval(original_matrix,min_value,max_value)
    
    min_agreem = convert_P_sequence_to_agrements(min_P_vals)
    max_agreem = convert_P_sequence_to_agrements(max_P_vals)

    k1 = determine_number_of_agrements(min_agreem)
    k2 = determine_number_of_agrements(max_agreem)

    return get_total_number_of_eigen_values_in_interval(k1,k2)

def calcualte_number_of_eigen_vals_in_interval_bisection(original_matrix, min_value, max_value):
    min_P_vals, max_P_vals = calculate_P_values_in_interval(original_matrix,min_value,max_value)
    
    min_agreem = convert_P_sequence_to_agrements(min_P_vals)
    max_agreem = convert_P_sequence_to_agrements(max_P_vals)

    k1 = determine_number_of_agrements(min_agreem)
    k2 = determine_number_of_agrements(max_agreem)
    if k1 == k2 and k1 != 0:
        return True
    else: return False



def bisection(original_matrix, min_value, max_value):
    number_of_eigen_vals = calcualte_number_of_eigen_vals_in_interval(original_matrix, min_value, max_value)
    if number_of_eigen_vals == 0:
        return False
    else:
        c = min_value
        while((max_value - min_value >= 0.01)):
            c = (max_value+min_value)/2

            smaller_interval_select = calcualte_number_of_eigen_vals_in_interval_bisection(original_matrix, c, max_value)
            if smaller_interval_select == True:
                min_value = c
                print("Lower")
            else:
                print("Higher")
                max_value = c

                
        return min_value, max_value





if __name__ == "__main__":

    A = np.array([
        [-2,1,0,0],
        [1,-2,1,0],
        [0,1,-2,1],
        [0,0,1,-2]
    ])

    use_matrix = None

    in_1 = input("Do you want to use pre-programmed matrix or input? Note: input has to be in tri-diagonalized form and be netered by rows. Enter Y or N\n")
    if in_1 == 'Y':
        number_of_rows = input("Enter number of rows in your matrix (number)\n")
        
        new_matrix = np.zeros((int(number_of_rows),int(number_of_rows))).astype(np.float16)
        for i in range(int(number_of_rows)):
            row = input(f"Please enter the numbers of the row {i+1} separating with comma (,)\n")
            
            separated = row.split(',')
            new_row = []
            for count, r in enumerate(separated):
                new_matrix[i][count] = float(r)
        use_matrix = new_matrix
    else:
        use_matrix = A

    print("Please enter the interval to search next!")
    in_2 = input("First, enter miminum value of the interval\n")
    minimum = float(in_2)

    in_3 = input("Now, enter maximim value of the interval\n")
    maximum = float(in_3)


    answer = bisection(use_matrix, minimum, maximum)
    if answer:
        min, max = answer
        print(f'One of the eigen values in interval [{minimum},{maximum}] lies between [{min}, {max}]')
    
    
    