import numpy as np
from matplotlib import pyplot as plt
import math
from itertools import product

def total_number_of_terms(r,n):
    top = math.factorial(n+r)
    bottom = math.factorial(r)*math.factorial(n)
    return int(top/bottom)

def define_weights_vector(n):
    weight_symbol = "w"
    weight_matrix = []
    for i in range(n+1):
        term_number = str(i+1)
        n_symbol = weight_symbol+term_number
        weight_matrix.append(n_symbol)

    return weight_matrix

def define_weights_vector_real_number(n):
    weight_matrix = []
    weight_symbol = "w"
    for i in range(1, n+2):
        term_number = str(i)
        n_symbol = weight_symbol+term_number
        addItem = {n_symbol: i}
        weight_matrix.append(addItem)

    return weight_matrix

def define_functions_vector(n):
    function_symbol = "x"
    function_matrix = []
    for i in range(n):
        term_number = str(i+1)
        n_symbol = function_symbol+term_number
        function_matrix.append(n_symbol)

    return function_matrix

def get_all_combinations(r,n):
    all_combinations = []
    for combination in product(range(1, n+1), repeat=r):
        all_combinations.append(combination)

    final_result = []
    for i in all_combinations:
        array = list(i)
        array.sort()
        if array not in final_result:
            final_result.append(array)
    return final_result

def clean_up_combinations(initial_combinations):
    final_combinations = []
    for combinations in initial_combinations:
        for combination in combinations:
            if combination not in final_combinations:
                final_combinations.append(combination)
    final_combinations.sort()
    return final_combinations

def get_all_terms(n,r, weights, functions):
    all_combinations = []    
    #get_all_combinations_for_summations                                                        
    for i in range(1,r+1):
        for d in range(1, n+1):
            combinations = get_all_combinations(i,d)
            all_combinations.append(combinations)
    final_combinations = clean_up_combinations(all_combinations)
    all_terms = []
    all_terms_2 = []
    for combination in final_combinations:
        weight_term = ""
        function_term = ""
        weight_number = ""
        
        for x in combination:
            weight_term += weights[x-1]
            weight_number += str(x)
            function_term += functions[x-1]
        combined_term = weight_term+function_term
        all_terms.append(combined_term)
        weight_combined = "w"+weight_number
        # function_combined = "x"+function_number
        combined_t = weight_combined + function_term
        all_terms_2.append(combined_t)

    all_terms.append(weights[n])
    all_terms.sort()
    all_terms_2.append(weights[n])
    
    return all_terms, all_terms_2


def insert_real_numbers_in_place_in_function(real_numbers_weigths, all_terms):
    final_results = []
    for term in all_terms:
        if "x" not in term:
            for item in real_numbers_weigths:
                if term in item.keys():
                    real_weight = item[term]
            combine_weights_with_functions = str(real_weight)
        else:
            separate_elements = [term[i:i+2] for i in range(0, len(term), 2)]
            real_weight = 1
            functions = ""
            for element in separate_elements:
                if "w" in element:
                    
                    for item in real_numbers_weigths:
                        if element in item.keys():
                            real_weight_temp = item[element]
                            real_weight *= real_weight_temp
                else:
                    functions += element
            combine_weights_with_functions = str(real_weight)+functions
        final_results.append(combine_weights_with_functions)

    return " + ".join(final_results)
        
        



if __name__ == "__main__":
    n = 3
    r = 3
    weights = define_weights_vector(n)
    functions = define_functions_vector(n)

    terms = total_number_of_terms(r,n)

    all_terms_v1, all_terms_v2 = get_all_terms(n,r, weights, functions)

    print(f"Total number of terms {terms}")
    print("All found terms:")
    print(" + ".join(all_terms_v2))
    print(f"The number of found terms is {len(all_terms_v2)}")

    real_numbers_weigths = define_weights_vector_real_number(n)
    print("We initialize the weigth vector with number in range from 1 to n")
    print(real_numbers_weigths)
    final_result_with_numbers = insert_real_numbers_in_place_in_function(real_numbers_weigths, all_terms_v1)
    print("The final function appears as:")
    print(final_result_with_numbers)
