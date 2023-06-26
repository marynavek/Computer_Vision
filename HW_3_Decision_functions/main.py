import numpy as np
from matplotlib import pyplot as plt

def simple_seperable_classes():
    c1_pt_1 = [-2,-1,0]
    c1_pt_2 = [-3,2,0]
    c1_pt_3 = [1,3,0]
    c1_pt_4 = [3,6,0]
    c1_pt_5 = [4,5,0]
    c1_pt_6 = [5,6,0]

    # c2_pt_1 = [-1,0,0] #make it lineraly unseparable
    c2_pt_1 = [-1,-3,0]
    c2_pt_2 = [4,-2,0]
    c2_pt_3 = [3,-1,0]
    c2_pt_4 = [5,-1,0]
    c2_pt_5 = [5,2,0]

    class_1 = np.array([c1_pt_1, c1_pt_2, c1_pt_3, c1_pt_4, c1_pt_5, c1_pt_6])
    class_2 = np.array([c2_pt_1, c2_pt_2, c2_pt_3, c2_pt_4, c2_pt_5])

    return class_1, class_2

def positive_class(point, weights):
    intermediate_value = np.dot(point, weights)
    c = 0.5
    if intermediate_value <= 0:
        new_weight = weights + c*point
        return new_weight
    else: return weights

def negative_class(point, weights):
    intermediate_value = np.dot(point, weights)
    c = 0.5
    if intermediate_value >= 0:
        new_weight = weights - c*point
        return new_weight
    else: return weights

def interation(class_1, class_2, start_weights):
    init_weigth = start_weights
    change_counter = 0
    for point in class_1:
        updated_weight = positive_class(point, init_weigth)
        if  not (updated_weight == init_weigth).all():
            init_weigth = updated_weight
            change_counter += 1

    for point in class_2:
        updated_weight = negative_class(point, init_weigth)
        if not (updated_weight == init_weigth).all():
            init_weigth = updated_weight
            change_counter += 1

    return change_counter, init_weigth
    

def plot_desicion_function(weights):
    y1_func = 10
    y2_func = -10
    x1_func = (weights[1]*(-y1_func) - weights[2])/weights[0]   
    x2_func = (weights[1]*(-y2_func) - weights[2])/weights[0]   
    
    function = np.array([[x1_func, x2_func], [y1_func, y2_func]])
    return function


if __name__ == "__main__":
    class1, class2 = simple_seperable_classes()
    start_weights = [0,0,0]
    change_counter = -1
    updated_weights = None
    list_of_functions = None
    fig, ax = plt.subplots()
    while change_counter !=0:
        if change_counter == -1:
            new_change_counter, new_weigth = interation(class1, class2, start_weights)
            change_counter = new_change_counter
            updated_weights = new_weigth
        else:
            new_change_counter, new_weigth = interation(class1, class2, updated_weights)
            change_counter = new_change_counter
            updated_weights = new_weigth
        
        function = plot_desicion_function(updated_weights)
        print(function)
        x1, y1, z1 = class1.T
        x2, y2, z2 = class2.T
        plt.scatter(x1, y1, c='red')
        plt.scatter(x2, y2, c='blue')
        plt.plot(function[0], function[1])
        plt.grid()
        plt.show()
        
        
        print(updated_weights)
        print(change_counter)

    
    fig, ax = plt.subplots()
    x1, y1, z1 = class1.T
    x2, y2, z2 = class2.T
    plt.scatter(x1, y1, c='red')
    plt.scatter(x2, y2, c='blue')

    function = plot_desicion_function(updated_weights)
    plt.plot(function[0], function[1])
    plt.grid()
    plt.show()



