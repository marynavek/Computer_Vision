def printAllKLength(set, k, storage):
 
    n = len(set)
    printAllKLengthRec(set, "", n, k, storage)
 
# The main recursive method
# to print all possible
# strings of length k
def printAllKLengthRec(set, prefix, n, k, storage):
     
    # Base case: k is 0,
    # print prefix
    if (k == 0) :
        storage.append
        return
 
    # One by one add all characters
    # from set and recursively
    # call for k equals to k-1
    for i in range(n):
 
        # Next character of input added
        newPrefix = prefix + set[i]
         
        # k is decreased, because
        # we have added a new character
        printAllKLengthRec(set, newPrefix, n, k - 1)


if __name__ == "__main__":
     

    from itertools import product
    total_rolls = []
    for roll in product([1], repeat = 1):
        total_rolls.append(roll)
    print(type(total_rolls))
    print(type(total_rolls[0]))
    new_array = []
    for i in total_rolls:
        array = list(i)
        array.sort()
        if array not in new_array:
            new_array.append(array)
    
    print(type(new_array))
    print(type(new_array[0]))

    print(new_array)
    print(new_array[0])

    # storage = []
    # print("First Test")
    # set1 = ['1', '2', '3']
    # k = 2
    # printAllKLength(set1, k, storage)
     
    # print("\nSecond Test")
    # set2 = ['1', '2', '3']
    # k = 3
    # printAllKLength(set2, k)