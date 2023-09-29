def greedy_test_minimization(test_suite, test_coverage):
    """
    A greedy algorithm for test suite minimization.
    
    Args:
        test_suite (list): A list of test cases to be minimized.
        test_coverage (function): A function that takes a single test case
                                  as input and returns the set of covered
                                  elements by that test case.
    
    Returns:
        list: A minimized test suite that maintains the same coverage
              as the original test suite.
    """
    # Create a set of all uncovered elements
    uncovered_elements = set.union(*[test_coverage(test_case) for test_case in test_suite])
    
    # Initialize the minimized test suite as an empty list
    minimized_test_suite = []
    
    # Iterate until all elements are covered or no more test cases are available
    while uncovered_elements and test_suite:
        # Find the test case that covers the most uncovered elements
        best_test_case = max(test_suite, key=lambda x: len(test_coverage(x) & uncovered_elements))
        
        # Add the best test case to the minimized test suite and remove it from the original test suite
        minimized_test_suite.append(best_test_case)
        test_suite.remove(best_test_case)
        
        # Update the set of uncovered elements
        uncovered_elements -= test_coverage(best_test_case)
    
    return minimized_test_suite

test_cases = {
    "test1": [1, 2],
    "test2": [2, 3],
    "test3": [2, 3, 4],
    "test4": [3, 4],
    "test5": [1, 5],
    "test6": [1, 2],
    "test7": [2, 3],
    "test8": [2, 3, 4],
    "test9": [3, 4],
    "test10": [1, 5],
    "test11": [1, 2],
    "test12": [2, 3],
    "test13": [2, 3, 4],
    "test14": [3, 4],
    "test15": [1, 5],
    "test16": [1, 2],
    "test17": [2, 3],
    "test18": [2, 3, 4],
    "test19": [3, 4],
    "test20": [1, 5],
    "test21": [1, 2],
    "test22": [2, 3],
    "test23": [2, 3, 4],
    "test24": [3, 4],
    "test25": [1, 5]
    
}
    
#test_cases = {
#    "test1": [1, 2],
#    "test2": [1, 3, 5],
#    "test3": [2, 3, 4],
#    "test4": [3, 4],
#    "test5": [4],
#    "test6": [5],
#    "test7": [5]
#}

def test_coverage(test_case):
    """
    Returns the set of covered elements by a test case.
    """
    return set(test_case)

test_suite = list(test_cases.values())

# Print the original test suite and its coverage
print("Original test suite:")
print(test_suite)
print("Coverage of each test case:")
for test_case in test_suite:
    print(test_coverage(test_case))

# Minimize the test suite
minimized_test_suite = greedy_test_minimization(test_suite, test_coverage)

# Print the minimized test suite and its coverage
print("Minimized test suite:")
print(minimized_test_suite)
print("Coverage of each test case:")
for test_case in minimized_test_suite:
    print(test_coverage(test_case))

