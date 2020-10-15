# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os

# Import SCPInstance from Utilities
from utilities import SCPInstance

# IH1: First Improvement
def ih1(ch_results_array, scp_instances_dir, random_seed=42):
    """
    IH1: Choose the first neighbour.
    """

    # Set Numpy random seed
    np.random.seed(seed=random_seed)
    
    # Read the array information
    # SCP Instance Filename is at index 0
    scp_instance_filename = ch_results_array[0]
    
    # Processed Solution is at index 3
    initial_solution = ch_results_array[3]

    # Processed Cost is at index 4
    initial_cost = ch_results_array[4]


    # Get the SCP Instance
    scp_instance_path = os.path.join(scp_instances_dir, scp_instance_filename)

    # Load the SCP Instance Object
    scp_instance = SCPInstance(scp_instance_filename=scp_instance_path)

    # Build Row X Column Matrix
    problem_matrix = np.zeros((scp_instance.scp_number_of_rows, scp_instance.scp_number_of_columns), dtype=int)
    # Fill Problem Matrix
    for row_idx, row in enumerate(scp_instance.scp_instance_all_rows):
        for column in row:
            problem_matrix[row_idx, column-1] = 1
    
    
    # Variables in Memory: We create several memory variables that will be useful through the algorithm
    # Columns Availability: Variable To Use in Flips/Swaps, 1 if available, 0 if not available
    columns_availability = [1 for i in range(problem_matrix.shape[1])]
    for col in initial_solution:
        columns_availability[col] = 0

    
    # Rows Availability: Variable to check for rows that are covered/not covered 1 if covered, 0 if not covered
    rows_availability = [1 for i in range(problem_matrix.shape[0])]
    for row_idx, _ in enumerate(rows_availability):
        if np.sum(problem_matrix[row_idx, :]) == 0:
            rows_availability[row_idx] = 0
    
    
    # Rows Frequency Problem: Number of Times Each Row Appears in the Problem Matrix
    rows_freq_problem = [0 for i in range(problem_matrix.shape[0])]
    for row_idx, _ in enumerate(rows_freq_problem):
        rows_freq_problem[row_idx] = np.sum(problem_matrix[row_idx, :])


    # Rows Frequency Solution: Number of Times Each Row Appears in the Solution
    rows_freq_solution = [0 for i in range(problem_matrix.shape[0])]
    for col in initial_solution:
        for row_idx, _ in enumerate(rows_freq_solution):
            rows_freq_solution[row_idx] += problem_matrix[row_idx, col]
    

    # Column Frequency: Number of Times Each Column Appears in the Problem Matrix
    column_freq_problem = [0 for i in range(problem_matrix.shape[1])]
    for col_idx, _ in enumerate(column_freq_problem):
        column_freq_problem[col_idx] = np.sum(problem_matrix[:, col_idx])

    
    



    pass

# IH2: Best Improvement
def ih2(ch_results_array, scp_instances_dir, random_seed=42):
    """
    IH2: Choose the best neighbour.
    """

    # Set Numpy random seed
    np.random.seed(seed=random_seed)
    pass

# IH3: Hybrid Approach
def ih3(ch_results_array, scp_instances_dir, random_seed=42):
    """
    IH3: A (tentative) hybrid approach.
    """

    # Set Numpy random seed
    np.random.seed(seed=random_seed)
    pass