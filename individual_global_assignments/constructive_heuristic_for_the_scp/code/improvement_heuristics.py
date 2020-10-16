# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os

# Import SCPInstance from Utilities
from utilities import SCPInstance

# IH1: First Improvement
def ih1(ch_results_array, scp_instances_dir, random_seed=42, max_iterations=1000000, patience=1000, tabu_thr=10):
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

    
    # Tabu Search for Columnns: Columns in the solution begin with value -1, the rest with 0; until the value of tabu_thr the column is usable
    tabu_columns = [0 for i in range(problem_matrix.shape[1])]
    for col_idx, _ in enumerate(tabu_columns):
        if col_idx in initial_solution:
            tabu_columns[col_idx] = -1
    

    # Initialise variables
    # Current solution
    current_solution = initial_solution.copy()
    
    # Current cost
    current_cost = 0
    for col in current_solution:
        current_cost += scp_instance.scp_instance_column_costs[col]

    # If current cost is different from the processed cost, we will take this last into account
    if current_cost != initial_cost:
        initial_cost = current_cost
    

    # Initialise number of iterations
    nr_iteration = 1

    # Initialise number of iterations in patience
    nr_patience = 1
    

    # Begin algorithm
    while (nr_iteration <= max_iterations) and (nr_patience <= patience):
        print("Iteration: {} | Patience: {}".format(nr_iteration, nr_patience))
        # Generate a neighbour-solution
        # Create a condition that decides that we have found a proper neighbour
        valid_neighbours = False
        while valid_neighbours != True:
            # Choose a column of our solution that will be swaped, we are assuming Uniform PD
            swap_column = np.random.choice(a=current_solution)

            # Neighbours
            # First we verify the rows that this column covers
            rows_covered_by_swap_column = list()
            for row, row_value in enumerate(problem_matrix[:, swap_column]):
                if row_value == 1:
                    rows_covered_by_swap_column.append(row)
            
            # Then we subtract 1 from the row freq column to check if there are columns that suddenly become unavailable
            rows_freq_solution_after_swap = rows_freq_solution.copy()
            for row in rows_covered_by_swap_column:
                rows_freq_solution_after_swap[row] -= 1
            
            
            # Now we have to verify if there is some row that is unavailable
            uncovered_rows_after_swap = list()
            for row, row_freq in enumerate(rows_freq_solution_after_swap):
                if row_freq <= 0:
                    uncovered_rows_after_swap.append(row)


            # We have two options:
            neighbours = list()
            # Option 1: All rows are covered
            if len(uncovered_rows_after_swap) == 0:
                # If all the rows are covered this column is redundant!
                new_solution = current_solution.copy()
                new_solution.remove(swap_column)
                # Therefore, we found a valid neighbour solution
                valid_neighbours = True

            # Option 2: We have uncovered rows
            else:
                # This way, the neighbours must contain at least the uncovered rows after swap
                # We check the available columns first
                for col, col_avail in enumerate(columns_availability):
                    # print("Column: ", col)
                    # It must respect the constraints related with the availability and tabu search
                    if (col != swap_column) and (col_avail == 1) and (tabu_columns[col] >= 0) and (tabu_columns[col] <= 10):
                        # Check the rows that this col covers
                        temp_rows_col_covers = list()
                        for row, row_value in enumerate(problem_matrix[:, col]):
                            if row_value == 1:
                                temp_rows_col_covers.append(row)
                        
                        # Now let's check this column covers all the uncovered rows
                        try:
                            for row in uncovered_rows_after_swap:
                                temp_rows_col_covers.remove(row)
                            
                            neighbours.append(row)
                        
                        except:
                            neighbours = neighbours.copy()
                

                # Now we should have neighbours (or not)
                if len(neighbours) >= 0:
                    valid_neighbours = True
        

        # If we just removed a column
        if len(neighbours) == 0:
            # Let's compute the cost of the new solution
            new_cost = np.sum([scp_instance.scp_instance_column_costs[col] for col in new_solution])
            # It is better, our current cost is the new cost
            if new_cost < current_cost:
                current_cost = new_cost
                
                # And our current solution is the new_solution
                current_solution = new_solution

                # Updates
                # Columns Availability
                for col in current_solution:
                    columns_availability[col] = 0
                
                # Rows Frequency Solution
                rows_freq_solution = [0 for i in range(problem_matrix.shape[0])]
                for col in current_solution:
                    for row_idx, _ in enumerate(rows_freq_solution):
                        rows_freq_solution[row_idx] += problem_matrix[row_idx, col]
                
                # Iterations
                nr_iteration += 1
                nr_patience = 0
            
            else:
                nr_iteration += 1
                nr_patience += 1
        

        # Otherwise, we have neighbours
        else:
            # Here, we choose the FIRST neighbour we find
            chosen_neighbour = np.random.choice(a=neighbours)
            
            # Let's perform the swap
            new_solution = current_solution.copy()
            new_solution.remove(swap_column)
            if chosen_neighbour not in new_solution:
                new_solution.append(chosen_neighbour)

            # Compute the new cost
            new_cost = np.sum([scp_instance.scp_instance_column_costs[col] for col in new_solution])
            # It is better, our current cost is the new cost
            if new_cost < current_cost:
                current_cost = new_cost
                
                # And our current solution is the new_solution
                current_solution = new_solution

                # Updates
                # Columns Availability
                for col in current_solution:
                    columns_availability[col] = 0
                
                # Do not forget the swap column!
                columns_availability[swap_column] = 1

                # Tabu Search
                # The neighbour
                tabu_columns[chosen_neighbour] += 1
                # The swap column
                tabu_columns[swap_column] += 1
                # Check if we have to reset Tabu Search
                for col, tabu_value in enumerate(tabu_columns):
                    if tabu_value >= 20:
                        tabu_columns[col] = 0

                
                # Rows Frequency Solution
                rows_freq_solution = [0 for i in range(problem_matrix.shape[0])]
                for col in current_solution:
                    for row_idx, _ in enumerate(rows_freq_solution):
                        rows_freq_solution[row_idx] += problem_matrix[row_idx, col]
                
                # Iterations
                nr_iteration += 1
                nr_patience = 0
            
            else:
                nr_iteration += 1
                nr_patience += 1
        

    # Final solutions
    final_cost = current_cost
    final_solution = current_solution.copy()

    return initial_solution, initial_cost, final_solution, final_cost

# IH2: Best Improvement
def ih2(ch_results_array, scp_instances_dir, random_seed=42, max_iterations=1000000, patience=1000):
    """
    IH2: Choose the best neighbour.
    """

    # Set Numpy random seed
    np.random.seed(seed=random_seed)
    pass

# IH3: Hybrid Approach
def ih3(ch_results_array, scp_instances_dir, random_seed=42, max_iterations=1000000, patience=1000):
    """
    IH3: A (tentative) hybrid approach.
    """

    # Set Numpy random seed
    np.random.seed(seed=random_seed)
    pass