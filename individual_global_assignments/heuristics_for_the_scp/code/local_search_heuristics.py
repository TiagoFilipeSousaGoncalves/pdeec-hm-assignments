# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os
import random
import math

# Import SCPInstance from Utilities
from utilities import SCPInstance

# LSH1: Simulated Annealing (We add a patience and a tabu procedure)
def lsh1(ih_results_array, scp_instances_dir, random_seed=42, initial_temperature=1000, final_temperature=0.1, cooling_ratio_alpha=0.99, tabu_thr=10):

    # Set Numpy random seed
    np.random.seed(seed=random_seed)
    
    # Read the array information
    # SCP Instance Filename is at index 0
    scp_instance_filename = ih_results_array[0]

    # Processed Solution is at index 3
    initial_solution = ih_results_array[3]

    # Processed Cost is at index 4
    initial_cost = ih_results_array[4]
    


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
    iteration = 1

    # Initialise temperature patience
    temperature_patience = 0

    # History: Save the Iteration and the Cost Value in that iteration to obtain good plots
    history = list()
    history.append([iteration, initial_cost, initial_temperature])

    # Current temperature
    current_temperature = initial_temperature
    
    

    # Begin algorithm
    while current_temperature > final_temperature:
        # Select a random neighbour
        # Valid neighbour finding success variable
        valid_neighbour = list()

        # Let's generate more neighbours at a time
        while len(valid_neighbour) < 20:
            # We generate a possible candidate neighbour based on a swap
            swap_column = np.random.choice(a=current_solution)

            # Neighbours
            # Should we "swap or remove" to find a new neighbour?
            swap_or_remove_or_insert = np.random.choice(a=[0, 1, 2])

            # Option 1: All rows are covered and our previous redundancy routines had bugs or were not effective
            if swap_or_remove_or_insert == 0:
                # If all the rows are covered this column is redundant!
                candidate_neighbour = current_solution.copy()
                candidate_neighbour.remove(swap_column)
                # Therefore, we found a valid neighbour solution
                # print("removed column")
                valid_neighbour.append(candidate_neighbour)

            # Option 2: It's a swap!
            elif swap_or_remove_or_insert == 1:
                # Check availability
                candidate_neighbour_columns = list()
                for col, col_avail in enumerate(columns_availability):
                    if col_avail == 1:
                        candidate_neighbour_columns.append(col)
                
                # Create a procedure to find a proper candidate column
                candidate_column_found = False
                while candidate_column_found != True:
                    candidate_column = np.random.choice(a=candidate_neighbour_columns)
                    if (tabu_columns[candidate_column] >= 0) and (tabu_columns[candidate_column] <= 10):
                        candidate_column_found = True
                
                # Generate candidate neighbour
                candidate_neighbour = current_solution.copy()
                candidate_neighbour.remove(swap_column)
                candidate_neighbour.append(candidate_column)
                valid_neighbour.append(candidate_neighbour)
            
            # Option 3: It's an insert!
            elif swap_or_remove_or_insert == 2:
                 # Check availability
                 candidate_neighbour_columns = list()
                 for col, col_avail in enumerate(columns_availability):
                     if col_avail == 1:
                         candidate_neighbour_columns.append(col)
                
                 # Create a procedure to find a proper candidate column
                 candidate_column_found = False
                 while candidate_column_found != True:
                     candidate_column = np.random.choice(a=candidate_neighbour_columns)
                     if (tabu_columns[candidate_column] >= 0) and (tabu_columns[candidate_column] <= 10):
                         candidate_column_found = True
                
                # Generate candidate neighbour
                 candidate_neighbour = current_solution.copy()
                 # candidate_neighbour.remove(swap_column)
                 candidate_neighbour.append(candidate_column)
                 valid_neighbour.append(candidate_neighbour)

        

        # First check if all neighbours keep universitality
        possible_neighbours = list()
        for neigh_idx, neigh in enumerate(valid_neighbour):
            rows_covered_by_neighbour = [0 for i in range(problem_matrix.shape[0])]
            for row, _ in enumerate(rows_covered_by_neighbour):
                for col in neigh:
                    if problem_matrix[row, col] == 1:
                        rows_covered_by_neighbour[row] = 1
            if int(np.sum(rows_covered_by_neighbour)) == int(problem_matrix.shape[0]):
                possible_neighbours.append(neigh)
        
        # We have possible neighbours! (check the universitality of the solution)
        if len(possible_neighbours) > 0:
            possible_neighbours_costs = list()
            for neigh in possible_neighbours:
                neigh_cost = np.sum([scp_instance.scp_instance_column_costs[col] for col in neigh])
                possible_neighbours.append(neigh_cost)
            
            # We choose the best neighbour
            best_neighbour = possible_neighbours[np.argmin(possible_neighbours_costs)]
            best_neighbour_cost = possible_neighbours_costs[np.argmin(possible_neighbours_costs)]

            cost_difference = current_cost - best_neighbour_cost
            
            # Now, evaluate costs
            if cost_difference > current_cost:
                current_solution = best_neighbour.copy()
                current_cost = best_neighbour_cost

            
            else:
                # Probability threshold
                if random.uniform(0, 1) < math.exp(cost_difference / current_temperature):
                    current_solution = best_neighbour.copy()
                    current_cost = best_neighbour_cost


        # Temperature update
        # rint(temperature_patience)
        current_temperature *= cooling_ratio_alpha
        # temperature_patience = 0
        # print("Temperature decreased from {} to {}.".format(current_temperature, current_temperature*cooling_ratio_alpha))
        print("Initial Cost: {} | Current Cost: {}".format(initial_cost, current_cost))

        # Updates
        # Columns Availability
        for col, _ in enumerate(columns_availability):
            if col in current_solution:
                columns_availability[col] = 0
            else:
                columns_availability[col] = 1
        

        # Tabu Search
        for col, col_tabu in enumerate(tabu_columns):
            if col in current_solution:
                tabu_columns[col] = -1
            elif col in best_neighbour and col not in current_solution:
                tabu_columns[col] += 1
                if tabu_columns[col] == tabu_thr:
                    tabu_columns[col] = 0

        
        # Rows Frequency Solution
        rows_freq_solution = [0 for i in range(problem_matrix.shape[0])]
        for col in current_solution:
            for row_idx, _ in enumerate(rows_freq_solution):
                rows_freq_solution[row_idx] += problem_matrix[row_idx, col]
        

        # Iterations
        iteration += 1
        
        # History
        history.append([iteration, current_cost, current_temperature])

        # Temperature Patience
        temperature_patience += 1


    # Final
    final_solution = current_solution.copy()
    final_cost = current_cost.copy()
    print("Initial Cost: {} | Final Cost: {}".format(initial_cost, final_cost))

    return initial_solution, initial_cost, final_solution, final_cost, history



# LSH2: Local Search VNS/Hill Climber
def lsh2(ih_results_array, scp_instances_dir, random_seed=42, initial_temperature=1000, final_temperature=0.001, cooling_ratio_alpha=0.99, tabu_thr=10):
    pass