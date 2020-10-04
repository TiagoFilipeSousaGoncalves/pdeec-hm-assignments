# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os


# Let's create a SCPInstance class to work with our instances
class SCPInstance:
    def __init__(self, scp_instance_filename):
        # We need to store the filename to further compare results
        self.scp_instance_filename = scp_instance_filename

        # We should rename the files properties so it becomes easier to extract and manipulate properties 
        # Open the .txt file
        scp_instance_textfile = open(self.scp_instance_filename)

        # Read file lines
        # scp_file_lines = scp_instance_textfile.readlines()
        # scp_file_lines = [l.strip() for l in scp_file_lines]
        
        # Read file lines
        scp_file_lines = list()
        for i, line in enumerate(scp_instance_textfile):
            # l = [e.remove('\n') for e in line]
            # l = [e.remove('') for e in l]
            scp_file_lines.append(line.split(" "))

        # Close the file
        scp_instance_textfile.close()

        # Perform some data cleaning
        for i, line in enumerate(scp_file_lines):
            # print(line)
            line.remove('\n')
            line.remove('')

        # Let's start by storing some properties
        # Number of Rows
        self.scp_number_of_rows = int(scp_file_lines[0][0])
        
        # Number of Columns
        self.scp_number_of_columns = int(scp_file_lines[0][1])

        # Map of Costs Per Column
        attribute_map = list()
        for i, line in enumerate(scp_file_lines):
            if i > 0:
                if len(attribute_map) < self.scp_number_of_columns:
                    # attribute_map.append(int(line))
                    for att in line:
                        attribute_map.append(int(att))
                else:
                    row_index = i
                    break 
        
        self.scp_instance_column_costs = np.array(attribute_map, dtype=int)


        # Rows with that rows that contain them
        # print(scp_file_lines[-1])
        all_rows = list()
        row = list()
        for i, line in enumerate(scp_file_lines):
            if len(all_rows) < self.scp_number_of_rows:
                if i == row_index:
                    row_size = int(line[0])
                    # print(subset_size)
                    
                elif i > row_index:
                    if len(row) < row_size:
                        # subset.append(int(line))
                        for element in line:
                            row.append(int(element))
                    
                    else:
                        all_rows.append(np.array(row, dtype=int))
                        row_index = i
                        row_size = int(line[0])
                        # print(subset_size)
                        row = list()
        
        # Append last subset
        all_rows.append(np.array(row, dtype=int))
        # Assign this variable to an attribute variable of the instance 
        self.scp_instance_all_rows = all_rows


# Function CH1: Constructive Heuristics Nr. 1
def ch1(set_covering_problem_instance, post_processing=False, random_seed=42):
    """
    CH1 - picking in each constructive step first a still un-covered element and then, second, a random set that covers this element.
    Greedy Minimum Set Cover Algorithm (Greedy MSCP)
    """
    
    # Set random seed
    np.random.seed(seed=random_seed)

    # Status variables
    # Rows that we need to cover: we can update this list along the way
    rows_to_be_covered = [i for i in range(set_covering_problem_instance.scp_number_of_rows)]
    print("Rows that we need to cover: {}".format(rows_to_be_covered))
    # Number of rows that we still need to cover
    number_of_rows_to_be_covered = set_covering_problem_instance.scp_number_of_rows
    print("Number fo rows that we need to cover: {}".format(number_of_rows_to_be_covered))
    
    # Number of elements that we already covered: will be updated along the way
    nr_of_rows_covered = 0 
    print("Number of rows already covered: {}".format(nr_of_rows_covered))

    # Number of sets that cover an element: each index is an element and the value is the number of sets
    nr_of_columns_that_cover_a_row = [0 for i in range(len(rows_to_be_covered))]
    for row, columns_that_cover_row in enumerate(set_covering_problem_instance.scp_instance_all_rows):
        nr_of_columns_that_cover_a_row[row] = len(columns_that_cover_row)
    print("Number of columns that cover each row: {}".format(nr_of_columns_that_cover_a_row))

    # Columns that are eligible for use: this is turned off when we choose a set that contains an element
    candidate_columns = [i+1 for i in range(set_covering_problem_instance.scp_instance_column_costs.shape[0])]
    print("Candidate Columns {}".format(candidate_columns))

    # Columns already used
    rows_already_covered = list()

    # Sets that are available for further use
    # candidate_rows = [i for i, _ in enumerate(set_covering_problem_instance.scp_instance_all_subsets)]
    # print("Candidate rows {}".format(candidate_sets))

    # The current solution cost: this is an unweighted setting so each set has the value of 1. We want to minimized the number of sets here.
    current_cost = 0
    # print(current_cost)

    # The list of sets to be added (we add the index of each set that is part of the SCPInstance object)
    current_solution = list()
    # print(current_solution)

    # Begin CH1
    print("Starting CH1 Algorithm...")
    
    # Iteration counter for debugging purposes
    iteration = 1

    # The program must continue until we cover all the rows
    while nr_of_rows_covered < number_of_rows_to_be_covered:
        # Print current iteration
        print("Iteration {}".format(iteration))

        # Step 1: Pick randomly a still uncovered row
        uncovered_row = np.random.choice(rows_to_be_covered)
        print("Uncovered row: {}".format(uncovered_row))

        # Step 2: Check which columns contain this row
        # Go through candidate columns
        # Create a temp list to update (in each iteration it is reset)
        # columns_that_contain_un_element = list()
        columns_that_contain_un_row = set_covering_problem_instance.scp_instance_all_subsets[uncovered_row].copy()
        # Check it is part of candidate columns
        columns_that_contain_un_row = [i for i in columns_that_contain_un_row if i in candidate_columns]
        
        # Check the indices in the candidate sets list
        # for col_idx in set_covering_problem_instance.scp_instance_all_subsets[uncovered_row]:
            # Choose the set that we will evaluate
            # current_columns = set_covering_problem_instance.scp_instance_all_subsets[set_idx].copy()
            # Check if this set contains the index of the uncovered element
            # The dataset contains indices that start in 1, we should "Pythonize" this
            # current_set = [i-1 for i in current_set]
            # Now, we check if the current set has an index that point to the element
            # for e_idx in current_set:
                # TODO: Check if this indice matters in this iteration by evaluating candidate columns
                # if e_idx in candidate_columns:
                # if e_idx == uncovered_element:
                    # if set_idx not in sets_that_contain_un_element:
                        # sets_that_contain_un_element.append(set_idx)
                    # Now check if any element index in the current set corresponds to the uncovered element
                    # if set_covering_problem_instance.scp_instance_attribute_map[e_idx] == uncovered_element:
                        # We append the set_idx to the list of sets that contain the element
                        # if set_idx not in sets_that_contain_un_element:
                            # sets_that_contain_un_element.append(set_idx)
        
        # Compute Cost Effectiveness of each column (evaluate the number of rows that each column contains)
        # First, we need to compute the rows that are covered by the solution
        if len(current_solution) > 0:
            covered_rows_list_per_column = [list() for i, _ in enumerate(columns_that_contain_un_row)]
            # Iterate through the rows of the column in the columns list
            for c_i, column in enumerate(columns_that_contain_un_row):
                for r_i, row in enumerate(rows_to_be_covered):
                    if column in row:
                        covered_rows_list_per_column[c_i].append(r_i)
            
            # Compare with the rows covered by our solution
            difference_between_column_sets = [0 for i, _ in enumerate(columns_that_contain_un_row)]
            for c_i, covered_rows in enumerate(covered_rows_list_per_column):
                for row in covered_rows:
                    if row not in rows_already_covered:
                        difference_between_column_sets[c_i] += 1
            
            # Cost effectivenes per column
            costs_effectiveness = [0 for i, _ in enumerate(difference_between_column_sets)]
            for c_i, column in enumerate(columns_that_contain_un_row):
                column_cost = set_covering_problem_instance.scp_instance_column_costs[column-1]
                effective_cost = column_cost / difference_between_column_sets[c_i]
                costs_effectiveness[c_i] = effective_cost
            
            selected_columns = [columns_that_contain_un_row[np.argmin(costs_effectiveness)]]
            
        
        else:
            # In the first iteration we should evaluate the number of rows covered by each possible column
            number_of_rows_per_column = [0 for i, _ in enumerate(columns_that_contain_un_row)]
            for c_i, column in enumerate(columns_that_contain_un_row):
                for r_i, row in enumerate(rows_to_be_covered):
                    if column in row:
                        number_of_rows_per_column[c_i] += 1
            
            # Cost effectiveness
            costs_effectiveness = [0 for i, _ in enumerate(number_of_rows_per_column)]
            for c_i, column in enumerate(columns_that_contain_un_row):
                column_cost = set_covering_problem_instance.scp_instance_column_costs[column-1]
                effective_cost = column_cost / number_of_rows_per_column[c_i]
                costs_effectiveness[c_i] = effective_cost
            
            selected_columns = [columns_that_contain_un_row[np.argmin(costs_effectiveness)]]


        # With the possible list of sets that contain an element we are now able to randomly choose a set
        selected_columns = np.random.choice(selected_columns)

        # Append this set to the solution
        current_solution.append(selected_columns)

        # We have to check the costs matrix
        current_cost = 0
        for solution_column in current_solution:
            current_cost += set_covering_problem_instance.scp_instance_column_costs[solution_column-1]
        
        # Update candidate columns
        candidate_columns.remove(selected_columns)

        # Update rows to be covered and rows already covered
        # Rows to be covered and rows already covered
        for _, solution_column in enumerate(current_solution):
            for r_i, row in rows_to_be_covered:
                if solution_column in set_covering_problem_instance.scp_instance_all_subsets[row]:
                    rows_to_be_covered.remove(row)
                    rows_already_covered.append(row)
        
        
        # Update number of rows to be covered
        nr_of_rows_covered = len(rows_already_covered)


        
        # Some debugging prints
        print("Number of Rows Already Covered: {}".format(nr_of_rows_covered))
        print("Current Cost: {}".format(current_cost))
        print("Current Solution {}".format(current_solution))
        print("Uncovered Rows: {}".format(rows_to_be_covered))
        print("Candidate Columns: {}".format(candidate_columns))
        iteration += 1

    # Select if we apply redundancy elimination
    if post_processing:
        pass
    
    final_cost = current_cost
    final_solution = current_solution.copy()



    return final_solution, final_cost




# Function CH2: Constructive Heuristics Nr. 2
def ch2(set_covering_problem_instance, post_processing=False, random_seed=42):
    """
    CH2 - TBD
    """
    # Set random seed
    np.random.seed(seed=random_seed)
    
    pass


# Function CH2: Constructive Heuristics Nr. 3
def ch3(set_covering_problem_instance, post_processing=False, random_seed=42):
    """
    CH3 - TBD
    """
    # Set random seed
    np.random.seed(seed=random_seed)
    
    pass