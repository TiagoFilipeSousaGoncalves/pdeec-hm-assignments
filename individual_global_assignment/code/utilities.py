# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os
# Import for the LP Method
from scipy.optimize import linprog


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
def ch1(set_covering_problem_instance, post_processing=True, random_seed=42):
    """
    CH1 - Picking in each constructive step first a still un-covered element and then, second, the set with lower cost effectiveness that covers this element.
    """
    
    # Set random seed
    np.random.seed(seed=random_seed)

    # Build Row X Column Matrix
    problem_matrix = np.zeros((set_covering_problem_instance.scp_number_of_rows, set_covering_problem_instance.scp_number_of_columns), dtype=int)
    # Fill Problem Matrix
    for row_idx, row in enumerate(set_covering_problem_instance.scp_instance_all_rows):
        for column in row:
            problem_matrix[row_idx, column-1] = 1
    # De bugging print
    # print(problem_matrix)
    # print(np.where(problem_matrix==1))
    # print(problem_matrix.shape)


    # Set some control variables
    # Rows covered
    rows_covered = list()
    
    # Number of Columns that Cover a Row
    number_of_columns_that_cover_a_row = list()
    # Let's compute the initial number of columns that cover each row
    for row in range(problem_matrix.shape[0]):
        number_of_columns_that_cover_a_row.append(np.sum(problem_matrix[row, :]))
    # Debugging point
    # print(number_of_columns_that_cover_a_row)
    # print(len(number_of_columns_that_cover_a_row))
    
    # Columns to be considered candidates to be added
    candidate_columns_to_be_added = [i for i in range(set_covering_problem_instance.scp_number_of_columns)]
    # Debugging points
    # print(candidate_columns_to_be_added)
    # print(len(candidate_columns_to_be_added))

    # Solution
    solution = list()

    # Begin CH1 Algorithm
    while len(rows_covered) < problem_matrix.shape[0]:
        # Step 1: Pick a random row
        possible_rows = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]
        # Debugging point
        # print(len(possible_rows))
        chosen_row = np.random.choice(possible_rows)

        # Step 2: Discover all the columns that contain this row
        possible_columns = list()
        for col_idx, column in enumerate(problem_matrix[chosen_row]):
            # Column must be 1 and it has to be in the list of candidate columns 
            if column == 1 and (col_idx in candidate_columns_to_be_added):
                possible_columns.append(col_idx)
        # Debugging print
        # print(len(possible_columns)==number_of_columns_that_cover_a_row[chosen_row])
        # break

        # Step 3: Compute Cost Effectiveness
        # We create a list to append the cost effectiveness per column candidate
        possible_columns_costs_effectiveness = list()
        # Go trough possible columns
        for poss_col in possible_columns:
            # We must see the differences between our current solution and a possible chosen column to evaluate the gain
            if len(solution) == 0:
                # Check per column the number of rows it adds
                rows_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, poss_col]):
                    if row == 1:
                        rows_this_column_adds.append(row_idx)
                # In this case our solution contains zero columns and therefore zero rows, so we will evaluate directly the col that has much more rows for the less cost
                difference_of_this_col_to_solution = len(rows_this_column_adds)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[poss_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
            else:
                # Check per column th number of rows it adds
                rows_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, poss_col]):
                    if row == 1:
                        rows_this_column_adds.append(row_idx)
                # Here, our len(solution) > 0 so we have to analyse the difference between a possible column and the current solution
                solution_rows = list()
                for s_col in solution:
                    for row_idx, row in enumerate(problem_matrix[:, s_col]):
                        if row == 1 and (row_idx not in solution_rows):
                            solution_rows.append(row_idx)
                
                # Check differences between that rows that we already cover and the possibilities
                # Evalute difference between possible column and our solution
                difference_between_column_and_curr_solution = [i for i in rows_this_column_adds if i not in solution_rows]
                # This will enter the formula of cost effectiveness
                difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[poss_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
        
        # We now have the cost effectiveness of each column, so we should select the one with lower costs
        selected_column = possible_columns[np.argmin(possible_columns_costs_effectiveness)]
        
        # We can append this row to the solution
        solution.append(selected_column)
        
        # Compute current cost
        current_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in solution])

        # Update candidate columns to be added
        candidate_columns_to_be_added.remove(selected_column)

        # Update rows covered
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1 and (row_idx not in rows_covered):
                rows_covered.append(row_idx)
        
        # Update number of columns that cover a row
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1:
                number_of_columns_that_cover_a_row[row_idx] -= 1
        
        # Some status prints
        # Rows covered
        # print("Rows Covered: {}".format(rows_covered))
        # print("Number of Rows Covered: {}".format(len(rows_covered)))
        
        # Number of Candidate Columns
        # print("Number of Candidate Columns: {}".format(len(candidate_columns_to_be_added)))

        # Current cost
        # print("Current cost: {}".format(current_cost))

        # Current solution
        # print("Current solution: {}".format(solution))

    # Final Statements
    final_solution = solution
    final_cost = current_cost
    # print("Final Solution is: {}".format(final_solution))
    print("Final Cost is: {}".format(final_cost))
        


    
    # Select if we apply redundancy elimination
    if post_processing:
        # Approach 1
        # Initialise variables again
        # Rows covered
        rows_covered = list()

        # Rows to be covered
        rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # We check each row in our candidate columns
        while len(rows_covered) < problem_matrix.shape[0]:
            # Check col with higher number of rows covered
            nr_rows_convered_by_columns = list()
            for possible_col in candidate_columns:
                nr_rows_convered_by_columns.append(np.sum(problem_matrix[:, possible_col]))
            
            # Check the col with higher number of rows
            column_with_more_rows = candidate_columns[np.argmax(nr_rows_convered_by_columns)]

            # Check which rows are covered by this column
            rows_covered_by_col_with_more_rows = list()
            for row_idx, row in enumerate(problem_matrix[:, column_with_more_rows]):
                if row == 1:
                    rows_covered_by_col_with_more_rows.append(row_idx)
            
            # Check columns that cover rows that may be contained in the col with more rows
            columns_contained_in_col_with_more_rows = list()
            for other_cold_idx, other_col in enumerate(candidate_columns):
                if other_col != column_with_more_rows:
                    rows_by_other_col = list()
                    for row_idx, row in enumerate(problem_matrix[:, other_col]):
                        if row == 1:
                            rows_by_other_col.append(row_idx)
                    
                    # Check if this column is contained in the column with more rows covered
                    for row in rows_covered_by_col_with_more_rows:
                        if row in rows_by_other_col:
                            rows_by_other_col.remove(row)
                    
                    # If the len(rows_by_other_col) == 0, then it is contained in this column
                    if len(rows_by_other_col) == 0:
                        columns_contained_in_col_with_more_rows.append(other_col)
            
            # Remove redundant columns
            if len(columns_contained_in_col_with_more_rows) > 0:
                # We remove them from candidate columns list
                for col in columns_contained_in_col_with_more_rows:
                    candidate_columns.remove(col)
                
            # We add the column with more rows to the solution
            processed_solution.append(column_with_more_rows)

            # We remove this columns from candidate columns
            candidate_columns.remove(column_with_more_rows)

            # We update rows covered and rows to be covered
            # Rows to be covered
            for row in rows_covered_by_col_with_more_rows:
                if row in rows_to_be_covered:
                    rows_to_be_covered.remove(row)
            # Rows covered
            for row in rows_covered_by_col_with_more_rows:
                if row not in rows_covered:
                    rows_covered.append(row)
            
            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

        
        # Final Solution
        final_processed_solution = processed_solution
        final_processed_cost = processed_cost
        # print("Final Solution after processing: {}".format(final_processed_solution))
        print("Final Cost after processing: {}".format(final_processed_cost))

        return final_solution, final_cost, final_processed_solution, final_processed_cost




        """
        # Approach 2
        # Its almost the same as the algorithm above, but now we have a closed set to search
        # Let's initialise some variables again
        # Rows covered
        rows_covered = list()
        
        # Rows to be convered
        # rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed Solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # Now we iterate the same way as the algorithm above
        while len(rows_covered) < problem_matrix.shape[0]:
            # Now, we don't choose a random row, we start by the column with less effective cost
            possible_columns_costs_effectiveness = list()
            for possible_col in candidate_columns:
                # First iteration
                if len(processed_solution) == 0:
                    rows_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                        if row == 1:
                            rows_this_column_adds.append(row_idx)
                    
                    # In this case, the solution contains zero elements, so we only need to check the difference to an empty set
                    difference_of_this_col_to_solution = len(rows_this_column_adds)
                    # To avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                        possible_columns_costs_effectiveness.append(effective_cost)
                    else:
                        possible_columns_costs_effectiveness.append(np.inf)
                
                # Here, the processed solution already has elements
                else:
                    rows_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                        if row == 1:
                            rows_this_column_adds.append(row_idx)
                    
                    # In this case, we already have elements in the solution, so we need to compare them
                    solution_rows = list()
                    for s_col in processed_solution:
                        for row_idx, row in enumerate(problem_matrix[:, s_col]):
                            if row == 1 and (row_idx not in solution_rows):
                                solution_rows.append(row_idx)
                    
                    # Check differences between that rows that we already cover and the possibilities
                    # Evalute difference between possible column and our solution
                    difference_between_column_and_curr_solution = [i for i in rows_this_column_adds if i not in solution_rows]
                    # This will enter the formula of cost effectiveness
                    difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                    # To avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                        possible_columns_costs_effectiveness.append(effective_cost)
                    else:
                        possible_columns_costs_effectiveness.append(np.inf)
                
            # Now, we select column with less cost effectiveness
            selected_column = candidate_columns[np.argmin(possible_columns_costs_effectiveness)]

            # Append the column to the processed solution
            processed_solution.append(selected_column)

            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

            # Update candidate columns to be added
            candidate_columns.remove(selected_column)
            # candidate_columns_to_be_added.remove(selected_column)

            # Update rows covered
            for row_idx, row in enumerate(problem_matrix[:, selected_column]):
                if row == 1 and (row_idx not in rows_covered):
                    rows_covered.append(row_idx)
        
        # Get final results
        final_solution = processed_solution
        final_cost = processed_cost
        """


    return final_solution, final_cost



# Function CH2: Constructive Heuristics Nr. 2
def ch2(set_covering_problem_instance, post_processing=True, random_seed=42):
    """
    CH2 - Evaluate all the sets, pick the one with smaller cost effectiveness and add to the solution.
    """
    # Set random seed
    np.random.seed(seed=random_seed)

    # Initialise variables
    # Build Row X Column Matrix
    problem_matrix = np.zeros((set_covering_problem_instance.scp_number_of_rows, set_covering_problem_instance.scp_number_of_columns), dtype=int)
    # Fill Problem Matrix
    for row_idx, row in enumerate(set_covering_problem_instance.scp_instance_all_rows):
        for column in row:
            problem_matrix[row_idx, column-1] = 1
    # De bugging print
    # print(problem_matrix)
    # print(np.where(problem_matrix==1))
    # print(problem_matrix.shape)


    # Set some control variables
    # Rows covered
    rows_covered = list()
    
    # Number of Columns that Cover a Row
    number_of_columns_that_cover_a_row = list()
    # Let's compute the initial number of columns that cover each row
    for row in range(problem_matrix.shape[0]):
        number_of_columns_that_cover_a_row.append(np.sum(problem_matrix[row, :]))
    # Debugging point
    # print(number_of_columns_that_cover_a_row)
    # print(len(number_of_columns_that_cover_a_row))
    
    # Columns to be considered candidates to be added
    candidate_columns_to_be_added = [i for i in range(set_covering_problem_instance.scp_number_of_columns)]
    # Debugging points
    # print(candidate_columns_to_be_added)
    # print(len(candidate_columns_to_be_added))

    # Solution
    solution = list()

    # Begin CH2 Algorithm
    while len(rows_covered) < problem_matrix.shape[0]:
        # Step 1: Analyse all the candidate columns to be added and check which one has the lower effective cost
        possible_columns_costs_effectiveness = list()
        for poss_col_idx, possible_col in enumerate(candidate_columns_to_be_added):
            # First iteration
            if len(solution) > 0:
                rows_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_this_column_adds.append(row_idx)
                
                # We evaluate the difference of this col to the solution
                difference_of_this_col_to_solution = len(rows_this_column_adds)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
            
            # In the following iterations, our solution already has columns
            else:
                rows_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_this_column_adds.append(row_idx)
                
                # Now we check the solution rows
                solution_rows = list()
                for s_col in solution:
                    for row_idx, row in enumerate(problem_matrix[:, s_col]):
                        if row == 1 and (row_idx not in solution_rows):
                            solution_rows.append(row_idx)
                
                # Now we check differences between solution and this col
                difference_between_column_and_curr_solution = [i for i in rows_this_column_adds if i not in solution_rows]
                # Count the different rows
                difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                # Avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
        
        # Now we update variables
        # We now have the cost effectiveness of each column, so we should select the one with lower costs
        selected_column = candidate_columns_to_be_added[np.argmin(possible_columns_costs_effectiveness)]
        
        # We can append this row to the solution
        solution.append(selected_column)
        
        # Compute current cost
        current_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in solution])

        # Update candidate columns to be added
        candidate_columns_to_be_added.remove(selected_column)

        # Update rows covered
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1 and (row_idx not in rows_covered):
                rows_covered.append(row_idx)
        
        # Update number of columns that cover a row
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1:
                number_of_columns_that_cover_a_row[row_idx] -= 1
        
        # Some status prints
        # Rows covered
        # print("Rows Covered: {}".format(rows_covered))
        # print("Number of Rows Covered: {}".format(len(rows_covered)))
        
        # Number of Candidate Columns
        # print("Number of Candidate Columns: {}".format(len(candidate_columns_to_be_added)))

        # Current cost
        # print("Current cost: {}".format(current_cost))

        # Current solution
        # print("Current solution: {}".format(solution))

    # Final Statements
    final_solution = solution
    final_cost = current_cost
    # print("Final Solution is: {}".format(final_solution))
    print("Final Cost is: {}".format(final_cost))







    
    # Select if we apply redundancy elimination
    if post_processing:
        # Approach 1
        # Initialise variables again
        # Rows covered
        rows_covered = list()

        # Rows to be covered
        rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # We check each row in our candidate columns
        while len(rows_covered) < problem_matrix.shape[0]:
            # Check col with higher number of rows covered
            nr_rows_convered_by_columns = list()
            for possible_col in candidate_columns:
                nr_rows_convered_by_columns.append(np.sum(problem_matrix[:, possible_col]))
            
            # Check the col with higher number of rows
            column_with_more_rows = candidate_columns[np.argmax(nr_rows_convered_by_columns)]

            # Check which rows are covered by this column
            rows_covered_by_col_with_more_rows = list()
            for row_idx, row in enumerate(problem_matrix[:, column_with_more_rows]):
                if row == 1:
                    rows_covered_by_col_with_more_rows.append(row_idx)
            
            # Check columns that cover rows that may be contained in the col with more rows
            columns_contained_in_col_with_more_rows = list()
            for other_cold_idx, other_col in enumerate(candidate_columns):
                if other_col != column_with_more_rows:
                    rows_by_other_col = list()
                    for row_idx, row in enumerate(problem_matrix[:, other_col]):
                        if row == 1:
                            rows_by_other_col.append(row_idx)
                    
                    # Check if this column is contained in the column with more rows covered
                    for row in rows_covered_by_col_with_more_rows:
                        if row in rows_by_other_col:
                            rows_by_other_col.remove(row)
                    
                    # If the len(rows_by_other_col) == 0, then it is contained in this column
                    if len(rows_by_other_col) == 0:
                        columns_contained_in_col_with_more_rows.append(other_col)
            
            # Remove redundant columns
            if len(columns_contained_in_col_with_more_rows) > 0:
                # We remove them from candidate columns list
                for col in columns_contained_in_col_with_more_rows:
                    candidate_columns.remove(col)
                
            # We add the column with more rows to the solution
            processed_solution.append(column_with_more_rows)

            # We remove this columns from candidate columns
            candidate_columns.remove(column_with_more_rows)

            # We update rows covered and rows to be covered
            # Rows to be covered
            for row in rows_covered_by_col_with_more_rows:
                if row in rows_to_be_covered:
                    rows_to_be_covered.remove(row)
            # Rows covered
            for row in rows_covered_by_col_with_more_rows:
                if row not in rows_covered:
                    rows_covered.append(row)
            
            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

        
        # Final Solution
        final_processed_solution = processed_solution
        final_processed_cost = processed_cost
        # print("Final Solution after processing: {}".format(final_processed_solution))
        print("Final Cost after processing: {}".format(final_processed_cost))

        return final_solution, final_cost, final_processed_solution, final_processed_cost




        """
        # Approach 2
        # Its almost the same as the algorithm above, but now we have a closed set to search
        # Let's initialise some variables again
        # Rows covered
        rows_covered = list()
        
        # Rows to be convered
        # rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed Solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # Now we iterate the same way as the algorithm above
        while len(rows_covered) < problem_matrix.shape[0]:
            # Now, we don't choose a random row, we start by the column with less effective cost
            possible_columns_costs_effectiveness = list()
            for possible_col in candidate_columns:
                # First iteration
                if len(processed_solution) == 0:
                    rows_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                        if row == 1:
                            rows_this_column_adds.append(row_idx)
                    
                    # In this case, the solution contains zero elements, so we only need to check the difference to an empty set
                    difference_of_this_col_to_solution = len(rows_this_column_adds)
                    # To avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                        possible_columns_costs_effectiveness.append(effective_cost)
                    else:
                        possible_columns_costs_effectiveness.append(np.inf)
                
                # Here, the processed solution already has elements
                else:
                    rows_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                        if row == 1:
                            rows_this_column_adds.append(row_idx)
                    
                    # In this case, we already have elements in the solution, so we need to compare them
                    solution_rows = list()
                    for s_col in processed_solution:
                        for row_idx, row in enumerate(problem_matrix[:, s_col]):
                            if row == 1 and (row_idx not in solution_rows):
                                solution_rows.append(row_idx)
                    
                    # Check differences between that rows that we already cover and the possibilities
                    # Evalute difference between possible column and our solution
                    difference_between_column_and_curr_solution = [i for i in rows_this_column_adds if i not in solution_rows]
                    # This will enter the formula of cost effectiveness
                    difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                    # To avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                        possible_columns_costs_effectiveness.append(effective_cost)
                    else:
                        possible_columns_costs_effectiveness.append(np.inf)
                
            # Now, we select column with less cost effectiveness
            selected_column = candidate_columns[np.argmin(possible_columns_costs_effectiveness)]

            # Append the column to the processed solution
            processed_solution.append(selected_column)

            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

            # Update candidate columns to be added
            candidate_columns.remove(selected_column)
            # candidate_columns_to_be_added.remove(selected_column)

            # Update rows covered
            for row_idx, row in enumerate(problem_matrix[:, selected_column]):
                if row == 1 and (row_idx not in rows_covered):
                    rows_covered.append(row_idx)
        
        # Get final results
        final_solution = processed_solution
        final_cost = processed_cost
        """
    

    return final_solution, final_cost


# Function CH3: Constructive Heuristics Nr. 3
def ch3(set_covering_problem_instance, post_processing=True, random_seed=42):
    """
    CH3 - Pick the less covered row, choose the column that covers this row which has a lower value of cost effectiveness and add to the final solution.
    """
    # Set random seed
    np.random.seed(seed=random_seed)

    # Initialise variables
    # Build Row X Column Matrix
    problem_matrix = np.zeros((set_covering_problem_instance.scp_number_of_rows, set_covering_problem_instance.scp_number_of_columns), dtype=int)
    # Fill Problem Matrix
    for row_idx, row in enumerate(set_covering_problem_instance.scp_instance_all_rows):
        for column in row:
            problem_matrix[row_idx, column-1] = 1
    # De bugging print
    # print(problem_matrix)
    # print(np.where(problem_matrix==1))
    # print(problem_matrix.shape)


    # Set some control variables
    # Rows covered
    rows_covered = list()
    
    # Number of Columns that Cover a Row
    number_of_columns_that_cover_a_row = list()
    # Let's compute the initial number of columns that cover each row
    for row in range(problem_matrix.shape[0]):
        number_of_columns_that_cover_a_row.append(np.sum(problem_matrix[row, :]))
    # Debugging point
    # print(number_of_columns_that_cover_a_row)
    # print(len(number_of_columns_that_cover_a_row))
    
    # Columns to be considered candidates to be added
    candidate_columns_to_be_added = [i for i in range(set_covering_problem_instance.scp_number_of_columns)]
    # Debugging points
    # print(candidate_columns_to_be_added)
    # print(len(candidate_columns_to_be_added))

    # Solution
    solution = list()

    # Begin CH3 Algorithm
    while len(rows_covered) < problem_matrix.shape[0]:
        # Step 1: We check "less" covered rows first
        number_of_cols_of_less_cov_row = number_of_columns_that_cover_a_row[np.argmin(number_of_columns_that_cover_a_row)]
        # We need to be sure that there aren't other less covered rows as well
        less_covered_rows = list()
        for row_idx, cols_that_cover_row in enumerate(number_of_columns_that_cover_a_row):
            if cols_that_cover_row == number_of_cols_of_less_cov_row:
                less_covered_rows.append(row_idx)
        
        # Now that we have the list of less covered rows, we may check the one that contains the cols with lower value of cost-effectiveness
        lower_cost_effectiveness_per_row = list()
        for row in less_covered_rows:
            colums_that_cover_this_row = list()
            for col_idx, column in enumerate(problem_matrix[row, :]):
                if column == 1 and (col_idx in candidate_columns_to_be_added):
                    colums_that_cover_this_row.append(col_idx)
            
            # Now let's check the cost-effectiveness of the columns that cover this row
            cost_eff_of_cols_that_cov_this_row = list()
            if len(solution) == 0:
                for col in colums_that_cover_this_row:
                    rows_that_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, col]):
                        if row == 1:
                            rows_that_this_column_adds.append(row_idx)
                    # Compute the difference for the solution
                    difference_of_this_col_to_solution = len(rows_that_this_column_adds)
                    # Avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[col] / difference_of_this_col_to_solution
                        cost_eff_of_cols_that_cov_this_row.append(effective_cost)
                    else:
                        cost_eff_of_cols_that_cov_this_row.append(np.inf)
            else:
                for col in colums_that_cover_this_row:
                    rows_that_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, col]):
                        if row == 1:
                            rows_that_this_column_adds.append(row_idx)

                    # Here we have to check the rows covered by our solution
                    # Now we check the solution rows
                    solution_rows = list()
                    for s_col in solution:
                        for row_idx, row in enumerate(problem_matrix[:, s_col]):
                            if row == 1 and (row_idx not in solution_rows):
                                solution_rows.append(row_idx)
                    
                    # Now we check differences between solution and this col
                    difference_between_column_and_curr_solution = [i for i in rows_that_this_column_adds if i not in solution_rows]
                    # Count the different rows
                    difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                    # Avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[col] / difference_of_this_col_to_solution
                        cost_eff_of_cols_that_cov_this_row.append(effective_cost)
                    else:
                        cost_eff_of_cols_that_cov_this_row.append(np.inf)
                    
            
            # We append the minimum cost effectiveness per row
            lower_cost_effectiveness_per_row.append(np.min(cost_eff_of_cols_that_cov_this_row))
        
        # We now select the row which we want to cover
        selected_row = less_covered_rows[np.argmin(lower_cost_effectiveness_per_row)]
        
        # We now check the columns that cover this row
        columns_that_cover_selected_row = list()
        for col_idx, column in enumerate(problem_matrix[selected_row, :]):
            if column == 1 and (col_idx in candidate_columns_to_be_added):
                columns_that_cover_selected_row.append(col_idx)
        
        # We check the cost effectivenesses
        possible_columns_costs_effectiveness = list()
        for poss_col_idx, possible_col in enumerate(columns_that_cover_selected_row):
            if len(solution) == 0:
                rows_that_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_that_this_column_adds.append(row_idx)
                
                # We evaluate the difference of this col to the solution
                difference_of_this_col_to_solution = len(rows_that_this_column_adds)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
            
            # Here we have to compare against our solution
            else:
                rows_that_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_that_this_column_adds.append(row_idx)
                
                # Here we have to check the rows covered by our solution
                # Now we check the solution rows
                solution_rows = list()
                for s_col in solution:
                    for row_idx, row in enumerate(problem_matrix[:, s_col]):
                        if row == 1 and (row_idx not in solution_rows):
                            solution_rows.append(row_idx)
                
                # Now we check differences between solution and this col
                difference_between_column_and_curr_solution = [i for i in rows_that_this_column_adds if i not in solution_rows]
                # Count the different rows
                difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
        
        # Now we update our problem
        # We now have the cost effectiveness of each column, so we should select the one with lower costs
        selected_column = candidate_columns_to_be_added[np.argmin(possible_columns_costs_effectiveness)]
        
        # We can append this row to the solution
        solution.append(selected_column)
        
        # Compute current cost
        current_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in solution])

        # Update candidate columns to be added
        candidate_columns_to_be_added.remove(selected_column)

        # Update rows covered
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1 and (row_idx not in rows_covered):
                rows_covered.append(row_idx)
        
        # Update number of columns that cover a row
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1:
                number_of_columns_that_cover_a_row[row_idx] -= 1
        
        # Some status prints
        # Rows covered
        # print("Rows Covered: {}".format(rows_covered))
        # print("Number of Rows Covered: {}".format(len(rows_covered)))
        
        # Number of Candidate Columns
        # print("Number of Candidate Columns: {}".format(len(candidate_columns_to_be_added)))

        # Current cost
        # print("Current cost: {}".format(current_cost))

        # Current solution
        # print("Current solution: {}".format(solution))

    # Final Statements
    final_solution = solution
    final_cost = current_cost
    # print("Final Solution is: {}".format(final_solution))
    print("Final Cost is: {}".format(final_cost))


    # Select if we apply redundancy elimination
    if post_processing:
        # Approach 1
        # Initialise variables again
        # Rows covered
        rows_covered = list()

        # Rows to be covered
        rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # We check each row in our candidate columns
        while len(rows_covered) < problem_matrix.shape[0]:
            # Check col with higher number of rows covered
            nr_rows_convered_by_columns = list()
            for possible_col in candidate_columns:
                nr_rows_convered_by_columns.append(np.sum(problem_matrix[:, possible_col]))
            
            # Check the col with higher number of rows
            column_with_more_rows = candidate_columns[np.argmax(nr_rows_convered_by_columns)]

            # Check which rows are covered by this column
            rows_covered_by_col_with_more_rows = list()
            for row_idx, row in enumerate(problem_matrix[:, column_with_more_rows]):
                if row == 1:
                    rows_covered_by_col_with_more_rows.append(row_idx)
            
            # Check columns that cover rows that may be contained in the col with more rows
            columns_contained_in_col_with_more_rows = list()
            for other_cold_idx, other_col in enumerate(candidate_columns):
                if other_col != column_with_more_rows:
                    rows_by_other_col = list()
                    for row_idx, row in enumerate(problem_matrix[:, other_col]):
                        if row == 1:
                            rows_by_other_col.append(row_idx)
                    
                    # Check if this column is contained in the column with more rows covered
                    for row in rows_covered_by_col_with_more_rows:
                        if row in rows_by_other_col:
                            rows_by_other_col.remove(row)
                    
                    # If the len(rows_by_other_col) == 0, then it is contained in this column
                    if len(rows_by_other_col) == 0:
                        columns_contained_in_col_with_more_rows.append(other_col)
            
            # Remove redundant columns
            if len(columns_contained_in_col_with_more_rows) > 0:
                # We remove them from candidate columns list
                for col in columns_contained_in_col_with_more_rows:
                    candidate_columns.remove(col)
                
            # We add the column with more rows to the solution
            processed_solution.append(column_with_more_rows)

            # We remove this columns from candidate columns
            candidate_columns.remove(column_with_more_rows)

            # We update rows covered and rows to be covered
            # Rows to be covered
            for row in rows_covered_by_col_with_more_rows:
                if row in rows_to_be_covered:
                    rows_to_be_covered.remove(row)
            # Rows covered
            for row in rows_covered_by_col_with_more_rows:
                if row not in rows_covered:
                    rows_covered.append(row)
            
            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

        
        # Final Solution
        final_processed_solution = processed_solution
        final_processed_cost = processed_cost
        # print("Final Solution after processing: {}".format(final_processed_solution))
        print("Final Cost after processing: {}".format(final_processed_cost))

        return final_solution, final_cost, final_processed_solution, final_processed_cost

    

    return final_solution, final_cost

# Function CH4: Constructive Heuristics Nr. 4
def ch4(set_covering_problem_instance, post_processing=True, random_seed=42):
    """
    CH4 - Pick the more covered row, choose the column that covers this row which has a lower value of cost effectiveness and add to the final solution.
    """
    # Set random seed
    np.random.seed(seed=random_seed)

    # Initialise variables
    # Build Row X Column Matrix
    problem_matrix = np.zeros((set_covering_problem_instance.scp_number_of_rows, set_covering_problem_instance.scp_number_of_columns), dtype=int)
    # Fill Problem Matrix
    for row_idx, row in enumerate(set_covering_problem_instance.scp_instance_all_rows):
        for column in row:
            problem_matrix[row_idx, column-1] = 1
    # De bugging print
    # print(problem_matrix)
    # print(np.where(problem_matrix==1))
    # print(problem_matrix.shape)


    # Set some control variables
    # Rows covered
    rows_covered = list()
    
    # Number of Columns that Cover a Row
    number_of_columns_that_cover_a_row = list()
    # Let's compute the initial number of columns that cover each row
    for row in range(problem_matrix.shape[0]):
        number_of_columns_that_cover_a_row.append(np.sum(problem_matrix[row, :]))
    # Debugging point
    # print(number_of_columns_that_cover_a_row)
    # print(len(number_of_columns_that_cover_a_row))
    
    # Columns to be considered candidates to be added
    candidate_columns_to_be_added = [i for i in range(set_covering_problem_instance.scp_number_of_columns)]
    # Debugging points
    # print(candidate_columns_to_be_added)
    # print(len(candidate_columns_to_be_added))

    # Solution
    solution = list()

    # Begin CH4 Algorithm
    while len(rows_covered) < problem_matrix.shape[0]:
        # Step 1: We check "less" covered rows first
        number_of_cols_of_less_cov_row = number_of_columns_that_cover_a_row[np.argmax(number_of_columns_that_cover_a_row)]
        # We need to be sure that there aren't other less covered rows as well
        less_covered_rows = list()
        for row_idx, cols_that_cover_row in enumerate(number_of_columns_that_cover_a_row):
            if cols_that_cover_row == number_of_cols_of_less_cov_row:
                less_covered_rows.append(row_idx)
        
        # Now that we have the list of less covered rows, we may check the one that contains the cols with lower value of cost-effectiveness
        lower_cost_effectiveness_per_row = list()
        for row in less_covered_rows:
            colums_that_cover_this_row = list()
            for col_idx, column in enumerate(problem_matrix[row, :]):
                if column == 1 and (col_idx in candidate_columns_to_be_added):
                    colums_that_cover_this_row.append(col_idx)
            
            # Now let's check the cost-effectiveness of the columns that cover this row
            cost_eff_of_cols_that_cov_this_row = list()
            if len(solution) == 0:
                for col in colums_that_cover_this_row:
                    rows_that_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, col]):
                        if row == 1:
                            rows_that_this_column_adds.append(row_idx)
                    # Compute the difference for the solution
                    difference_of_this_col_to_solution = len(rows_that_this_column_adds)
                    # Avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[col] / difference_of_this_col_to_solution
                        cost_eff_of_cols_that_cov_this_row.append(effective_cost)
                    else:
                        cost_eff_of_cols_that_cov_this_row.append(np.inf)
            else:
                for col in colums_that_cover_this_row:
                    rows_that_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, col]):
                        if row == 1:
                            rows_that_this_column_adds.append(row_idx)

                    # Here we have to check the rows covered by our solution
                    # Now we check the solution rows
                    solution_rows = list()
                    for s_col in solution:
                        for row_idx, row in enumerate(problem_matrix[:, s_col]):
                            if row == 1 and (row_idx not in solution_rows):
                                solution_rows.append(row_idx)
                    
                    # Now we check differences between solution and this col
                    difference_between_column_and_curr_solution = [i for i in rows_that_this_column_adds if i not in solution_rows]
                    # Count the different rows
                    difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                    # Avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[col] / difference_of_this_col_to_solution
                        cost_eff_of_cols_that_cov_this_row.append(effective_cost)
                    else:
                        cost_eff_of_cols_that_cov_this_row.append(np.inf)
                    
            
            # We append the minimum cost effectiveness per row
            lower_cost_effectiveness_per_row.append(np.min(cost_eff_of_cols_that_cov_this_row))
        
        # We now select the row which we want to cover
        selected_row = less_covered_rows[np.argmin(lower_cost_effectiveness_per_row)]
        
        # We now check the columns that cover this row
        columns_that_cover_selected_row = list()
        for col_idx, column in enumerate(problem_matrix[selected_row, :]):
            if column == 1 and (col_idx in candidate_columns_to_be_added):
                columns_that_cover_selected_row.append(col_idx)
        
        # We check the cost effectivenesses
        possible_columns_costs_effectiveness = list()
        for poss_col_idx, possible_col in enumerate(columns_that_cover_selected_row):
            if len(solution) == 0:
                rows_that_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_that_this_column_adds.append(row_idx)
                
                # We evaluate the difference of this col to the solution
                difference_of_this_col_to_solution = len(rows_that_this_column_adds)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
            
            # Here we have to compare against our solution
            else:
                rows_that_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_that_this_column_adds.append(row_idx)
                
                # Here we have to check the rows covered by our solution
                # Now we check the solution rows
                solution_rows = list()
                for s_col in solution:
                    for row_idx, row in enumerate(problem_matrix[:, s_col]):
                        if row == 1 and (row_idx not in solution_rows):
                            solution_rows.append(row_idx)
                
                # Now we check differences between solution and this col
                difference_between_column_and_curr_solution = [i for i in rows_that_this_column_adds if i not in solution_rows]
                # Count the different rows
                difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
        
        # Now we update our problem
        # We now have the cost effectiveness of each column, so we should select the one with lower costs
        selected_column = candidate_columns_to_be_added[np.argmin(possible_columns_costs_effectiveness)]
        
        # We can append this row to the solution
        solution.append(selected_column)
        
        # Compute current cost
        current_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in solution])

        # Update candidate columns to be added
        candidate_columns_to_be_added.remove(selected_column)

        # Update rows covered
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1 and (row_idx not in rows_covered):
                rows_covered.append(row_idx)
        
        # Update number of columns that cover a row
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1:
                number_of_columns_that_cover_a_row[row_idx] -= 1
        
        # Some status prints
        # Rows covered
        # print("Rows Covered: {}".format(rows_covered))
        # print("Number of Rows Covered: {}".format(len(rows_covered)))
        
        # Number of Candidate Columns
        # print("Number of Candidate Columns: {}".format(len(candidate_columns_to_be_added)))

        # Current cost
        # print("Current cost: {}".format(current_cost))

        # Current solution
        # print("Current solution: {}".format(solution))

    # Final Statements
    final_solution = solution
    final_cost = current_cost
    # print("Final Solution is: {}".format(final_solution))
    print("Final Cost is: {}".format(final_cost))


    # Select if we apply redundancy elimination
    if post_processing:
        # Approach 1
        # Initialise variables again
        # Rows covered
        rows_covered = list()

        # Rows to be covered
        rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # We check each row in our candidate columns
        while len(rows_covered) < problem_matrix.shape[0]:
            # Check col with higher number of rows covered
            nr_rows_convered_by_columns = list()
            for possible_col in candidate_columns:
                nr_rows_convered_by_columns.append(np.sum(problem_matrix[:, possible_col]))
            
            # Check the col with higher number of rows
            column_with_more_rows = candidate_columns[np.argmax(nr_rows_convered_by_columns)]

            # Check which rows are covered by this column
            rows_covered_by_col_with_more_rows = list()
            for row_idx, row in enumerate(problem_matrix[:, column_with_more_rows]):
                if row == 1:
                    rows_covered_by_col_with_more_rows.append(row_idx)
            
            # Check columns that cover rows that may be contained in the col with more rows
            columns_contained_in_col_with_more_rows = list()
            for other_cold_idx, other_col in enumerate(candidate_columns):
                if other_col != column_with_more_rows:
                    rows_by_other_col = list()
                    for row_idx, row in enumerate(problem_matrix[:, other_col]):
                        if row == 1:
                            rows_by_other_col.append(row_idx)
                    
                    # Check if this column is contained in the column with more rows covered
                    for row in rows_covered_by_col_with_more_rows:
                        if row in rows_by_other_col:
                            rows_by_other_col.remove(row)
                    
                    # If the len(rows_by_other_col) == 0, then it is contained in this column
                    if len(rows_by_other_col) == 0:
                        columns_contained_in_col_with_more_rows.append(other_col)
            
            # Remove redundant columns
            if len(columns_contained_in_col_with_more_rows) > 0:
                # We remove them from candidate columns list
                for col in columns_contained_in_col_with_more_rows:
                    candidate_columns.remove(col)
                
            # We add the column with more rows to the solution
            processed_solution.append(column_with_more_rows)

            # We remove this columns from candidate columns
            candidate_columns.remove(column_with_more_rows)

            # We update rows covered and rows to be covered
            # Rows to be covered
            for row in rows_covered_by_col_with_more_rows:
                if row in rows_to_be_covered:
                    rows_to_be_covered.remove(row)
            # Rows covered
            for row in rows_covered_by_col_with_more_rows:
                if row not in rows_covered:
                    rows_covered.append(row)
            
            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

        
        # Final Solution
        final_processed_solution = processed_solution
        final_processed_cost = processed_cost
        # print("Final Solution after processing: {}".format(final_processed_solution))
        print("Final Cost after processing: {}".format(final_processed_cost))

        return final_solution, final_cost, final_processed_solution, final_processed_cost

    

    return final_solution, final_cost


# Function CH5: Constructive Heuristics Nr. 5
def ch5(set_covering_problem_instance, post_processing=True, random_seed=42):
    """
    CH5 - Solve the linear programming problem, check most common element frequency (f) and assing xj* >= 1/f
    """
    # Set random seed
    np.random.seed(seed=random_seed)

    # Initialise variables
    # Build Row X Column Matrix
    problem_matrix = np.zeros((set_covering_problem_instance.scp_number_of_rows, set_covering_problem_instance.scp_number_of_columns), dtype=int)
    # Fill Problem Matrix
    for row_idx, row in enumerate(set_covering_problem_instance.scp_instance_all_rows):
        for column in row:
            problem_matrix[row_idx, column-1] = 1
    # De bugging print
    # print(problem_matrix)
    # print(np.where(problem_matrix==1))
    # print(problem_matrix.shape)


    # Set some control variables
    # Rows covered
    rows_covered = list()
    
    # Number of Columns that Cover a Row
    number_of_columns_that_cover_a_row = list()
    # Let's compute the initial number of columns that cover each row
    for row in range(problem_matrix.shape[0]):
        number_of_columns_that_cover_a_row.append(np.sum(problem_matrix[row, :]))
    # Debugging point
    # print(number_of_columns_that_cover_a_row)
    # print(len(number_of_columns_that_cover_a_row))
    
    # Columns to be considered candidates to be added
    candidate_columns_to_be_added = [i for i in range(set_covering_problem_instance.scp_number_of_columns)]
    # Debugging points
    # print(candidate_columns_to_be_added)
    # print(len(candidate_columns_to_be_added))

    # Solution
    solution = list()

    # Variables for the Linear Programming Problem
    # We want to minimize the sum of the costs of the available sets
    objective_to_minimize = [set_covering_problem_instance.scp_instance_column_costs[c] for c in candidate_columns_to_be_added]
    # Left side inequality
    lhs_ineq = problem_matrix.copy()
    # Scipy only deals with <= constraints, since we are dealing with a >=, we multiply by -1
    lhs_ineq = lhs_ineq * -1
    # Right side inequality
    rhs_ineq = np.array([1 for i in range(problem_matrix.shape[0])], dtype=int)
    # Scipy only deals with <= constraints, since we are dealing with a >=, we multiply by -1
    rhs_ineq = rhs_ineq * -1
    # Bounds: all the coeficientes must be higher than zero
    bnd = [(0, np.inf) for i in range(problem_matrix.shape[1])]
    # Solve the problem
    opt = linprog(c=objective_to_minimize, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method="revised simplex")
    print(opt)


    # Begin CH5 Algorithm
    """
    while len(rows_covered) < problem_matrix.shape[0]:
        # Step 1: We check "less" covered rows first
        number_of_cols_of_less_cov_row = number_of_columns_that_cover_a_row[np.argmax(number_of_columns_that_cover_a_row)]
        # We need to be sure that there aren't other less covered rows as well
        less_covered_rows = list()
        for row_idx, cols_that_cover_row in enumerate(number_of_columns_that_cover_a_row):
            if cols_that_cover_row == number_of_cols_of_less_cov_row:
                less_covered_rows.append(row_idx)
        
        # Now that we have the list of less covered rows, we may check the one that contains the cols with lower value of cost-effectiveness
        lower_cost_effectiveness_per_row = list()
        for row in less_covered_rows:
            colums_that_cover_this_row = list()
            for col_idx, column in enumerate(problem_matrix[row, :]):
                if column == 1 and (col_idx in candidate_columns_to_be_added):
                    colums_that_cover_this_row.append(col_idx)
            
            # Now let's check the cost-effectiveness of the columns that cover this row
            cost_eff_of_cols_that_cov_this_row = list()
            if len(solution) == 0:
                for col in colums_that_cover_this_row:
                    rows_that_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, col]):
                        if row == 1:
                            rows_that_this_column_adds.append(row_idx)
                    # Compute the difference for the solution
                    difference_of_this_col_to_solution = len(rows_that_this_column_adds)
                    # Avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[col] / difference_of_this_col_to_solution
                        cost_eff_of_cols_that_cov_this_row.append(effective_cost)
                    else:
                        cost_eff_of_cols_that_cov_this_row.append(np.inf)
            else:
                for col in colums_that_cover_this_row:
                    rows_that_this_column_adds = list()
                    for row_idx, row in enumerate(problem_matrix[:, col]):
                        if row == 1:
                            rows_that_this_column_adds.append(row_idx)

                    # Here we have to check the rows covered by our solution
                    # Now we check the solution rows
                    solution_rows = list()
                    for s_col in solution:
                        for row_idx, row in enumerate(problem_matrix[:, s_col]):
                            if row == 1 and (row_idx not in solution_rows):
                                solution_rows.append(row_idx)
                    
                    # Now we check differences between solution and this col
                    difference_between_column_and_curr_solution = [i for i in rows_that_this_column_adds if i not in solution_rows]
                    # Count the different rows
                    difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                    # Avoid divide by 0
                    if difference_of_this_col_to_solution > 0:
                        effective_cost = set_covering_problem_instance.scp_instance_column_costs[col] / difference_of_this_col_to_solution
                        cost_eff_of_cols_that_cov_this_row.append(effective_cost)
                    else:
                        cost_eff_of_cols_that_cov_this_row.append(np.inf)
                    
            
            # We append the minimum cost effectiveness per row
            lower_cost_effectiveness_per_row.append(np.min(cost_eff_of_cols_that_cov_this_row))
        
        # We now select the row which we want to cover
        selected_row = less_covered_rows[np.argmin(lower_cost_effectiveness_per_row)]
        
        # We now check the columns that cover this row
        columns_that_cover_selected_row = list()
        for col_idx, column in enumerate(problem_matrix[selected_row, :]):
            if column == 1 and (col_idx in candidate_columns_to_be_added):
                columns_that_cover_selected_row.append(col_idx)
        
        # We check the cost effectivenesses
        possible_columns_costs_effectiveness = list()
        for poss_col_idx, possible_col in enumerate(columns_that_cover_selected_row):
            if len(solution) == 0:
                rows_that_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_that_this_column_adds.append(row_idx)
                
                # We evaluate the difference of this col to the solution
                difference_of_this_col_to_solution = len(rows_that_this_column_adds)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
            
            # Here we have to compare against our solution
            else:
                rows_that_this_column_adds = list()
                for row_idx, row in enumerate(problem_matrix[:, possible_col]):
                    if row == 1:
                        rows_that_this_column_adds.append(row_idx)
                
                # Here we have to check the rows covered by our solution
                # Now we check the solution rows
                solution_rows = list()
                for s_col in solution:
                    for row_idx, row in enumerate(problem_matrix[:, s_col]):
                        if row == 1 and (row_idx not in solution_rows):
                            solution_rows.append(row_idx)
                
                # Now we check differences between solution and this col
                difference_between_column_and_curr_solution = [i for i in rows_that_this_column_adds if i not in solution_rows]
                # Count the different rows
                difference_of_this_col_to_solution = len(difference_between_column_and_curr_solution)
                # To avoid divide by 0
                if difference_of_this_col_to_solution > 0:
                    effective_cost = set_covering_problem_instance.scp_instance_column_costs[possible_col] / difference_of_this_col_to_solution
                    possible_columns_costs_effectiveness.append(effective_cost)
                
                else:
                    possible_columns_costs_effectiveness.append(np.inf)
        
        # Now we update our problem
        # We now have the cost effectiveness of each column, so we should select the one with lower costs
        selected_column = candidate_columns_to_be_added[np.argmin(possible_columns_costs_effectiveness)]
        
        # We can append this row to the solution
        solution.append(selected_column)
        
        # Compute current cost
        current_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in solution])

        # Update candidate columns to be added
        candidate_columns_to_be_added.remove(selected_column)

        # Update rows covered
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1 and (row_idx not in rows_covered):
                rows_covered.append(row_idx)
        
        # Update number of columns that cover a row
        for row_idx, row in enumerate(problem_matrix[:, selected_column]):
            if row == 1:
                number_of_columns_that_cover_a_row[row_idx] -= 1
        
        # Some status prints
        # Rows covered
        # print("Rows Covered: {}".format(rows_covered))
        # print("Number of Rows Covered: {}".format(len(rows_covered)))
        
        # Number of Candidate Columns
        # print("Number of Candidate Columns: {}".format(len(candidate_columns_to_be_added)))

        # Current cost
        # print("Current cost: {}".format(current_cost))

        # Current solution
        # print("Current solution: {}".format(solution))

    # Final Statements
    final_solution = solution
    final_cost = current_cost
    # print("Final Solution is: {}".format(final_solution))
    print("Final Cost is: {}".format(final_cost))


    # Select if we apply redundancy elimination
    if post_processing:
        # Approach 1
        # Initialise variables again
        # Rows covered
        rows_covered = list()

        # Rows to be covered
        rows_to_be_covered = [i for i in range(problem_matrix.shape[0]) if i not in rows_covered]

        # Processed solution
        processed_solution = list()

        # Candidate columns
        candidate_columns = final_solution.copy()

        # We check each row in our candidate columns
        while len(rows_covered) < problem_matrix.shape[0]:
            # Check col with higher number of rows covered
            nr_rows_convered_by_columns = list()
            for possible_col in candidate_columns:
                nr_rows_convered_by_columns.append(np.sum(problem_matrix[:, possible_col]))
            
            # Check the col with higher number of rows
            column_with_more_rows = candidate_columns[np.argmax(nr_rows_convered_by_columns)]

            # Check which rows are covered by this column
            rows_covered_by_col_with_more_rows = list()
            for row_idx, row in enumerate(problem_matrix[:, column_with_more_rows]):
                if row == 1:
                    rows_covered_by_col_with_more_rows.append(row_idx)
            
            # Check columns that cover rows that may be contained in the col with more rows
            columns_contained_in_col_with_more_rows = list()
            for other_cold_idx, other_col in enumerate(candidate_columns):
                if other_col != column_with_more_rows:
                    rows_by_other_col = list()
                    for row_idx, row in enumerate(problem_matrix[:, other_col]):
                        if row == 1:
                            rows_by_other_col.append(row_idx)
                    
                    # Check if this column is contained in the column with more rows covered
                    for row in rows_covered_by_col_with_more_rows:
                        if row in rows_by_other_col:
                            rows_by_other_col.remove(row)
                    
                    # If the len(rows_by_other_col) == 0, then it is contained in this column
                    if len(rows_by_other_col) == 0:
                        columns_contained_in_col_with_more_rows.append(other_col)
            
            # Remove redundant columns
            if len(columns_contained_in_col_with_more_rows) > 0:
                # We remove them from candidate columns list
                for col in columns_contained_in_col_with_more_rows:
                    candidate_columns.remove(col)
                
            # We add the column with more rows to the solution
            processed_solution.append(column_with_more_rows)

            # We remove this columns from candidate columns
            candidate_columns.remove(column_with_more_rows)

            # We update rows covered and rows to be covered
            # Rows to be covered
            for row in rows_covered_by_col_with_more_rows:
                if row in rows_to_be_covered:
                    rows_to_be_covered.remove(row)
            # Rows covered
            for row in rows_covered_by_col_with_more_rows:
                if row not in rows_covered:
                    rows_covered.append(row)
            
            # Compute current cost
            processed_cost = np.sum([set_covering_problem_instance.scp_instance_column_costs[c] for c in processed_solution])

        
        # Final Solution
        final_processed_solution = processed_solution
        final_processed_cost = processed_cost
        # print("Final Solution after processing: {}".format(final_processed_solution))
        print("Final Cost after processing: {}".format(final_processed_cost))

        return final_solution, final_cost, final_processed_solution, final_processed_cost

    

    return final_solution, final_cost
    """
    return None