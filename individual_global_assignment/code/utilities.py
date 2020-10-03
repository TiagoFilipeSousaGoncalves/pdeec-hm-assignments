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
        # Number of Subsets
        self.scp_number_of_subsets = int(scp_file_lines[0][0])
        
        # Number of Attributes
        self.scp_number_of_attributes = int(scp_file_lines[0][1])

        # Attribute Map
        attribute_map = list()
        for i, line in enumerate(scp_file_lines):
            if i > 0:
                if len(attribute_map) < self.scp_number_of_attributes:
                    # attribute_map.append(int(line))
                    for att in line:
                        attribute_map.append(int(att))
                else:
                    subset_index = i
                    break 
        
        self.scp_instance_attribute_map = np.array(attribute_map, dtype=int)


        # Subsets
        # print(scp_file_lines[-1])
        all_subsets = list()
        subset = list()
        for i, line in enumerate(scp_file_lines):
            if len(all_subsets) < self.scp_number_of_subsets:
                if i == subset_index:
                    subset_size = int(line[0])
                    # print(subset_size)
                    
                elif i > subset_index:
                    if len(subset) < subset_size:
                        # subset.append(int(line))
                        for element in line:
                            subset.append(int(element))
                    
                    else:
                        all_subsets.append(np.array(subset, dtype=int))
                        subset_index = i
                        subset_size = int(line[0])
                        # print(subset_size)
                        subset = list()
        
        # Append last subset
        all_subsets.append(np.array(subset, dtype=int))
        # Assign this variable to an attribute variable of the instance 
        self.scp_instance_all_subsets = all_subsets


# Function CH1: Constructive Heuristics Nr. 1
def ch1(set_covering_problem_instance, post_processing=False, random_seed=42):
    """
    CH1 - picking in each constructive step first a still un-covered element and then, second, a random set that covers this element.
    """
    
    # Set random seed
    np.random.seed(seed=random_seed)

    # Status variables
    # Elements that we need to cover: we can update this list along the way
    elements_to_be_covered = list(np.unique(set_covering_problem_instance.scp_instance_attribute_map))
    # print(elements_to_be_covered)
    
    # Number of elements that we already covered: will be updated along the way
    nr_of_elements_covered = 0 
    # print(nr_of_elements_covered)

    # Number of sets that cover an element: each index is an element and the value is the number of sets
    nr_of_sets_that_cover_an_element = [0 for i in range(len(elements_to_be_covered))]
    # print(nr_of_sets_that_cover_an_element)

    # Columns that are eligible for use: this is turned off when we choose a set that contains an element
    candidate_columns = [i for i in range(set_covering_problem_instance.scp_instance_attribute_map.shape[0])]
    # print(candidate_columns)

    # Sets that are available for further use
    candidate_sets = [i for i, _ in enumerate(set_covering_problem_instance.scp_instance_all_subsets)]
    # print(candidate_sets)

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

    # The program must continue until we cover all the elements
    while nr_of_elements_covered != len(elements_to_be_covered):
        # Print current iteration
        print("Iteration {}".format(iteration))

        # Step 1: Pick randomly a still uncovered element
        uncovered_element = np.random.choice(elements_to_be_covered)

        # Step 2: Check which sets contain this element
        # Go through candidate sets
        # Create a temp list to update (in each iteration it is reset)
        sets_that_contain_un_element = list()
        
        # Check the indices in the candidate sets list
        for set_idx in candidate_sets:
            # Choose the set that we will evaluate
            current_set = set_covering_problem_instance.scp_instance_all_subsets[set_idx].copy()
            # Check if this set contains the index of the uncovered element
            # The dataset contains indices that start in 1, we should "Pythonize" this
            current_set = [i-1 for i in current_set]
            # Now, we check if the current set has an index that point to the element
            for e_idx in current_set:
                # Check if this indice matters in this iteration by evaluating candidate columns
                if e_idx in candidate_columns:
                    # Now check if any element index in the current set corresponds to the uncovered element
                    if set_covering_problem_instance.scp_instance_attribute_map[e_idx] == uncovered_element:
                        # We append the set_idx to the list of sets that contain the element
                        if set_idx not in sets_that_contain_un_element:
                            sets_that_contain_un_element.append(set_idx)
            
        
        # With the possible list of sets that contain an element we are now able to randomly choose a set
        selected_set = np.random.choice(sets_that_contain_un_element)

        # Append this set to the solution
        current_solution.append(selected_set)

        # Update the cost: we add 1 per set that is added to the solution
        current_cost = len(current_solution)

        # Remove this set from candidate sets
        candidate_sets.remove(selected_set)

        # Let's update the number of elements to be covered and the columns in candidate columns
        # We update first the number of elements covered
        selected_set_elements = list()
        # Let's choose the subset
        selected_set_array = set_covering_problem_instance.scp_instance_all_subsets[selected_set].copy()
        # We "Pythonize" the subset array of elements
        selected_set_array = [i-1 for i in selected_set_array]
        # Let's check the elements that are present in this subset
        for element_idx in selected_set_array:
            element = set_covering_problem_instance.scp_instance_attribute_map[element_idx]
            if element not in selected_set_elements:
                selected_set_elements.append(element)
        
        # Now we check if this elements are available in the remaining elements to be added to the final solution
        # We update the number of elements that are already covered by this set
        for element in selected_set_elements:
            if element in elements_to_be_covered:
                elements_to_be_covered.remove(element)
                nr_of_elements_covered += 1
        
        # We can also update the columns of the "attribute map" and deactivate/remove the ones that contain the same element
        for col_idx in candidate_columns:
            if set_covering_problem_instance.scp_instance_attribute_map[col_idx] in selected_set_elements:
                candidate_columns.remove(col_idx)
        
        # Some debugging prints
        print("Number of Elements Already Covered: {}".format(nr_of_elements_covered))
        print("Current Cost: {}".format(current_cost))
        print("Current Solution")
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