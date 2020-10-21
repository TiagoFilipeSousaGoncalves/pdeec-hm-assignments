# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os

# Custom Imports
from utilities import SCPInstance, ch1, ch2, ch3, ch4, ch5
from tests import test_SCPInstances

# Directories
data_dir = '../data'
results_dir = '../results'
scp_instances_dir = os.path.join(data_dir, 'scp_instances')

# List of SCP Instances: Should be 42
scp_instances_list =[i for i in os.listdir(scp_instances_dir) if not i.startswith('.')]
print("Number of instances in folder: {}".format(len(scp_instances_list)))

# Check if the .txt files are not corrupted
test_result = test_SCPInstances()
# If we pass the test, we run everything
if test_result == 0:
    # Run each CH from all the CH set
    for ch_idx, ch in enumerate([ch1, ch2, ch3, ch4, ch5]):
        print("Current: CH{}".format(ch_idx+1))
        # We create a results array to append all the results 
        results_array = np.zeros(shape=(len(scp_instances_list)+1, 5), dtype='object')

        # We add the columns meaning to the 1st line
        results_array[0, 0] = 'SCP Instance Filename'
        results_array[0, 1] = 'Solution'
        results_array[0, 2] = 'Cost'
        results_array[0, 3] = 'Processed Solution'
        results_array[0, 4] = 'Processed Cost'

        # We go through all the SCP Instances available
        for scp_idx, scp_instance_filename in enumerate(scp_instances_list):
            scp_instance = SCPInstance(os.path.join(scp_instances_dir, scp_instance_filename))

            # We add the filename to the results array
            results_array[scp_idx+1, 0] = scp_instance_filename

            # We obtain the results
            final_solution, final_cost, final_processed_solution, final_processed_cost = ch(set_covering_problem_instance=scp_instance, post_processing=True, random_seed=42)

            # We add the results to the array
            results_array[scp_idx+1, 1] = final_solution
            results_array[scp_idx+1, 2] = final_cost
            results_array[scp_idx+1, 3] = final_processed_solution
            results_array[scp_idx+1, 4] = final_processed_cost
        
        # We save this results into a Numpy array in results directory
        # Create results dir if not present
        if os.path.isdir(results_dir) == False:
            os.mkdir(results_dir)
        
        # Create results array filename
        results_array_filename = 'ch{}_results.npy'.format(ch_idx+1)
        np.save(file=os.path.join(results_dir, results_array_filename), arr=results_array)
    
# Finish statement
print("Contructive Heuristics for SCP, finished.")