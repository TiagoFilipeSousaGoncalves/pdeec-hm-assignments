# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os
from timeit import default_timer as timer

# Custom Imports
from local_search_heuristics import lsh1
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
    # Run each lsh for non-processed and processed solutions
    for lsh_idx, lsh in enumerate([lsh1]):
        print("Current: LSH{}".format(lsh_idx+1))
        # And for non-processed and processed solutions:
        for input_results_array in ['ih_1', 'ih_2', 'ih_3', 'ih_4']:
            for appon_array in ['ch1', 'ch2', 'ch5', 'ch4', 'ch3']:
                for boolean in [True, False]:
                    print("Current: {} | Post Processed: {}".format(input_results_array, boolean))
                    # We create a results array to append all the results 
                    results_array = np.zeros(shape=(len(scp_instances_list)+1, 7), dtype='object')

                    # We add the columns meaning to the 1st line
                    results_array[0, 0] = 'SCP Instance Filename'
                    results_array[0, 1] = 'Initial Solution'
                    results_array[0, 2] = 'Initial Cost'
                    results_array[0, 3] = 'Final Solution'
                    results_array[0, 4] = 'Final Cost'
                    results_array[0, 5] = 'History'
                    results_array[0, 6] = 'Elapsed Time Seconds'

                    # Load input results array
                    ih_array = np.load(
                        file=os.path.join(results_dir, 'improvement', '{}_appon_{}_processed_solution_{}.npy'.format(input_results_array, appon_array, boolean)),
                        allow_pickle=True
                        )
                    
                    # Go through all the ch results instances
                    ih_array = ih_array[1:, :]
                    for scp_idx, scp_instance_results in enumerate(ih_array):
                        print("Instance {} of {}".format(scp_idx+1, ih_array.shape[0]))
                        # Start time
                        start = timer()

                        # Append filename
                        results_array[scp_idx+1, 0] = scp_instance_results[0]

                        # Obtain the results
                        initial_solution, initial_cost, final_solution, final_cost, history = lsh(ih_results_array=scp_instance_results, scp_instances_dir=scp_instances_dir)
                        

                        # End time
                        end = timer()
                        
                        # Compute Elapsed Time 
                        elapsed_time = end - start 


                        # Add the results to the array
                        results_array[scp_idx+1, 1] = initial_solution
                        results_array[scp_idx+1, 2] = initial_cost
                        results_array[scp_idx+1, 3] = final_solution
                        results_array[scp_idx+1, 4] = final_cost
                        results_array[scp_idx+1, 5] = history
                        results_array[scp_idx+1, 6] = elapsed_time

                    # We save this results into a Numpy array in results directory
                    # Create results dir if not present
                    if os.path.isdir(results_dir) == False:
                        os.mkdir(results_dir)
                
                    # Create results array filename
                    results_array_filename = 'lsh_{}_appon_{}_and_{}_processed_solution_{}.npy'.format(lsh_idx+1, input_results_array, appon_array, boolean)
                    np.save(file=os.path.join(results_dir, 'local_based_search', results_array_filename), arr=results_array)

# Finish statement
print("Local Based Search Metaheuristics for SCP, finished.")