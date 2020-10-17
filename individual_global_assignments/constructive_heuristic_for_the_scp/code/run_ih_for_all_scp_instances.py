# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os
from timeit import default_timer as timer

# Custom Imports
# from utilities import SCPInstance, ch1, ch2, ch3, ch4, ch5
from improvement_heuristics import ih1, ih2, ih3, ih4
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
    # Run each IH for non-processed and processed solutions
    # for ih_idx, ih in enumerate([ih1, ih2, ih3, ih4]):
    for ih_idx, ih in enumerate([ih1]):
        print("Current: IH{}".format(ih_idx+1))
        # And for non-processed and processed solutions:
        for boolean in [False, True]:
            for input_results_array in ['ch1', 'ch2', 'ch5', 'ch3', 'ch4']:
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
                ch_array = np.load(
                    file=os.path.join(results_dir, '{}_results.npy'.format(input_results_array)),
                    allow_pickle=True
                    )
                
                # Go through all the ch results instances
                ch_array = ch_array[1:, :]
                for scp_idx, scp_instance_results in enumerate(ch_array):
                    print("Instance {} of {}".format(scp_idx+1, ch_array.shape[0]))
                    # Start time
                    start = timer()

                    # Append filename
                    results_array[scp_idx+1, 0] = scp_instance_results[0]

                    # Obtain the results
                    try:
                        initial_solution, initial_cost, final_solution, final_cost, history = ih(ch_results_array=scp_instance_results, scp_instances_dir=scp_instances_dir, use_processed_solution=boolean, random_seed=42, max_iterations=1000000, patience=1000, tabu_thr=10)
                    
                    except:
                        initial_solution, initial_cost, final_solution, final_cost, history = ih(ch_results_array=scp_instance_results, scp_instances_dir=scp_instances_dir, use_processed_solution=boolean, random_seed=42, set_minimization_repetition_factor=5, hill_climbing_repetition_factor=5)
                    

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
            results_array_filename = 'ih_{}_appon_{}_processed_solution_{}.npy'.format(ih_idx+1, input_results_array, boolean)
            np.save(file=os.path.join(results_dir, results_array_filename), arr=results_array)

# Finish statement
print("Improvement Heuristics for SCP, finished.")