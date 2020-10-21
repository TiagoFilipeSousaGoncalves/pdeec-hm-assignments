# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _pickle as cPickle
import os
# from utilities import SCPInstance, ch1, ch2, ch3, ch4, ch5
from improvement_heuristics import ih1, ih2, ih3, ih4
from tests import test_SCPInstances


# Directories
data_dir = '../data'
scp_instances_dir = os.path.join(data_dir, 'scp_instances')

results_dir = '../results'
results_array = os.path.join(results_dir, 'ch1_results.npy')
results_array = np.load(results_array, allow_pickle=True)

# scp_instances_list =[i for i in os.listdir(scp_instances_dir) if not i.startswith('.')]

# print("Number of instances in folder: {}".format(len(scp_instances_list)))

test_result = test_SCPInstances()

if test_result == 0:
    for i in [10]:
        results_array = results_array[i]
        print("IH 1")
        _, initial_cost_11, _, final_cost_11 = ih1(ch_results_array=results_array, scp_instances_dir=scp_instances_dir, random_seed=42, max_iterations=1000000, patience=1000, tabu_thr=10)
        print("IH 2")
        _, initial_cost_12, _, final_cost_12 = ih2(ch_results_array=results_array, scp_instances_dir=scp_instances_dir, random_seed=42, max_iterations=1000000, patience=1000, tabu_thr=10)
        print("IH 4")
        _, initial_cost_14, _, final_cost_14 = ih4(ch_results_array=results_array, scp_instances_dir=scp_instances_dir, random_seed=42, set_minimization_repetition_factor=5, hill_climbing_repetition_factor=5)
    # for i in range(len(scp_instances_list)):
        # test = SCPInstance(os.path.join(scp_instances_dir, scp_instances_list[i]))
        # print(test.scp_instance_filename, test.scp_number_of_rows, test.scp_number_of_columns, test.scp_instance_column_costs.shape, len(test.scp_instance_all_rows))
        # print("\nCH1: ")
        # sol, cost, proc_sol, proc_cost = ch1(set_covering_problem_instance=test, post_processing=True)
        # print("\nCH2: ")
        # sol, cost, proc_sol, proc_cost = ch2(set_covering_problem_instance=test, post_processing=True)
        # print("\nCH3: ")
        # sol, cost, proc_sol, proc_cost = ch3(set_covering_problem_instance=test, post_processing=True)
        # print("\nCH4: ")
        # sol, cost, proc_sol, proc_cost = ch4(set_covering_problem_instance=test, post_processing=True)
        # sol, cost = ch1(set_covering_problem_instance=test, post_processing=False)
        # print("\nCH5: ")
        # sol, cost, proc_sol, proc_cost = ch5(set_covering_problem_instance=test, post_processing=True)
# sol, cost, proc_sol, proc_cost

print(initial_cost_11, final_cost_11)
print(initial_cost_12, final_cost_12)
print(initial_cost_14, final_cost_14)