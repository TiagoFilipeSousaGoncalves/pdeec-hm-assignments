# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os
from utilities import SCPInstance

# Function to test if the SCPInstance is working
def test_SCPInstances(data_dir = '../data', scp_instances_dir = 'scp_instances'):
    # Get file list
    scp_instances_list = os.listdir(os.path.join(data_dir, scp_instances_dir))
    scp_instances_list = [i for i in scp_instances_list if not i.startswith('.')]

    # Go trough list
    for i, scp_instance in enumerate(scp_instances_list):
        scp_instance = SCPInstance(os.path.join(data_dir, scp_instances_dir, scp_instance))

        if scp_instance.scp_number_of_columns != scp_instance.scp_instance_column_costs.shape[0]:
            print("Error in number of attributes in SCPInstance {}.".format(scp_instance.scp_instance_filename))

            return 1
        
        elif len(scp_instance.scp_instance_all_rows) != scp_instance.scp_number_of_rows:
            print("Error in number of subsets in SCPInstance {}.".format(scp_instance.scp_instance_filename))

            return 2
    
    return 0