# Imports
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import _pickle as cPickle
import os

# Custom Imports
from utilities import SCPInstance, ch1, ch2, ch3, ch4, ch5
from improvement_heuristics import ih1, ih2, ih3
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
    pass
