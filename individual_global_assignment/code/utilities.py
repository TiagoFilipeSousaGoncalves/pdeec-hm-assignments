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
            l = [e.strip() for e in line]
            scp_file_lines.append(l)

        # Close the file
        scp_instance_textfile.close()

        # Perform some data cleaning
        # for i, line in enumerate(scp_file_lines):
            # line.remove('\n')
            # line.remove('')

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
                    subset_size = int(line)
                    # print(subset_size)
                    
                elif i > subset_index:
                    if len(subset) < subset_size:
                        # subset.append(int(line))
                        for element in line:
                            subset.append(int(element))
                    
                    else:
                        all_subsets.append(np.array(subset, dtype=int))
                        subset_index = i
                        subset_size = int(line)
                        print(subset_size)
                        subset = list()
        
        self.scp_instance_all_subsets = all_subsets