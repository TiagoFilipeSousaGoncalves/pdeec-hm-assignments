# Imports
import numpy as np
import pandas as pd 
import os
import pickle

# Directories
results_dir = '../results/constructive'

# List Numpy results files
results_npy_files = os.listdir(results_dir)
results_npy_files = [f for f in results_npy_files if not f.startswith('.')]

# There should be 5 results files
if len(results_npy_files) == 5:
    excel_writer = pd.ExcelWriter(path=os.path.join(results_dir, 'results_excel_ch.xlsx'), engine='xlsxwriter')
    for idx, results_file in enumerate(results_npy_files):
        results_array = np.load(file=os.path.join(results_dir, results_file), allow_pickle=True)
        results_df = pd.DataFrame(data=results_array, columns=[c for c in results_array[0, :]])
        results_df.to_excel(excel_writer, sheet_name=results_file)
    
    excel_writer.save()

print("Finished.")