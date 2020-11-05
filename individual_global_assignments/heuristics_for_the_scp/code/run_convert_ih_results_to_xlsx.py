# Imports
import numpy as np
import pandas as pd 
import os
import pickle

# Directories
results_dir = '../results/improvement'

# List Numpy results files
results_npy_files = os.listdir(results_dir)
results_npy_files = [f for f in results_npy_files if not f.startswith('.')]

# There should be 40 results files
if len(results_npy_files) == 40:
    excel_writer = pd.ExcelWriter(path=os.path.join(results_dir, 'results_excel_ih.xlsx'), engine='xlsxwriter')
    for idx, results_file in enumerate(results_npy_files):
        results_array = np.load(file=os.path.join(results_dir, results_file), allow_pickle=True)
        results_df = pd.DataFrame(data=results_array, columns=[c for c in results_array[0, :]])
        sheet_name = results_file.split('.')[0]
        sheet_name = sheet_name.split('_')
        sheet_name = sheet_name[0] +  sheet_name[1] + '_' + sheet_name[3] + '_' + sheet_name[6]
        print(sheet_name)
        results_df.to_excel(excel_writer, sheet_name=sheet_name)
    
    excel_writer.save()

print("Finished.")