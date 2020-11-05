# Imports 
import numpy as np 
import matplotlib.pyplot as plt
import _pickle as cPickle
import os

# Directories
results_dir = '../results'
figs_dir = os.path.join(results_dir, 'figures')
if os.path.isdir(figs_dir) == False:
    os.mkdir(figs_dir)

# Results array
results_arr_fname = 'lsh_1_appon_ih_1_and_ch1_processed_solution_False.npy'
results = np.load(file=os.path.join(results_dir, "local_based_search", results_arr_fname), allow_pickle=True)

# Choose an instance
scp_instance_results = results[21]
history = scp_instance_results[5]
# print(history)
iterations = [i[0] for i in history]
costs = [i[1] for i in history]
temperatures = [i[2] for i in history]

# Begin figure


# plt.plot(iterations, costs, label='Cost')
plt.plot(iterations, temperatures, label='Temperature')

plt.xlabel("Iterations")
# plt.ylabel("Costs")
plt.ylabel("Temperature")



plt.title("LSH #1 Example")

plt.legend()

plt.savefig(os.path.join(figs_dir, "LSH1_Temp_Example.png"))