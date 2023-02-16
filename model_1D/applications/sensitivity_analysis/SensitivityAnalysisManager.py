import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm
import time

from SensitivityAnalysisFunction import SA_initialconditions, SA_coreparameters



# ---------------------------- SA SETUP -------------------------- #

# set parameter value bounds - ** edit as required for sensitivity analysis ** 

# parameters related to initial conditions
#problem = {
#    'num_vars': 3,
#    'names': ['seed', 'magnitude', 'spacing'],
#    'bounds': [[0, 100], [0, 0.25], [1, 6]] }

# all core and additional parameters
#problem = {
#    'num_vars': 8,
#    'names' : ['chemotaxis', 'cell-matrix drag', 'production rate', 'degradation rate', 'uptake rate', 'cell-matrix traction', 'aggregation', 'contact inhibition'],
#    'bounds': [[10, 50], [1e8, 1e10], [1e-11, 9e-11], [1e-4, 1e-3], [5e-12, 2e-10], [0, 10], [0, 10], [0, 10]]
#}

# additional parameters only 
problem = {
    'num_vars': 4,
    'names' : ['uptake rate', 'cell-matrix traction', 'aggregation', 'contact inhibition'],
    'bounds': [[0, 2e-10], [0, 10], [0, 6], [0, 10]]
}


param_values = saltelli.sample(problem, 1024, skip_values = 2048, calc_second_order = False)

nparams = len(param_values)

print(f'number of parameter samples to run = {nparams}')


# ------------------------------ SA -------------------------------- #

if __name__ ==  '__main__':

    # set number of processors to run on based on computer capacity - NB sensitivity analysis is expensive and > 4 cores is advised 
    n_cpu = mp.cpu_count() - 1

    start_time = time.time()

    with Pool(processes = n_cpu) as pool:
        #results = list(tqdm(pool.imap(SA_initialconditions, param_values), total = nparams))
        results = list(tqdm(pool.imap(SA_coreparameters, param_values), total = nparams))

    print('time taken', time.time() - start_time)


    #-- process and save results

    results = np.array(results)

    results_set = np.transpose(results)

    print(np.shape(results_set))

    #np.save('sa_outputs/initial_conds/SA_results_initialconditions.npy', results_set)
    np.save('sa_outputs/additional_params/SA_results_addparameters.npy', results_set)

    #-- analyse results (for each timepoint separately)

    ST_set = []   
    ST_conf = []   


    for result in results_set:
        Si = sobol.analyze(problem, result, print_to_console = True, calc_second_order = False)
        ST_set.append(Si['ST'])
        ST_conf.append(Si['ST_conf'])

    ST_set=np.array(ST_set)
    ST_set=np.transpose(ST_set)

    ST_conf=np.array(ST_conf)
    ST_conf=np.transpose(ST_conf)

    #-- save SA results for each timepoint on a new line 
#    with open('sa_outputs/additional_params/SA_results_addparameters_ST.txt', 'w') as f:
#        for ST in ST_set:
#            f.write(str(ST) + '\n')
