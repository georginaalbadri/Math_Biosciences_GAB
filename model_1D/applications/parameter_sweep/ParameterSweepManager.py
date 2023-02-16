import sys
sys.path.insert(0, '../../')

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from SALib.sample import saltelli
from tqdm import tqdm
import time

from applications.parameter_sweep.ParameterSweepFunction import PS_parameters



# Conduct parameter sweep using saltelli sampling to generate parameter samples given the parameter bounds


# ---------------------------- PS SETUP -------------------------- #

# set parameter value bounds - ** edit as required for parameter sweep ** (NB all inputs are dimensional values, they are nondimensionalised behind the scenes)

problem = {
    'num_vars': 5,
    'names' : ['chemotaxis', 'bound chemotaxis', 'cell-matrix drag', 'production rate', 'degradation rate'],
    'bounds': [[0, 50], [0, 50], [1e8, 1e10], [1e-11, 9e-11], [1e-4, 1e-3]]
}

param_values = saltelli.sample(problem, 128, skip_values = 256, calc_second_order = False)

nparams = len(param_values)

print(f'number of parameter samples to run = {nparams}')


# ------------------------------ PS -------------------------------- #

if __name__ ==  '__main__':

    n_cpu = mp.cpu_count() - 1

    start_time = time.time()

    with Pool(processes = n_cpu) as pool:
        results = list(tqdm(pool.imap(PS_parameters, param_values), total = nparams))

    print('time taken', time.time() - start_time)

