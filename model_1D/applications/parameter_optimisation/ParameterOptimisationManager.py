import numpy as np
import pyswarms as ps
import multiprocessing as mp

from ParameterOptimisationFunctions import PSO_Minimiser



# ---------------------------- Set up parameter optimisation problem --------------------------------

# set parameter names to optimise - ** edit as required to run PO **
parameters = ['gamma_n', 'chi_u', 'chi_b', 'alpha', 'delta']
dim = len(parameters)

# enter parameter minimum and maximum bounds in format [[min1, min2, min3], [max1, max2, max3]]
# NB these are dimensional values (nondimensionalised behind the scenes)
bounds = [[1e8, 0, 0, 1e-11, 1e-4], [1e10, 50, 50, 1e-10, 1e-3]]

options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5} #pyswarms global parameters default 

optimizer = ps.single.GlobalBestPSO(n_particles = 20, dimensions = dim, options=options, bounds=bounds)


# --------------------------------- Run optimisation --------------------------------------------

if __name__ == '__main__':

    # set number of processors - greater than 4 advised due to computational expense 
    n_cpu = mp.cpu_count() - 1

    cost, pos = optimizer.optimize(PSO_Minimiser, iters=10, n_processes = n_cpu)

    print('cost = ', cost)
    print('pos = ', pos)

    with open('po_outputs/pywsarms_output_binding_1D.txt', 'a') as f:
        print(f'{parameters} = {pos}', file=f)
        print(f'cost = {cost}', file=f)
