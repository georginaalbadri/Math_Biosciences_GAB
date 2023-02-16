import numpy as np
from scipy.ndimage import label

from src.ModelClasses import Geometry, CellType, Water, Solute, Grid, BoundSolute
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.TimestepManager import Timestep




#-----------------------------------------------------------------------------------------------------------------------

def PS_parameters(param_values_sample):

    # Unpack parameter values from sampler - *** edit parameter names as required based on input ***

    chemo = param_values_sample[0]
    chemo_b = param_values_sample[1]
    drag = param_values_sample[2]
    prod = param_values_sample[3]
    deg = param_values_sample[4]

    # Input model data

    geometry = Geometry(T = 1e5, L = 3e-3)

    cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=chemo, matrix_drag=drag, chemotaxis_strength_bound=chemo_b)

    water = Water(matrix_volume_fraction = 0.03)

    solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=prod, degradation_rate=deg, uptake_rate=0.0, binding_rate=1.5e-3)

    bound_solute = BoundSolute(initial_solute_concentration=0, degradation_rate=deg, unbinding_rate=3.7e-3)

    # Initialise grid

    grid = Grid(nx = 600)

    # Set initial cell and cell velocity distributions

    cell.distribution = cell.noisy_cell_distribution(water, grid)

    cell.velocity = np.zeros(grid.nx)

    solute.distribution = solute.uniform_solute_distribution(water, grid)

    bound_solute.distribution = bound_solute.uniform_solute_distribution(water, grid)

    # Nondimensionalise the input parameters

    Nondimensionalise(cell, solute, bound_solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4)

    # Calculate the required size and number of timesteps

    dt = CalculateTimestep(cell, solute, grid, dt_multiplier=100)
    Tmax = 2.0
    nsteps = int(Tmax/dt)

    # Run the model X times to account for stochasticity in the initial conditions

    seedlist = [1382016804, 3632472886, 3243918257, 3370522462, 1570078188, 2665051601, 2842329975, 926654616, 1391377836, 3802922179]

    allclustlist = []

    for seed in seedlist:

        # Reset initial distributions
        cell.distribution = cell.SA_cell_distribution(water, grid, seed=seed, noise_level=0.1, spacing=5)
        water.distribution = water.available_volume_fraction - cell.distribution
        cell.velocity = np.zeros(grid.nx)
        water.velocity = np.zeros(grid.nx)
        water.pressure = np.zeros(grid.nx)
        solute.distribution = solute.uniform_solute_distribution(water, grid)

        for t in range(nsteps+1):
            Timestep(cell, solute, bound_solute, water, grid, dt)


        #-- number of clusters
        nfilter = np.where(cell.distribution>0.1, cell.distribution, 0)
        _, clustnum = label(nfilter)

        #-- check number of clusters is correct (i.e. check for and remove instances of double peaks)
        clustcheck = []

        for height in [0.075, 0.08, 0.085, 0.09, 0.095]:
            nfiltercheck = np.where(cell.distribution > height, cell.distribution, 0)
            _, clustnumcheck = label(nfiltercheck)
            clustcheck.append(clustnumcheck)

        if min(clustcheck) < clustnum:
            allclustlist.append(min(clustcheck))
        else:
            allclustlist.append(clustnum)
    
    # Average the results over X runs

    avgclust = np.mean(allclustlist)

    # Return the average number of clusters and corresponding parameter values to txt file

    with open('ps_outputs/binding_cluster_number_parameter_sweep_data.txt', 'a') as f:
        f.write(f'{avgclust}, {chemo}, {chemo_b}, {drag}, {prod}, {deg} \n')

    return
    
#-----------------------------------------------------------------------------------------------------------------------



