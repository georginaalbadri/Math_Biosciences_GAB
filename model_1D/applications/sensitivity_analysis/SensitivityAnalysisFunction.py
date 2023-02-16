import sys
sys.path.insert(0, '../../')

import numpy as np
from scipy.ndimage import label

from src.ModelClasses import Geometry, CellType, Water, Solute, BoundSolute, Grid
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.TimestepManager import Timestep




#-----------------------------------------------------------------------------------------------------------------------

def SA_initialconditions(param_values_sample):

    #random_seed, magnitude, spacing of initial cell distribution 

    # Input model data

    geometry = Geometry(T = 1e5, L = 3e-3)

    cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=36.7, matrix_drag=4.6e9, matrix_traction=0.0, contact_inhibition=0.0)

    water = Water(matrix_volume_fraction = 0.03)

    solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=7.4e-11, degradation_rate=9.96e-4, uptake_rate=0.0)


    # Initialise grid

    grid = Grid(nx = 480) # nx must be divisible by spacing (1, 2, 3, 4, 5, 6)

    # Unpack parameter values from sampler

    seed = int(param_values_sample[0])
    magnitude = param_values_sample[1]
    spacing = int(param_values_sample[2])

    # Set initial cell and cell velocity distributions

    cell.distribution = cell.SA_cell_distribution(water, grid, seed, magnitude, spacing)

    cell.velocity = np.zeros(grid.nx)

    solute.distribution = solute.uniform_solute_distribution(water, grid)

    # Nondimensionalise the input parameters

    Nondimensionalise(cell, solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4)

    # Calculate the required size and number of timesteps

    dt = CalculateTimestep(cell, solute, grid, dt_multiplier = 200)
    Tmax = 4.0
    nsteps = int(Tmax/dt)

    # Run the model

    tlist = [int(i*nsteps/4) for i in range(1, 5)]

    clustlist = []

    for t in range(nsteps+1):
        Timestep(cell, solute, water, grid, dt)

        if t in tlist:
            
            #-- number of clusters
            nfilter = np.where(cell.distribution>0.1, cell.distribution, 0)
            _, clustnum = label(nfilter)
            

            #-- check number of clusters is correct
            clustcheck = []

            for height in [0.075, 0.08, 0.085, 0.09, 0.095]:
                nfiltercheck = np.where(cell.distribution > height, cell.distribution, 0)
                _, clustnumcheck = label(nfiltercheck)
                clustcheck.append(clustnumcheck)

            if min(clustcheck) < clustnum:
                clustlist.append(min(clustcheck) + 1)
            else:
                clustlist.append(clustnum + 1)


    return clustlist

#-----------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------

def SA_coreparameters(param_values_sample):

    # Unpack parameter values from sampler - *** edit as required ***

    #chemo = param_values_sample[0]
    #drag = param_values_sample[1]
    #prod = param_values_sample[2]
    #deg = param_values_sample[3]
    uptake = param_values_sample[0]
    trac = param_values_sample[1]
    agg = param_values_sample[2]
    inhib = param_values_sample[3]

    # Input model data

    geometry = Geometry(T = 1e5, L = 3e-3)

    cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=40.1, matrix_drag=4.34e-9, matrix_traction=trac, contact_inhibition=inhib, aggregation_strength=agg)

    water = Water(matrix_volume_fraction = 0.03)

    solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=8.92e-11, degradation_rate=9e-4, uptake_rate=uptake)

    bound_solute = BoundSolute(initial_solute_concentration = 0, unbinding_rate=0, degradation_rate=0)

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
    Tmax = 4.0
    nsteps = int(Tmax/dt)

    tlist = [int(i*nsteps/4) for i in range(1, 5)]

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

        clustlist = []

        for t in range(nsteps+1):
            Timestep(cell, solute, bound_solute, water, grid, dt)

            if t in tlist:
            
                #-- number of clusters
                nfilter = np.where(cell.distribution>0.1, cell.distribution, 0)
                _, clustnum = label(nfilter)


                #-- check number of clusters is correct
                clustcheck = []

                for height in [0.075, 0.08, 0.085, 0.09, 0.095]:
                    nfiltercheck = np.where(cell.distribution > height, cell.distribution, 0)
                    _, clustnumcheck = label(nfiltercheck)
                    clustcheck.append(clustnumcheck)

                if min(clustcheck) < clustnum:
                    clustlist.append(min(clustcheck) + 1)
                else:
                    clustlist.append(clustnum + 1)

        allclustlist.append(clustlist)
    
    # Average the results over X runs

    avgclustlist = np.mean(allclustlist, axis=0)


    return avgclustlist
    
#-----------------------------------------------------------------------------------------------------------------------



