import sys
sys.path.insert(1, '../../')

import numpy as np
from scipy.ndimage import label
import pyswarms as ps

from src.ModelClasses import Geometry, CellType, Water, Solute, BoundSolute, Grid
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.TimestepManager import Timestep



@ps.cost
def PSO_Minimiser(x):

    # Unpack parameter sample x - ** edit as required based on inputs in ParameterOptimisationManager **

    gamma, chi_u, chi_b, alpha, delta = x

    # Input model data to dataclasses

    geometry = Geometry(T = 1e5, L = 3e-3)

    cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=chi_u, chemotaxis_strength_bound=chi_b, matrix_drag=gamma, matrix_traction=0.0, contact_inhibition=0.0)

    water = Water(matrix_volume_fraction=0.03, matrix_drag=5e7)

    solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=alpha, uptake_rate=0, degradation_rate=delta, binding_rate=1.5e-3)

    bound_solute = BoundSolute(initial_solute_concentration = 0, degradation_rate=delta, unbinding_rate=3.7e-3)

    # Initialise grid

    grid = Grid(nx = 500)

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

    # Run the model X times to account for stochasticity in the initial conditions

    seedlist = [1382016804, 3632472886, 3243918257, 3370522462, 1570078188, 2665051601, 2842329975, 926654616, 1391377836, 3802922179]

    clustlist = []

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

        #-- check number of clusters is correct
        clustcheck = []
        for height in [0.075, 0.08, 0.085, 0.09, 0.095]:
            nfiltercheck = np.where(cell.distribution > height, cell.distribution, 0)
            _, clustnumcheck = label(nfiltercheck)
            clustcheck.append(clustnumcheck)

        clustcheck = min(clustcheck)
        clustlist.append(min(clustnum, clustcheck))
    
    # Average the results over X runs

    avgclustnum = np.mean(clustlist)

    #print('avg clustnum =', avgclustnum)

    minimise = 20 - avgclustnum

    return minimise 
