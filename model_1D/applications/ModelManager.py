import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from src.ModelClasses import Geometry, CellType, Water, Solute, Grid, BoundSolute
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.TimestepManager import Timestep
from tqdm import tqdm
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)


# Input model data - optimised for binding

#geometry = Geometry(T = 1e5, L = 3e-3)
#
#cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=1.36, chemotaxis_strength_bound=42.4, matrix_drag=9.89e9, matrix_traction=0.0, contact_inhibition=0.0, aggregation_strength=0)
#
#water = Water(matrix_volume_fraction=0.03, matrix_drag=5e7)
#
#solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=8.74e-11, uptake_rate=0, degradation_rate=8.21e-4, binding_rate=1.5e-3)
#
#bound_solute = BoundSolute(initial_solute_concentration=0, unbinding_rate=3.7e-3, degradation_rate=8.21e-4)


# Input model data - optimised core

geometry = Geometry(T = 1e5, L = 3e-3)

cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=36.7, chemotaxis_strength_bound=0, matrix_drag=4.62e9, matrix_traction=0.0, contact_inhibition=0.0, aggregation_strength=0)

water = Water(matrix_volume_fraction=0.03, matrix_drag=5e7)

solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=7.41e-11, uptake_rate=0, degradation_rate=9.96e-4, binding_rate=0)

bound_solute = BoundSolute(initial_solute_concentration=0, unbinding_rate=0, degradation_rate=0)

# Initialise grid

grid = Grid(nx = 500)

# Set initial cell and cell velocity distributions

cell.distribution = cell.noisy_cell_distribution(water, grid)

cell.velocity = np.zeros(grid.nx)

solute.distribution = solute.uniform_solute_distribution(water, grid)

bound_solute.distribution = bound_solute.uniform_solute_distribution(water, grid)

initial_cell_distribution = cell.distribution
initial_solute_distribution = solute.distribution

# Nondimensionalise the input parameters

Nondimensionalise(cell, solute, bound_solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4)

# Calculate the required size and number of timesteps

dt = CalculateTimestep(cell, solute, grid, dt_multiplier=200)
print(dt)
Tmax = 2
nsteps = int(Tmax/dt)


param_list = [0, 0.5, 1.5]
cell_dist_list = []

plt.figure(figsize=(6, 4))
legend = []

for param in param_list:

    #cell.matrix_traction = param * geometry.T / 1e4
    #solute.uptake_rate = param * geometry.T / 1e-9
    cell.aggregation_strength = param * geometry.T / 1e4
    #cell.contact_inhibition = param * geometry.T / 1e4

    # Run the model

    for t in tqdm(range(nsteps)):
        Timestep(cell, solute, bound_solute, water, grid, dt)

    # plt.plot(grid.dx*np.arange(grid.nx), cell.distribution, label='{}'.format(cell.matrix_traction))
    plt.plot(grid.dx*np.arange(grid.nx), cell.distribution)
    if np.abs(cell.aggregation_strength) > 999 or np.abs(cell.aggregation_strength) < 0.01 and cell.aggregation_strength != 0:
        legend.append(r'$\nu$ = ' + f'{cell.aggregation_strength:.0E}')
    else:
        legend.append(r'$\nu$ = ' + f'{cell.aggregation_strength:.0f}')

#plt.title('Addition of cell-cell contact inhibition')
plt.ylabel('Cell volume fraction [-]')
plt.xlabel('Height [-]')
plt.legend(legend, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('figures/additional_params/cell_aggregation.png', dpi = 300)


"""
# Plot the results

plt.figure()
plt.plot(grid.dx*np.arange(grid.nx), initial_cell_distribution, 'b', label='Initial cell distribution')
plt.plot(grid.dx*np.arange(grid.nx), cell.distribution, 'r', label='Final cell distribution')
plt.legend()
plt.savefig('outputs/CellDistribution_binding.png', dpi=300)


plt.figure()
plt.plot(grid.dx*np.arange(grid.nx), water.distribution, 'r', label='Water distribution')
plt.legend()
plt.savefig('outputs/WaterDistribution.png', dpi=300)

plt.figure()
plt.plot(grid.dx*np.arange(grid.nx), water.pressure, 'b', label='Water pressure')
plt.legend()
plt.savefig('outputs/WaterPressure.png', dpi=300)

plt.figure()
plt.plot(grid.dx*np.arange(grid.nx), initial_solute_distribution, 'b', label='Initial solute distribution')
plt.plot(grid.dx*np.arange(grid.nx), solute.distribution, 'r', label='Solute distribution')
plt.plot(grid.dx*np.arange(grid.nx), bound_solute.distribution, 'g', label='Bound solute distribution')
plt.legend()
plt.savefig('outputs/SoluteDistribution.png', dpi=300)

plt.figure()
plt.plot(grid.dx*np.arange(grid.nx), cell.velocity, 'b', label='Cell velocity')
plt.legend()
plt.savefig('outputs/CellVelocity.png', dpi=300)
"""