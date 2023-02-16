import numpy as np
import matplotlib.pyplot as plt
from src.ModelClasses import Geometry, CellType, Water, Solute, Grid, BoundSolute
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.TimestepManager import Timestep
from tqdm import tqdm


# Input model geometry and (dimensional) parameters

geometry = Geometry(T = 1e5, L = 3e-3)

cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=1.36, chemotaxis_strength_bound=42.4, matrix_drag=9.89e9, matrix_traction=0.0, contact_inhibition=0.0, aggregation_strength=0)

water = Water(matrix_volume_fraction=0.03, matrix_drag=5e7)

solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=8.74e-11, uptake_rate=0, degradation_rate=8.21e-4, binding_rate=1.5e-3)

bound_solute = BoundSolute(initial_solute_concentration=0, unbinding_rate=3.7e-3, degradation_rate=8.21e-4)

# Initialise FD grid

grid = Grid(nx = 500)

# Set initial cell, cell velocity, and VEGF distributions

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

# Run the model

for t in tqdm(range(nsteps)):
    Timestep(cell, solute, bound_solute, water, grid, dt)

# Plot the results

plt.figure()
plt.plot(grid.dx*np.arange(grid.nx), initial_cell_distribution, 'b', label='Initial cell distribution')
plt.plot(grid.dx*np.arange(grid.nx), cell.distribution, 'r', label='Final cell distribution')
plt.legend()
plt.savefig('outputs/CellDistribution_bindingmodel.png', dpi=300)

"""
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