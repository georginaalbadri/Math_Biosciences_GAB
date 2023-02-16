import sys
sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from src.ModelClasses import CellType, Water, Solute, Grid, BoundSolute
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.TimestepManager import Timestep
from tqdm import tqdm


# Input model data

grid = Grid(T = 1e5, L = 3e-3, H = 3e-3, nx = 100, ny = 100, h=1/100)

print(grid.h)

cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.05, chemotaxis_strength=1.36, bound_chemotaxis_strength=42.4, matrix_drag=9.89e9, matrix_traction=0, contact_inhibition=10.0, aggregation_strength=0)

water = Water(matrix_volume_fraction=0.03, matrix_drag=5e7)

solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0, production_rate=8.74e-11, uptake_rate=0, degradation_rate=8.21e-4, binding_rate=1.5e-3)

bound_solute = BoundSolute(initial_solute_concentration = 0, degradation_rate=8.21e-4, unbinding_rate=3.7e-3)

# Set initial cell and cell velocity distributions

cell.distribution = cell.noisy_cell_distribution(water, grid)

#plt.imshow(cell.distribution, origin='lower')
#plt.colorbar()
#plt.show()

cell.x_velocity = np.zeros((grid.nx, grid.ny))
cell.y_velocity = np.zeros((grid.nx, grid.ny))

solute.distribution = solute.uniform_solute_distribution(water, grid)
bound_solute.distribution = bound_solute.uniform_solute_distribution(water, grid)

initial_cell_distribution = cell.distribution
initial_solute_distribution = solute.distribution

# Nondimensionalise the input parameters

Nondimensionalise(cell, solute, bound_solute, water, grid, cM = 1e-9, cell_viscosity = 1e4)


# Calculate the required size and number of timesteps

dt = CalculateTimestep(cell, solute, grid, dt_multiplier=5)
print(dt)
Tmax = 4.0
nsteps = int(Tmax/dt)
print(nsteps)

# Run the model

cons_init = np.sum(cell.distribution)

for t in tqdm(range(nsteps+1)):
    Timestep(cell, solute, bound_solute, water, grid, dt)

    # Plot the results

    #if t in [int(nsteps/8), int(nsteps/4), int(nsteps/2), int(3*nsteps/4), nsteps]:
    if t in [nsteps]:

        plt.figure()
        plt.imshow(cell.distribution, aspect='auto', origin='lower', interpolation='none', cmap='inferno')
        plt.colorbar()
        #plt.savefig('outputs_1611/binding_inhib/cell_distribution_100_inhib10_t{:.2f}.png'.format(t*dt), dpi=300)
        #np.save('outputs_1611/binding_inhib/cell_distribution_100_inhib10_t{:.2f}.npy'.format(t*dt), cell.distribution)
        #plt.figure()
        #plt.imshow(cell.x_velocity, aspect='auto', origin='lower', interpolation='none', cmap='inferno')
        #plt.colorbar()
        #plt.savefig('cell_x_velocity_t{:.2f}.png'.format(t*dt), dpi=300)
        #plt.figure()
        #plt.imshow(cell.y_velocity, aspect='auto', origin='lower', interpolation='none', cmap='inferno')
        #plt.colorbar()
        #plt.savefig('cell_y_velocity_t{:.2f}.png'.format(t*dt), dpi=300)
        #plt.figure()
        #plt.imshow(solute.distribution, aspect='auto', origin='lower', interpolation='none', cmap='inferno')
        #plt.colorbar()
        #plt.savefig('solute_distribution_t{:.2f}.png'.format(t*dt), dpi=300)

        #plt.show()

cons_final = np.sum(cell.distribution)

print('cons_error =', (cons_final - cons_init)/cons_init)

plt.show()