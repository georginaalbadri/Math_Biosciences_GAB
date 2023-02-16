import numpy as np
from pytest import approx
from src.TimestepManager import Timestep
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.ModelClasses import Geometry, Grid, CellType, Water, Solute


def test_Timestep():
    """
    Test timestep manager runs successfully 
    """

    # Input model data
    geometry = Geometry(T = 1e5, L = 1e-3)
    endothelial = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.1)
    water = Water(matrix_volume_fraction = 0.03, available_volume_fraction = 0.97)
    grid = Grid(nx = 10)
    solute = Solute(diffusion_rate = 1e-11, initial_solute_concentration = 0.1)

    # Set initial cell and cell velocity distributions
    endothelial.distribution = endothelial.noisy_cell_distribution(water, grid)
    endothelial.velocity = np.zeros(grid.nx)
    solute.distribution = solute.uniform_solute_distribution(water, grid)

    # Nondimensionalise the input parameters
    Nondimensionalise(endothelial, solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4)

    # Calculate the required size and number of timesteps
    dt = CalculateTimestep(endothelial, solute, grid, dt_multiplier=100)
    nsteps = 1

    # Run the model
    for t in range(nsteps):
        Timestep(endothelial, solute, water, grid, dt)

    # Test timestep functionality 
    assert len(endothelial.distribution) == len(endothelial.velocity) == len(solute.distribution) == \
                                len(water.pressure) == len(water.velocity) == len(water.distribution) == grid.nx

    tot_vol = endothelial.distribution + water.distribution + water.matrix_volume_fraction

    assert tot_vol.all() == approx(1.0, rel=1e-6)

    return 