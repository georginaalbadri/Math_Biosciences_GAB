import numpy as np
import matplotlib.pyplot as plt
from src.CellSolver import cell_solver
from src.PressureSolver import pressure_solver
from src.CellVelocitySolver import x_velocity_solver, y_velocity_solver
from src.SoluteSolver import vegf_solver
from src.WaterVelocitySolver import water_velocity_solver



def Timestep(cell, solute, bound_solute, water, grid, dt):

    """
    Updates the cell distribution, water distribution, solute distribution, water pressure distribution, 
    water velocity distribution, and cell velocity distribution for a single timestep.

    Parameters
    ----------
    cell : Cell object
    solute : Solute object
    water : Water object
    grid : Grid object
    dt : time step (float)

    Returns
    -------
    None.

    """

    # Extract common parameters

    nx = grid.nx
    ny = grid.ny
    h = grid.h
    dt = dt
    m = 1 - water.available_volume_fraction

    # Save previous water and solute distributions

    water.previous_distribution = water.distribution
    solute.previous_distribution = solute.distribution
    cell.prev_x_velocity = cell.x_velocity

    # Use solvers to update each distribution in turn

    cell.distribution = cell_solver(cell, nx, ny, h, dt)

    water.distribution = water.available_volume_fraction - cell.distribution

    water.pressure = pressure_solver(water, h, ny, dt)

    water.x_velocity, water.y_velocity = water_velocity_solver(water, nx, ny, h)

    solute.distribution, bound_solute.distribution = vegf_solver(cell, water, solute, bound_solute, nx, ny, h, dt)

    cell.x_velocity = x_velocity_solver(cell, water, solute, bound_solute, ny, h)

    cell.y_velocity = y_velocity_solver(cell, water, solute, bound_solute, ny, h)

    cell.x_velocity = x_velocity_solver(cell, water, solute, bound_solute, ny, h)

    
    # check for volume conservation and consistency 
    if np.any(water.distribution < 0):
        raise ValueError('Negative volumes in water distribution')
    if np.any(cell.distribution < 0):
        raise ValueError('Negative volumes in cell distribution')
    assert (water.distribution + cell.distribution + m).all() == 1

    
    return

