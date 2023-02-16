import numpy as np
from src.CellSolver import cell_solver
from src.PressureSolver import pressure_solver
from src.WaterVelocitySolver import water_velocity_solver
from src.SoluteSolver import solute_solver_no_flux, bound_solute_solver
from src.CellVelocitySolver import cell_velocity_solver



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
    dx = grid.dx
    dt = dt
    phi = water.available_volume_fraction
    m = 1 - water.available_volume_fraction

    # Save previous water and solute distributions
    water.previous_distribution = water.distribution
    solute.previous_distribution = solute.distribution

    # Use solvers to update each distribution in turn

    cell.distribution = cell_solver(cell, nx, dx, dt)

    water.distribution = water.available_volume_fraction - cell.distribution

    water.pressure = pressure_solver(water, m, nx, dx, dt)

    water.velocity = water_velocity_solver(water, m, nx, dx)

    solute.distribution = solute_solver_no_flux(solute, bound_solute, cell, water, phi, nx, dx, dt)

    bound_solute.distribution = bound_solute_solver(bound_solute, solute, cell, water, nx, dt)

    cell.velocity = cell_velocity_solver(cell, solute, bound_solute, water, phi, m, nx, dx)

    
    # check for volume conservation and consistency 
    if np.any(water.distribution < 0):
        #print("SA parameters: " + str(solute.uptake_rate) + ", " + str(cell.matrix_traction) + ", " + str(cell.aggregation_strength) + ", " + str(cell.contact_inhibition))
        #print("Sweep parameters: " + str(solute.chemotaxis_strength) + ", " + str(solute.chemotaxis_strength_bound) + ", " + str(cell.matrix_drag) + ", " + str(solute.production_rate) + ", " + str(solute.degradation_rate))
        raise ValueError('Negative volumes in water distribution')
    if np.any(solute.distribution < 0):
        #print("SA parameters: " + str(solute.uptake_rate) + ", " + str(cell.matrix_traction) + ", " + str(cell.aggregation_strength) + ", " + str(cell.contact_inhibition))
        #print("Sweep parameters: " + str(solute.chemotaxis_strength) + ", " + str(solute.chemotaxis_strength_bound) + ", " + str(cell.matrix_drag) + ", " + str(solute.production_rate) + ", " + str(solute.degradation_rate))
        raise ValueError('Negative volumes in solute distribution')
    if np.any(bound_solute.distribution < 0):
        #print("SA parameters: " + str(solute.uptake_rate) + ", " + str(cell.matrix_traction) + ", " + str(cell.aggregation_strength) + ", " + str(cell.contact_inhibition))
        #print("Sweep parameters: " + str(solute.chemotaxis_strength) + ", " + str(solute.chemotaxis_strength_bound) + ", " + str(cell.matrix_drag) + ", " + str(solute.production_rate) + ", " + str(solute.degradation_rate))
        raise ValueError('Negative volumes in bound_solute distribution')
    if np.any(cell.distribution < 0):
        #print("SA parameters: " + str(solute.uptake_rate) + ", " + str(cell.matrix_traction) + ", " + str(cell.aggregation_strength) + ", " + str(cell.contact_inhibition))
        #print("Sweep parameters: " + str(solute.chemotaxis_strength) + ", " + str(solute.chemotaxis_strength_bound) + ", " + str(cell.matrix_drag) + ", " + str(solute.production_rate) + ", " + str(solute.degradation_rate))
        raise ValueError('Negative volumes in cell distribution')
    assert (water.distribution + cell.distribution + m).all() == 1

    
    return

