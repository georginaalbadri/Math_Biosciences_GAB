import numpy as np



#----------------------------------------------------------------------------------------------------------------------

def Nondimensionalise(cell, solute, bound_solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4):
    """
    Nondimensionalises the model parameter values and overwrites dimensional values in the respective dataclasses
    """

    cell.diffusion_rate = cell.diffusion_rate * geometry.T / (geometry.L**2)
    solute.diffusion_rate = solute.diffusion_rate * geometry.T / (geometry.L**2)

    solute.production_rate = solute.production_rate * geometry.T / cM
    solute.uptake_rate = solute.uptake_rate * geometry.T / cM
    solute.uptake_constant = solute.uptake_constant / cM
    solute.degradation_rate = solute.degradation_rate * geometry.T 
    solute.binding_rate = solute.binding_rate * geometry.T 

    bound_solute.unbinding_rate = bound_solute.unbinding_rate * geometry.T 
    bound_solute.degradation_rate = bound_solute.degradation_rate * geometry.T

    cell.chemotaxis_strength = cell.chemotaxis_strength * geometry.T / cell_viscosity
    cell.aggregation_strength = cell.aggregation_strength * geometry.T / cell_viscosity
    cell.contact_inhibition = cell.contact_inhibition * geometry.T / cell_viscosity
    cell.matrix_traction = cell.matrix_traction * geometry.T / cell_viscosity

    cell.matrix_drag = cell.matrix_drag * geometry.L**2
    water.matrix_drag = water.matrix_drag * geometry.L**2

    return

#----------------------------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------------------------

def CalculateTimestep(cell, solute, grid, dt_multiplier = 100):
    """
    Calculates the timestep for the cell solver.
    """

    D_cell = cell.diffusion_rate
    D_solute = solute.diffusion_rate
    dx = grid.dx

    D = max(D_cell, D_solute)

    maxtimestep = (dx**2 / (2 * D))
    dt = dt_multiplier * maxtimestep

    return dt

#----------------------------------------------------------------------------------------------------------------------