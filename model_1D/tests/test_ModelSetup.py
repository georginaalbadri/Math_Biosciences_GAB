import numpy as np
from pytest import approx
from src.ModelSetup import Nondimensionalise, CalculateTimestep
from src.ModelClasses import Geometry, Grid, CellType, Water, Solute


def test_Geometry():
    """
    Test the Geometry class.
    """
    #-- set parameter values
    T = 1e5
    L = 1e-3

    #-- create geometry object
    geometry = Geometry(T = T, L = L)

    #-- check geometry parameters
    assert geometry.T == approx(T, rel=1e-3)
    assert geometry.L == approx(L, rel=1e-3)

    return


def test_Grid():
    """
    Test the Grid class.
    """
    #-- set parameter values
    nx = 500

    #-- create grid object
    grid = Grid(nx = nx)

    #-- check grid parameters
    assert grid.nx == nx
    assert grid.dx == approx(1/nx)

    return


def test_CellType():
    """
    Test the CellType class.
    """
    #-- set parameter values
    diffusion_rate = 1e-14
    initial_volume_fraction = 0.1

    #-- create cell type object
    cell_type = CellType(diffusion_rate = diffusion_rate, initial_volume_fraction = initial_volume_fraction)

    #-- check cell type parameters
    assert cell_type.diffusion_rate == approx(diffusion_rate, rel=1e-3)
    assert cell_type.initial_volume_fraction == approx(initial_volume_fraction, rel=1e-3)

    return


#TODO: test more dimensionless values
def test_Nondimensionalise():
    """
    Test the Nondimensionalise function.
    """
    #-- set parameter values
    T = 1e5
    L = 1e-3

    #-- create geometry object
    geometry = Geometry(T = T, L = L)

    #-- create cell type object
    cell = CellType(diffusion_rate = 1e-14, initial_volume_fraction = 0.1)
    water = Water(matrix_drag = 5e7, matrix_volume_fraction = 0.03, available_volume_fraction = 1 - 0.03)
    solute = Solute(diffusion_rate = 1e-14, initial_solute_concentration = 0.1)


    #-- nondimensionalise parameters
    Nondimensionalise(cell, solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4)

    #-- check nondimensionalised parameters
    assert cell.diffusion_rate == approx(1e-14*T/(L**2), rel=1e-3)

    return


def test_CalculateTimestep():
    """
    Test the CalculateTimestep function.
    """
    #-- set parameter values
    nx = 500
    dx = 1/nx
    D = 1e-14
    dt_multiplier = 100
    T = 1e5
    L = 1e-3
    D_nondim = D * T / L**2

    #-- create geometry 
    geometry = Geometry(T = T, L = L)

    #-- create grid
    grid = Grid(nx = nx, dx = dx)

    #-- create cell type object
    cell = CellType(diffusion_rate = D, initial_volume_fraction = 0.1)
    water = Water(matrix_drag = 5e7, matrix_volume_fraction = 0.03, available_volume_fraction = 1 - 0.03)
    solute = Solute(diffusion_rate = D, initial_solute_concentration = 0.1)

    #-- nondimensionalise parameters
    Nondimensionalise(cell, solute, water, geometry, cM = 1e-9, cell_viscosity = 1e4)

    #-- calculate timestep
    dt = CalculateTimestep(cell, solute, grid, dt_multiplier = dt_multiplier)

    #-- check timestep
    assert dt == approx(dt_multiplier*(dx**2)/(2*D_nondim), rel=1e-3)

    return