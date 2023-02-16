import numpy as np
from pytest import approx
from src.CellSolver import cell_solver
from src.ModelClasses import CellType




def test_CellSolver():

    """
    # Test diffusion solution against absolute (analytical) solution
    
    """

    #--  set parameter values
    nx = 200
    dx = 1/nx
    dx2 = dx*dx
    dt = 0.001
    D = 1e-4

    #-- set initial cell distribution, zero cell velocity distribution 

    vn = np.zeros((nx))

    n = np.ones((nx))
    for j in range(nx):
        n[j] = np.cos(np.pi*j*dx)

    cell = CellType(diffusion_rate = D)
    cell.distribution = n
    cell.velocity = vn

    #-- calculate numerical solution
    nsteps = int(1/dt)
    for t in range(nsteps):
        cell.distribution = cell_solver(cell, nx, dx, dt)
    
    numerical_solution = cell.distribution

    #-- calculate analytical solution (using separable variables)
    
    at = np.exp(- D * (np.pi) * (np.pi) * nsteps * dt)

    ax = np.empty((nx))
    for j in range(0, nx):
        ax[j] = np.cos(j * dx * (np.pi))

    analytical_solution = at * ax

    #-- check numerical against analytical solution 
    for i in range(nx):
        assert numerical_solution[i] == approx(analytical_solution[i], rel=1e-3)
    
    return 