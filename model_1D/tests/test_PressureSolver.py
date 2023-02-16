import numpy as np
from pytest import approx
from src.PressureSolver import pressure_solver
from src.ModelClasses import Water


def test_PressureSolver():
    """
    # Test solution against analytical solution 
    
    """

    #-- parameters
    nx = 1000
    dx = 1/ nx
    dx2 = dx*dx
    dt = 0.001

    gamma_w = 1
    m = 0.01

    #-- initial conditions and variables
    Wt = np.zeros((nx))
    for j in range(nx):
        Wt[j] = (j+1) * dx - 0.5


    nsteps = 1 / dt

    w = np.zeros((nx))
    w0 = np.zeros((nx))
    for j in range(nx):
        w[j] = (nsteps+1) * dt * ((j+1) * dx - 0.5) + 1
        w0[j] = (nsteps) * dt * ((j+1) * dx - 0.5) + 1

    water = Water(gamma_w, m)
    water.distribution = w
    water.previous_distribution = w0

    #-- solve numerically

    p = pressure_solver(water, m, nx, dx2, dt)

    #-- analytical solution for pressure gradient 

    Y = np.linspace(0, 1, nx)

    p_an = (gamma_w * m * Y * (Y - 1)) / (1 * (2 * (Y - 0.5) + 2))

    #-- check numerical against analytical solution 

    p_ngrad = (p[2:nx] - p[:nx-2]) / (2 * dx) 
    p_agrad = p_an[1:nx-1]

    assert np.shape(p_ngrad) == np.shape(p_agrad)
    assert [p_ngrad[i] == approx(p_agrad[i], rel = 1e-3) for i in range(len(p_ngrad))]

    return 
