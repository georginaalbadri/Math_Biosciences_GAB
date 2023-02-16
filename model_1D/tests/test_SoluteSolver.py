import numpy as np
from pytest import approx
from src.SoluteSolver import solute_solver_no_flux
from src.ModelClasses import CellType, Water, Solute


#TODO: test for solute_solver_diffusive_flux


def test_SoluteSolver():

    """
    # Test conservation of solute solver (with no production or degradation)
    
    """

    #--  parameter values
    nx = 600
    dx = 1/nx
    dx2 = dx*dx
    dt = 0.00001

    alpha = 0
    delta = 0
    phi = 0.97

    Dc = 0.01 * np.ones((nx))
    kappa = 0
    K = 0.1

    #-- initial conditions, variables 
    c = np.random.rand(nx)
    n = 0.01 * np.ones((nx))
    w = 0.99 * np.ones((nx))
    vw = np.random.rand((nx))
    w0 = w
    c0 = c

    #-- create cell, water and solute dataclasses
    cell = CellType()
    cell.distribution = n
    water = Water()
    water.distribution = w
    water.previous_distribution = w0
    water.velocity = vw
    solute = Solute(diffusion_rate=Dc, degradation_rate=delta, production_rate=alpha, uptake_rate=kappa, uptake_constant=K)
    solute.distribution = c
    solute.previous_distribution = c0

    #-- calculate total initial VEGF

    cint = np.sum(c0*w0)

    #-- solve numerically using matrix solver

    nsteps = 100

    for t in range(nsteps):
        solute.distribution = solute_solver_no_flux(solute=solute, cell=cell, water=water, phi=phi, nx=nx, dx=dx, dt=dt)

    #-- calculate total final VEGF

    cfin = np.sum(c*w) 

    #-- check conservation

    assert cfin == approx(cint, rel = 1e-3)

    return