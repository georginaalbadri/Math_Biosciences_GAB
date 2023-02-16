import numpy as np
from pytest import approx
from src.CellVelocitySolver import cell_velocity_solver
from src.ModelClasses import CellType, Water, Solute


def test_CellVelocitySolver():

    """
    # Test solution against analytical solution (for negligible drag regime)
    
    """

    #--  parameter values
    nx = 2000
    dx = 1/nx
    dx2 = dx*dx

    gamma_n = 0
    chi = 10
    m = 0.05
    nu = 0
    delta_a = 0
    eta = 0
    phi = 1 - m

    #-- solve numerically

    n = np.zeros((nx))
    for j in range(nx):
        n[j] = 0.01 + ((0.02 - 0.01) / (nx-1)) *(j)

    c = np.zeros((nx))
    for j in range(nx):
        c[j] = 0 + ((2 - 0) / (nx-1)) *(j)

    pw = np.zeros((nx))

    # make dataclasses 
    cell = CellType(chemotaxis_strength=chi, matrix_drag=gamma_n, matrix_traction=0, aggregation_strength=0, contact_inhibition=0)
    cell.distribution = n
    water = Water(matrix_volume_fraction=m, available_volume_fraction=phi)
    water.pressure = pw
    solute = Solute()
    solute.distribution = c

    vn = cell_velocity_solver(cell, solute, water, phi, m, nx, dx)

    #-- plot analytical solution

    d = 0.01
    e = 0.01
    a = 0
    b = 2

    alpha = (3 * chi * np.exp(-a)) / (4 * b)

    Y = np.linspace(0, 1, nx)

    v_an = - alpha * np.exp (- b * Y) + (alpha * ( np.exp(-b) - 1) * np.log(d + e * Y)) / ( np.log( (d + e) / d ) ) \
        + alpha - ((alpha * (np.exp(-b) - 1) * np.log(d)) / ( np.log( (d+e) / d) ))

    
    #-- check numerical against analytical solution 

    assert [vn[i] == approx(v_an[i], rel = 1e-3) for i in range(nx)]


    return