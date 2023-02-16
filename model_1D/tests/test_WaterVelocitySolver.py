import numpy as np
from pytest import approx
from src.WaterVelocitySolver import water_velocity_solver
from src.ModelClasses import Water



def test_WaterVelocitySolver():
    """
    Test water velocity solver
    """

    # parameters
    nx = 100
    dx = 1 / nx
    gamma_w = 1
    m = 0.01

    # pressure distribution
    pw = np.random.rand((nx))

    water = Water(gamma_w, m)
    water.pressure = pw

    # velocity distribution
    vw = water_velocity_solver(water, m, nx, dx)

    # check velocity distribution
    assert len(vw) == nx
    assert [vw[i] == approx(pw[i+1] - pw[i], rel = 1e-3) for i in range(nx-1)]

    return
