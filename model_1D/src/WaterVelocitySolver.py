import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def water_velocity_solver(water, m, nx, dx):
    """
    Solve water velocity distribution

    Parameters
    ----------
    water: Water dataclass
    pw : pressure distribution (numpy array)
    gamma_w : water-matrix drag
    m : matrix volume fraction
    nx : number of grid points
    dx : grid spacing

    Returns
    -------
    vw : water velocity distribution (numpy array)

    """
    gamma_w = water.matrix_drag
    pw = water.pressure

    vw = np.zeros((nx))
    vw[0] = - (1 / (gamma_w * m * 2 * dx)) * (-3*pw[0] + 4*pw[1] - pw[2])
    vw[1:nx-2] = - (1 / (gamma_w * m * 2 * dx)) * (pw[2:nx-1] - pw[:nx-3])
    vw[nx-1] = - (1 / (gamma_w * m * 2 * dx)) * (3*pw[nx-1] - 4*pw[nx-2] + pw[nx-3])

    return vw