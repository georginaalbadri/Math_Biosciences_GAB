import numpy as np




def water_velocity_solver(water, Nx, Ny, h):
    """

    Solves for water velocity field using water pressure field

    Parameters
    ----------
    water : Water object
    p: water pressure field at current time step
    gamma_w: water-matrix drag
    m: matrix volume fraction
    Nx : number of cells in x direction
    Ny : number of cells in y direction
    h : grid spacing

    Returns
    -------
    uw : x water velocity field at current time step
    vw : y water velocity field at current time step

    """

    p = water.pressure
    gamma_w = water.matrix_drag
    m = water.matrix_volume_fraction

    # initialise velocity fields
    coeff = (1 / (gamma_w * m * 2 * h))

    uw = np.zeros((Nx, Ny))
    vw = np.zeros((Nx, Ny))

    vw[:Nx, 0] = 0
    vw[:Nx, 1:-1] = - coeff * (p[:, 2:] - p[:, :-2])
    vw[:Nx, Ny-1] = 0

    uw[0, :] = 0
    uw[1:Nx-1, :] = - coeff * (p[2:, :] - p[:-2, :])
    uw[Nx-1, :] = 0

    return uw, vw