import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



# ------------------------------------------------------------------------

def cell_solver(cell, nx, dx, dt): 
    """
    
    Solves a 1D cell advection-diffusion equation using finite differences with the given parameters.

    Parameters
    ----------
    cell : CellType dataclass
    n0 : cell distribution at previous timestep t (numpy array)
    vn : cell velocity (numpy array)
    D : cell diffusion rate (float)
    nx : number of grid points (int)
    dx : grid spacing (float)
    dt : time step (float)
    
    Returns
    -------
    n : cell distribution at time t+dt (numpy array)    

    """
    n0 = cell.distribution
    vn = cell.velocity
    D = cell.diffusion_rate

    #-- matrix A (to solve An = B)

    diagonals = list((2 * D) + (dx / 2) * (vn[2:nx] - vn[0:nx-2]) + (dx**2 / dt))
    diagonals.insert(0, (2 * D) +  (dx / 2) * (-3*vn[0] + 4*vn[1] - vn[2]) + (dx**2 / dt))
    diagonals.append((2 * D) + (dx / 2) * (3*vn[nx-1] - 4*vn[nx-2] + vn[nx-3]) + (dx**2 / dt))

    below = list(- ((dx * vn[1:nx-1]) / 2) - D)
    below.append(- 2 * D)

    above = list(((dx * vn[1:nx-1]) / 2) - D)
    above.insert(0,  - 2 * D)

    A = sparse.diags([diagonals, below, above], [0, -1, 1])

    A = sparse.csr_matrix(A)

    #-- source terms B (An = B)

    B = np.zeros((nx))
    
    B[:nx] =  (dx**2 * n0[:nx] / dt)

    #-- solve matrix equation (An = B) for n

    n = spsolve(A, B)
    
    return n

# ------------------------------------------------------------------------ 
