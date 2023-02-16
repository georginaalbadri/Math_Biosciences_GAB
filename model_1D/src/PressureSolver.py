import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



# ----------------------------------------------------------------------------------------------

def pressure_solver(water, m, nx, dx, dt): 
    """
    Solve pressure distribution 

    Parameters
    ----------
    water : Water dataclass
    w : current water distribution (numpy array)
    w0 : previous water distribution (numpy array)
    gamma_w : water-matrix drag
    m : matrix volume fraction
    nx : number of grid points
    dx : grid spacing
    dt : time step

    Returns
    -------
    pw : pressure distribution (numpy array)
    
    """
    w = water.distribution
    w0 = water.previous_distribution
    gamma_w = water.matrix_drag

    #-- matrix A (Ap = B)

    
    diagonals = list(- 2 * w[:nx-1])
    below = list(- ((w[2:nx] - w[:nx-2])/4) + w[1:nx-1])
    above = list(((w[2:nx-1] - w[:nx-3])/4) + w[1:nx-2])
    above.insert(0, 2*w[0])



    A = sparse.diags([diagonals, below, above], [0, -1, 1])

    A = sparse.csr_matrix(A)

    #-- source terms B

    B = np.zeros((nx-1))

    B[:nx] =  (gamma_w * m * dx**2) * (w[:nx-1] - w0[:nx-1]) / dt

    #-- solve matrix equation for pw 

    pw = np.zeros((nx))

    pw[:nx-1] = spsolve(A, B)
    
    return pw 

#----------------------------------------------------------------------------------------------
