import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve



#------------------------------------------------------------------------------------

def cell_velocity_solver(cell, solute, bound_solute,  water, phi, m, nx, dx): 

    """
    Solve cell velocity distribution

    Parameters
    ----------
    cell : Cell dataclass
    solute : Solute dataclass
    water : Water dataclass
    n : current cell distribution (numpy array)
    c : current solute distribution (numpy array)
    pw : current water pressure distribution (numpy array)
    chi: chemotaxis strength (float)
    gamma_n : cell-matrix drag (float)
    nu : aggregation strength (float)
    delta_a : contact inhibition (float)
    phi : available volume fraction (float)
    m : matrix volume fraction (float)
    nx : number of grid points (int)
    dx : grid spacing (float)

    Returns
    -------
    vn : cell velocity distribution (numpy array)

    """

    n = cell.distribution
    c = solute.distribution
    cb = bound_solute.distribution 
    pw = water.pressure
    chi = cell.chemotaxis_strength
    chib = cell.chemotaxis_strength_bound
    gamma_n = cell.matrix_drag
    nu = cell.aggregation_strength
    delta_a = cell.contact_inhibition
    eta = cell.matrix_traction

    #-- matrix A (Av = B)

    dx2 = dx**2
    factorA = 1 / (3 * dx2)

    diagonals = list(- (8 * n[2:nx-2]) - (3 * dx2 * gamma_n * m * n[2:nx-2]))
    diagonals.insert(0, - (8 * n[1]) - (3 * dx2 * gamma_n * m * n[1]))
    diagonals.append(- (8 * n[nx-2]) - (3 * dx2 * gamma_n * m * n[nx-2]))

    below = list(4 * n[2:nx-2] - (n[3:nx-1] - n[1:nx-3]))
    below.append(4 * n[nx-2] - (n[nx-1] - n[nx-3]))

    above = list(4 * n[2:nx-2] + (n[3:nx-1] - n[1:nx-3]))
    above.insert(0, 4 * n[1] + (n[2] - n[0]))

    A = sparse.diags([diagonals, below, above], [0, -1, 1])

    An = factorA * sparse.csr_matrix(A)

    #-- source terms B

    B = np.zeros((nx-2))

    factorB = 1 / (2 * dx)

    B[:nx-2] = (n[1:nx-1] * (pw[2:nx] - pw[:nx-2])) \
            - chi * (np.exp(-c[1:nx-1]) * ((n[1:nx-1] * (c[2:nx] - c[:nx-2])) - (n[2:nx] - n[:nx-2])) ) \
            - chib * (np.exp(-cb[1:nx-1]) * ((n[1:nx-1] * (cb[2:nx] - cb[:nx-2])) - (n[2:nx] - n[:nx-2])) ) \
            + (n[2:nx] - n[:nx-2]) * (- (2 * nu * m) + ((3 * delta_a * n[1:nx-1]**2) / (phi - n[1:nx-1])) 
            + ( (delta_a * n[1:nx-1]**3) / ( (phi - n[1:nx-1])**2 ) ) + eta * m) 

    Bn = factorB * B

    #-- solve matrix equation for interior values of vn 

    vn = np.zeros((nx))

    vn[1:-1] = spsolve(An, Bn)

    #boundary conditions
    vn[0] = 0
    vn[nx-1] = 0

    
    return vn
# -----------------------------------------------------------------------------------   



