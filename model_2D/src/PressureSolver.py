import numpy as np
from scipy import sparse



# ------------------------------------------------------------------------

def B_matrix(j, w, ny): 

    diagonals = 4 * w[0:ny, j]
    below = list(+ ((w[2:ny, j] - w[:ny-2, j])/4) - w[1:ny-1, j])
    above = list(- ((w[2:ny, j] - w[:ny-2, j])/4) - w[1:ny-1, j])
    above.insert(0, - 2 * w[0, j])
    below.append(- 2*w[ny-1, j])

    B = sparse.diags([diagonals, below, above], [0, -1, 1])

    return B

# ------------------------------------------------------------------------

# j+1
def D1_matrix(j, w, ny):

    diagonals = - w[:ny, j] - ((w[:ny, j+1] - w[:ny, j-1]) / 4) 
    D1 = sparse.diags([diagonals], [0])

    return D1

# ------------------------------------------------------------------------

# j-1 
def D2_matrix(j, w, ny):

    diagonals = - w[:ny, j] + ((w[:ny, j+1] - w[:ny, j-1]) / 4) 
    D2 = sparse.diags([diagonals], [0])

    return D2

# ------------------------------------------------------------------------

def D0_matrix(w, ny):
        
    diagonals = - 2 * w[:ny, 0]
    D0 = sparse.diags([diagonals], [0])

    return D0

# ------------------------------------------------------------------------

def DN_matrix(w, ny):
        
    diagonals = - 2 * w[:ny, ny-1]
    DN = sparse.diags([diagonals], [0])

    return DN

# ------------------------------------------------------------------------


# ------------------------------------------------------------------------

def pressure_solver(water, h, ny, dt):
    """
    
    Solves for the water pressure field

    Parameters
    ----------
    water : Water object
    w: water velocity field at current time step
    w0: water velocity field at previous time step
    gamma_w: water-matrix drag
    m: matrix volume fraction
    
    Returns
    -------
    pw: water pressure field at current time step
    
    """

    w = water.distribution
    w0 = water.previous_distribution
    gamma_w = water.matrix_drag
    m = water.matrix_volume_fraction


    # create matrix A

    ny2 = (ny) * (ny)
    A = sparse.lil_matrix((ny2, ny2))

    A[0:ny, 0:ny] = B_matrix(0, w, ny)
    A[0:ny, ny:2*ny] = D0_matrix(w, ny)

    for i in range(1, ny-1):
        for j in range(0, ny-1):
            if i==j:
                A[i*ny:(i+1)*ny, j*ny:(j+1)*ny] = B_matrix(i, w, ny)
            if j == i+1:
                A[i*ny:(i+1)*ny, j*ny:(j+1)*ny] = D1_matrix(i, w, ny)
            if j == i-1:
                A[i*ny:(i+1)*ny, j*ny:(j+1)*ny] = D2_matrix(i, w, ny)

    A[(ny-1)*ny:ny2,(ny-1)*ny:ny2] = B_matrix(ny-1, w, ny)
    A[(ny-1)*ny:ny2,(ny-2)*ny:(ny-1)*ny] = DN_matrix(w, ny)
    
    # convert to csr format
    A = sparse.csr_matrix(A)

    # source terms 
    E = ((gamma_w * m * h**2) / dt) * (w0 - w)

    E1d = E.flatten(order='F')

    pw, exit_code, _, _, _, _, _, _ = sparse.linalg.lsmr(A, E1d, maxiter = 10000)
    if exit_code == 7:
        ValueError('LSMR requires more iterations for pressure solver')

    return pw.reshape((ny, ny), order='F')