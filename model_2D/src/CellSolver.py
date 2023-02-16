import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ------------------------------------------------------------------------

def B_matrix(j, un, vn, D, Nx, h, dt):
    # Nx = number of grid points in x direction
    # Ny = number of grid points in y direction
    # h = grid spacing
    # dt = time step
    # D = diffusion coefficient

    coeff = h / 2
    const = (h**2 / dt) + (4 * D)

    # n_{i0} terms
    diagonals = list(const + coeff * (un[2:Nx, j] - un[:Nx-2, j]) + coeff * (vn[1:Nx-1, j+1] - vn[1:Nx-1, j-1]))
    diagonals.insert(0, const + coeff * (-3*un[0, j] + 4*un[1, j] - un[2, j]) + coeff * (vn[0, j+1] - vn[0, j-1]))   
    diagonals.append(const + coeff * (3*un[Nx-1, j] - 4*un[Nx-2, j] + un[Nx-3, j]) + coeff * (vn[Nx-1, j+1] - vn[Nx-1, j-1]))  

    # n_{i-1,j} terms
    below = list(- (coeff * un[1:Nx-1, j]) - D)
    below.append(- 2 * D)   # no flux boundary conditions

    # n_{i+1,j} terms
    above = list( (coeff * un[1:Nx-1, j]) - D)
    above.insert(0, - 2 * D)   # no flux boundary conditions

    B = sparse.diags([above, diagonals, below], [1, 0, -1])

    return B

# ------------------------------------------------------------------------

# n_{i,0} terms
def B0_matrix(un, vn, D, Nx, h, dt):
    
    coeff = h / 2
    const = (h**2 / dt) + (4 * D)

    # n_{ij} terms
    diagonals = list(const + coeff * (un[2:Nx, 0] - un[:Nx-2, 0]) + coeff * (-3*vn[1:Nx-1, 0] + 4*vn[1:Nx-1, 1] - vn[1:Nx-1, 2]))
    diagonals.insert(0, const + coeff * (-3*un[0, 0] + 4*un[1, 0] - un[2, 0]) + coeff * (-3*vn[0, 0] + 4*vn[0, 1] - vn[0, 2]))  
    diagonals.append(const + coeff * (3*un[Nx-1, 0] - 4*un[Nx-2, 0] + un[Nx-3, 0]) + coeff * (-3*vn[Nx-1, 0] + 4*vn[Nx-1, 1] - vn[Nx-1, 2]))  

    # n_{i-1,j} terms
    below = list(- (coeff * un[1:Nx-1, 0]) - D)
    below.append(- 2 * D)   # no flux boundary conditions

    # n_{i+1,j} terms
    above = list( (coeff * un[1:Nx-1, 0]) - D)
    above.insert(0, - 2 * D)  # no flux boundary conditions

    B0 = sparse.diags([above, diagonals, below], [1, 0, -1])

    return B0

# ------------------------------------------------------------------------

# n_{i,Ny-1} terms
def BN_matrix(un, vn, D, Nx, Ny, h, dt):
    
    coeff = h / 2
    const = (h**2 / dt) + (4 * D)

    # n_{ij} terms
    diagonals = list(const + coeff * (un[2:Nx, Ny-1] - un[:Nx-2, Ny-1]) + coeff * (3*vn[1:Nx-1, Ny-1] - 4*vn[1:Nx-1, Ny-2] + vn[1:Nx-1, Ny-3]))
    diagonals.insert(0, const + coeff * (-3*un[0, Ny-1] + 4*un[1, Ny-1] - un[2, Ny-1]) + coeff * (3*vn[0, Ny-1] - 4*vn[0, Ny-2] + vn[0, Ny-3]))   
    diagonals.append(const + coeff * (3*un[Nx-1, Ny-1] - 4*un[Nx-2, Ny-1] + un[Nx-3, Ny-1]) + coeff * (3*vn[Nx-1, Ny-1] - 4*vn[Nx-1, Ny-2] + vn[Nx-1, Ny-3]))   

    # n_{i-1,j} terms
    below = list(- (coeff * un[1:Nx-1, Ny-1]) - D)
    below.append(- 2 * D)   # no flux boundary conditions

    # n_{i+1,j} terms
    above = list( (coeff * un[1:Nx-1, Ny-1]) - D)
    above.insert(0, - 2 * D)   # no flux boundary conditions

    BN = sparse.diags([above, diagonals, below], [1, 0, -1])

    return BN

# ------------------------------------------------------------------------

# n_{ij-1} terms
def D1_matrix(j, vn, D, Nx, h):
    # off-diagonal block matrix for A for j = 1, ... n-2

    coeff = h / 2
 
    diagonals = - (coeff * vn[:Nx, j]) - D
    
    D1 = sparse.diags([diagonals], [0])

    return D1

# ------------------------------------------------------------------------

# n_{ij+1} terms
def D2_matrix(j, vn, D, Nx, h):
    # off-diagonal block matrix for A for j=1, ... n-2 

    coeff = h / 2

    diagonals = (coeff * vn[:Nx, j]) - D
    
    D2 = sparse.diags([diagonals], [0])

    return D2

# ------------------------------------------------------------------------



# ------------------------------------------------------------------------

def cell_solver(cell, Nx, Ny, h, dt):
    """
    
    Solves a 2D cell advection-diffusion equation using finite differences with the given parameters.

    Parameters
    ----------
    cell : CellType dataclass
    n0 : cell distribution at previous timestep t (numpy array)
    un : horizontal cell velocity (numpy array)
    vn : vertical cell velocity (numpy array)
    D : cell diffusion rate (float)
    Nx : number of grid points in x (int)
    Ny : number of grid points in y (int)
    h : grid spacing (float)
    dt : time step (float)

    Returns
    -------
    n : cell distribution at current time t+dt (numpy array)    

    """
    
    n0 = cell.distribution
    un = cell.x_velocity
    vn = cell.y_velocity
    D = cell.diffusion_rate

    #-- matrix A (to solve An = B)

    # create block matrix A

    A = sparse.lil_matrix((Nx*Ny, Nx*Ny))

    # fill in the diagonal blocks
    A[0:Nx, 0:Nx] = B0_matrix(un, vn, D, Nx, h, dt)
    A[0:Nx, Nx:2*Nx] = -2 * D * np.identity(Nx)

    for i in range(1, Ny-1):   
        A[i*Nx:(i+1)*Nx, i*Nx:(i+1)*Nx] = B_matrix(i, un, vn, D, Nx, h, dt)
        A[i*Nx:(i+1)*Nx, (i-1)*Nx:i*Nx] = D1_matrix(i, vn, D, Nx, h)
        A[i*Nx:(i+1)*Nx, (i+1)*Nx:(i+2)*Nx] = D2_matrix(i, vn, D, Nx, h)

    A[(Ny-1)*Nx:Ny*Nx, (Ny-1)*Nx:Ny*Nx] = BN_matrix(un, vn, D, Nx, Ny, h, dt)
    A[(Ny-1)*Nx:Ny*Nx, (Ny-2)*Nx:(Ny-1)*Nx] = - 2 * D * np.identity(Nx)

    # convert to CSR format
    A = A.tocsr()

    # create RHS vector
    b = h**2 * n0.flatten(order='F') / dt

    # solve linear system
    n = spsolve(A, b)

    return n.reshape((Nx, Ny), order='F')

# ------------------------------------------------------------------------
