import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# TODO: split the number of grid points into Nx and Ny in x and y directions

#------------------------------------------------------

# x velocity matrix solver 

#------------------------------------------------------


def Bx_matrix(j, n, gamma_n, m, ny, h):
    #diagonal block matrix for A

    diagonals = list(- (14 * n[1:ny-1, j] / 3) - (h**2 * gamma_n * m * n[1:ny-1, j]))

    below = list((4 * n[2:ny-1, j] / 3) - (1 / 3) * (n[3:ny, j] - n[1:ny-2, j]))
    above = list((4 * n[1:ny-2, j] / 3) + (1 / 3) * (n[2:ny-1, j] - n[:ny-3, j]))

    B = sparse.diags([diagonals, below, above], [0, -1, 1])

    return B

def D1x_matrix(j, n, ny):
    # off-diagonal (j-1) block matrix for A for j = 1, ... n-2

    diagonals = n[1:ny-1, j] - (1 / 4) * (n[1:ny-1, j+1] - n[1:ny-1, j-1])

    D1 =  sparse.diags([diagonals], [0])

    return D1

def D2x_matrix(j, n, ny):
    # off-diagonal (j+1) block matrix for A for j = 1, ... n-2

    diagonals = n[1:ny-1, j] + (1 / 4) * (n[1:ny-1, j+1] - n[1:ny-1, j-1])

    D2 =  sparse.diags([diagonals], [0])
        
    return D2


def x_velocity_solver(cell, water, solute, bound_solute, ny, h): 
    """
    
    Solves for the x velocity field 

    Parameters
    ----------
    cell : CellType dataclass
    n : cell distribution at current timestep t+dt (numpy array)
    un0 : horizontal cell velocity at time t (numpy array)
    vn0 : vertical cell velocity at time t (numpy array)
    c : solute concentration at time t+dt (numpy array)
    pw: water pressure (numpy array)
    chi: chemotaxis strength (float)
    gamma_n: cell-matrix drag coefficient (float)
    delta_a: contact inhibition coefficient (float)
    nu: cell aggregation coefficient (float)
    eta: cell-matrix traction coefficient (float)
    phi: available volume fraction (float)
    m: matrix volume fraction (float)
    Ny: no. of grid points in y direction (int)
    h: grid spacing in y direction (float)

    Returns
    -------
    un : x velocity field at current time t+dt (numpy array)
    
    """

    n = cell.distribution
    vn = cell.y_velocity
    c = solute.distribution
    cb = bound_solute.distribution
    pw = water.pressure
    chi = cell.chemotaxis_strength
    chi_b = cell.bound_chemotaxis_strength
    gamma_n = cell.matrix_drag
    delta_a = cell.contact_inhibition
    nu = cell.aggregation_strength
    eta = cell.matrix_traction
    phi = water.available_volume_fraction
    m = water.matrix_volume_fraction


    #-- Main matrix A
    ny2 = (ny-2) * (ny-2)

    A = sparse.lil_matrix((ny2, ny2))

    for i in range(0, ny-2): 
        A[i*(ny-2):(i+1)*(ny-2), i*(ny-2):(i+1)*(ny-2)] = Bx_matrix(i+1, n, gamma_n, m, ny, h)
    for i in range(1, ny-2):
        A[i*(ny-2):(i+1)*(ny-2), (i-1)*(ny-2):i*(ny-2)] = D1x_matrix(i+1, n, ny)
    for i in range(0, ny-3):
        A[i*(ny-2):(i+1)*(ny-2), (i+1)*(ny-2):(i+2)*(ny-2)] = D2x_matrix(i+1, n, ny)

    A = sparse.csr_matrix(A)

    #-- source terms

    F = np.zeros((ny-2, ny-2))
    
    F[:, :] = ((h * n[1:ny-1, 1:ny-1] / 2) * (pw[2:ny, 1:ny-1] - pw[:ny-2, 1:ny-1])) \
            + ((h * chi / 2) * (np.exp(-c[1:ny-1, 1:ny-1])) * ((n[2:ny, 1:ny-1] - n[:ny-2, 1:ny-1]) - n[1:ny-1, 1:ny-1] * (c[2:ny, 1:ny-1] - c[:ny-2, 1:ny-1]))) \
            + ((h * chi_b / 2) * (np.exp(-cb[1:ny-1, 1:ny-1])) * ((n[2:ny, 1:ny-1] - n[:ny-2, 1:ny-1]) - n[1:ny-1, 1:ny-1] * (cb[2:ny, 1:ny-1] - cb[:ny-2, 1:ny-1]))) \
            + ((h / 2) * (n[2:ny, 1:ny-1] - n[:ny-2, 1:ny-1])) * (((3 * delta_a * n[1:ny-1, 1:ny-1]**2)/(phi - n[1:ny-1, 1:ny-1])) - ((delta_a * n[1:ny-1, 1:ny-1]**3) / (phi - n[1:ny-1, 1:ny-1])**2) - (2 * nu * n[1:ny-1, 1:ny-1]) ) \
            + (h * eta * m / 2) * (n[2:ny, 1:ny-1] - n[:ny-2, 1:ny-1]) \
            - (n[1:ny-1, 1:ny-1] / 12) * (vn[2:ny, 2:ny] + vn[:ny-2, :ny-2] - vn[:ny-2, 2:ny] - vn[2:ny, :ny-2]) \
            + (1 / 6) * (vn[1:ny-1, 2:ny] - vn[1:ny-1, :ny-2]) * (n[2:ny, 1:ny-1] - n[:ny-2, 1:ny-1]) \
            - (1 / 4) * (vn[2:ny, 1:ny-1] - vn[:ny-2, 1:ny-1]) * (n[1:ny-1, 2:ny] - n[1:ny-1, :ny-2]) 


    F1d = F.flatten(order='F')

    #-- solve matrix equation for interior values of un

    un = spsolve(A, F1d)

    # reformat into 2D
    un2d = np.zeros((ny, ny))

    for i in range(1, ny-1):
        for j in range(1, ny-1):
            un2d[i, j] =  un[(j-1)*(ny-2) + (i-1)] 
    
    return un2d


# --------------------------------------------------------------------------------------------






#------------------------------------------------------

# y velocity matrix solver 

#------------------------------------------------------

def By_matrix(j, n, gamma_n, m, ny, h):
    #diagonal block matrix for A

    diagonals = list(- (14 * n[1:ny-1, j] / 3) - (h**2 * gamma_n * m * n[1:ny-1, j]))
    #below = list(n[1:ny-1, j] - (1 / 4) * (n[2:ny, j] - n[:ny-2, j]))
    #above = list(n[1:ny-1, j] + (1 / 4) * (n[2:ny, j] - n[:ny-2, j]))
    # correction 16/11/22
    below = list(n[2:ny-1, j] - (1 / 4) * (n[3:ny, j] - n[1:ny-2, j]))
    above = list(n[1:ny-2, j] + (1 / 4) * (n[2:ny-1, j] - n[:ny-3, j]))    


    B = sparse.diags([diagonals, below, above], [0, -1, 1])

    return B

def D1y_matrix(j, n, ny):
    # off-diagonal (j-1) block matrix for A for j = 1, ... n-2

    diagonals = (4 * n[1:ny-1, j] / 3) - (1 / 3) * (n[1:ny-1, j+1] - n[1:ny-1, j-1])

    D1 =  sparse.diags([diagonals], [0])

    return D1

def D2y_matrix(j, n, ny):
    # off-diagonal (j+1) block matrix for A for j = 1, ... n-2

    diagonals = (4 * n[1:ny-1, j] / 3) + (1 / 3) * (n[1:ny-1, j+1] - n[1:ny-1, j-1])

    D2 =  sparse.diags([diagonals], [0])
        
    return D2


def y_velocity_solver(cell, water, solute, bound_solute, ny, h): 
    """
    
    Solves for the y velocity field 

    Parameters
    ----------
    cell : CellType dataclass
    n : cell distribution at current timestep t+dt (numpy array)
    un0 : horizontal cell velocity at time t (numpy array)
    vn0 : vertical cell velocity at time t (numpy array)
    c : solute concentration at time t+dt (numpy array)
    pw: water pressure (numpy array)
    chi: chemotaxis strength (float)
    gamma_n: cell-matrix drag coefficient (float)
    delta_a: contact inhibition coefficient (float)
    nu: cell aggregation coefficient (float)
    eta: cell-matrix traction coefficient (float)
    phi: available volume fraction (float)
    m: matrix volume fraction (float)
    Ny: no. of grid points in y direction (int)
    h: grid spacing in y direction (float)

    Returns
    -------
    vn : y velocity field at current time t+dt (numpy array)
    
    """

    n = cell.distribution
    un = cell.x_velocity
    c = solute.distribution
    cb = bound_solute.distribution
    pw = water.pressure
    chi = cell.chemotaxis_strength
    chi_b = cell.bound_chemotaxis_strength
    gamma_n = cell.matrix_drag
    delta_a = cell.contact_inhibition
    nu = cell.aggregation_strength
    eta = cell.matrix_traction
    phi = water.available_volume_fraction
    m = water.matrix_volume_fraction


    #-- Main matrix A 
    ny2 = (ny-2) * (ny-2)

    #i and j index corresponds to i+1 and j+1 globally (e.g for variable n)

    A = sparse.lil_matrix((ny2, ny2))

    #-- populate main matrix A

    for i in range(0, ny-2): 
        A[i*(ny-2):(i+1)*(ny-2), i*(ny-2):(i+1)*(ny-2)] = By_matrix(i+1, n, gamma_n, m, ny, h)
    for i in range(1, ny-2):
        A[i*(ny-2):(i+1)*(ny-2), (i-1)*(ny-2):i*(ny-2)] = D1y_matrix(i+1, n, ny)
    for i in range(0, ny-3):
        A[i*(ny-2):(i+1)*(ny-2), (i+1)*(ny-2):(i+2)*(ny-2)] = D2y_matrix(i+1, n, ny)


    A = sparse.csr_matrix(A)

    #-- source terms

    F = np.zeros((ny-2, ny-2))
    
    F[:, :] = ((h * n[1:ny-1, 1:ny-1] / 2) * (pw[1:ny-1, 2:ny] - pw[1:ny-1, :ny-2])) \
            + ((h * chi / 2) * (np.exp(-c[1:ny-1, 1:ny-1])) * ((n[1:ny-1, 2:ny] - n[1:ny-1, :ny-2]) - n[1:ny-1, 1:ny-1] * (c[1:ny-1, 2:ny] - c[1:ny-1, :ny-2]))) \
            + ((h * chi_b / 2) * (np.exp(-cb[1:ny-1, 1:ny-1])) * ((n[1:ny-1, 2:ny] - n[1:ny-1, :ny-2]) - n[1:ny-1, 1:ny-1] * (cb[1:ny-1, 2:ny] - cb[1:ny-1, :ny-2]))) \
            + ((h / 2) * (n[1:ny-1, 2:ny] - n[1:ny-1, :ny-2])) * (((3 * delta_a * n[1:ny-1, 1:ny-1]**2)/(phi - n[1:ny-1, 1:ny-1])) - ((delta_a * n[1:ny-1, 1:ny-1]**3) / (phi - n[1:ny-1, 1:ny-1])**2) - (2 * nu * n[1:ny-1, 1:ny-1]) ) \
            + (h * eta * m / 2) * (n[1:ny-1, 2:ny] - n[1:ny-1, :ny-2]) \
            - (n[1:ny-1, 1:ny-1] / 12) * (un[2:ny, 2:ny] + un[:ny-2, :ny-2] - un[:ny-2, 2:ny] - un[2:ny, :ny-2]) \
            + (1 / 6) * (un[2:ny, 1:ny-1] - un[:ny-2, 1:ny-1]) * (n[1:ny-1, 2:ny] - n[1:ny-1, :ny-2]) \
            - (1 / 4) * (un[1:ny-1, 2:ny] - un[1:ny-1, :ny-2]) * (n[2:ny, 1:ny-1] - n[:ny-2, 1:ny-1]) 


    F1d = F.flatten(order='F')

    #-- solve matrix equation for interior values of un

    vn = spsolve(A, F1d)

    # reformat into 2D
    vn2d = np.zeros((ny, ny))

    for i in range(1, ny-1):
        for j in range(1, ny-1):
            vn2d[i, j] =  vn[(j-1)*(ny-2) + (i-1)] 
    
    return vn2d