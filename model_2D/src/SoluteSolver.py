import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ------------------------------------------------------------------------

def B_matrix(j, uw, vw, w, w0, n, D, delta, kb, Nx, h, dt):

    coeff1 = (h**2)/dt
    coeff2 = h / 2

    # c{ij} terms
    diagonals = list(coeff1 * (w[1:Nx-1, j] - w0[1:Nx-1, j]) + coeff1 * w[1:Nx-1, j] + coeff2 * w[1:Nx-1, j] * (uw[2:Nx, j] - uw[:Nx-2, j]) + coeff2 * w[1:Nx-1, j] * (vw[1:Nx-1, j+1] - vw[1:Nx-1, j-1]) \
        + coeff2 * uw[1:Nx-1, j] * (w[2:Nx, j] - w[:Nx-2, j]) + coeff2 * vw[1:Nx-1, j] * (w[1:Nx-1, j+1] - w[1:Nx-1, j-1]) \
        + 4 * D * w[1:Nx-1, j] + h**2 * delta + h**2 * kb * n[1:Nx-1, j] * w[1:Nx-1, j])
    diagonals.insert(0, coeff1 * (w[0, j] - w0[0, j]) + coeff1 * w[0, j] + coeff2 * w[0, j] * (-3*uw[0, j] + 4*uw[1, j] - uw[2, j]) + coeff2 * w[0, j] * (vw[0, j+1] - vw[0, j-1]) \
        + coeff2 * uw[0, j] * (-3*w[0, j] + 4*w[1, j] - w[2, j]) + coeff2 * vw[0, j] * (w[0, j+1] - w[0, j-1]) \
        + 4 * D * w[0, j] + h**2 * delta + h**2 * kb * n[0, j] * w[0, j])
    diagonals.append(coeff1*(w[Nx-1, j] - w0[Nx-1, j]) + coeff1 * w[Nx-1, j] + coeff2 * w[Nx-1, j] * (3*uw[Nx-1, j] - 4*uw[Nx-2, j] + uw[Nx-3, j]) + coeff2 * w[Nx-1, j] * (vw[Nx-1, j+1] - vw[Nx-1, j-1]) \
        + coeff2 * uw[Nx-1, j] * (3*w[Nx-1, j] - 4*w[Nx-2, j] + w[Nx-3, j]) + coeff2 * vw[Nx-1, j] * (w[Nx-1, j+1] - w[Nx-1, j-1]) \
        + 4 * D * w[Nx-1, j] + h**2 * delta + h**2 * kb * n[Nx-1, j] * w[Nx-1, j])

    # c{i-1,j} terms
    below = list(- (coeff2 * w[1:Nx-1, j] * uw[1:Nx-1, j]) - D * w[1:Nx-1, j] + (D / 4) * (w[2:Nx, j] - w[:Nx-2, j]))
    below.append(- 2 * D * w[Nx-1, j])    # no flux bouwdary conditions

    # c{i+1,j} terms
    above = list( (coeff2 * w[1:Nx-1, j] * uw[1:Nx-1, j]) - D * w[1:Nx-1, j] - (D / 4) * (w[2:Nx, j] - w[:Nx-2, j]))
    above.insert(0, - 2 * D * w[0, j])     # no flux bouwdary conditions

    B = sparse.diags([above, diagonals, below], [1, 0, -1])

    return B

# ------------------------------------------------------------------------

# c{i,0} terms
def B0_matrix(uw, vw, w, w0, n, D, delta, kb, Nx, h, dt):
    
    coeff1 = h**2/dt
    coeff2 = h / 2
    
    # c{i0} terms
    diagonals = list(coeff1 * (w[1:Nx-1, 0] - w0[1:Nx-1, 0]) + coeff1 * w[1:Nx-1, 0] + coeff2 * w[1:Nx-1, 0] * (uw[2:Nx, 0] - uw[:Nx-2, 0]) + coeff2 * w[1:Nx-1, 0] * (-3*vw[1:Nx-1, 0] + 4*vw[1:Nx-1, 1] - vw[1:Nx-1, 2]) \
        + coeff2 * uw[1:Nx-1, 0] * (w[2:Nx, 0] - w[:Nx-2, 0]) + coeff2 * vw[1:Nx-1, 0] * (-3*w[1:Nx-1, 0] + 4*w[1:Nx-1, 1] - w[1:Nx-1, 2]) \
        + 4 * D * w[1:Nx-1, 0] + h**2 * delta + h**2 * kb * n[1:Nx-1, 0] * w[1:Nx-1, 0])
    diagonals.insert(0, coeff1 * (w[0, 0] - w0[0, 0]) + coeff1 * w[0, 0] + coeff2 * w[0, 0] * (-3*uw[0, 0] + 4*uw[1, 0] - uw[2, 0]) + coeff2 * w[0, 0] * (-3*vw[0, 0] + 4*vw[0, 1] - vw[0, 2]) \
        + coeff2 * uw[0, 0] * (-3*w[0, 0] + 4*w[1, 0] - w[2, 0]) + coeff2 * vw[0, 0] * (-3*w[0, 0] + 4*w[0, 1] - w[0, 2]) \
        + 4 * D * w[0, 0] + h**2 * delta + h**2 * kb * n[0, 0] * w[0, 0])
    diagonals.append(coeff1 * (w[Nx-1, 0] - w0[Nx-1, 0]) + coeff1 * w[Nx-1, 0] + coeff2 * w[Nx-1, 0] * (3*uw[Nx-1, 0] - 4*uw[Nx-2, 0] + uw[Nx-3, 0]) + coeff2 * w[Nx-1, 0] * (-3*vw[Nx-1, 0] + 4*vw[Nx-1, 1] - vw[Nx-1, 2]) \
        + coeff2 * uw[Nx-1, 0] * (3*w[Nx-1, 0] - 4*w[Nx-2, 0] + w[Nx-3, 0]) + coeff2 * vw[Nx-1, 0] * (-3*w[Nx-1, 0] + 4*w[Nx-1, 1] - w[Nx-1, 2]) \
        + 4 * D * w[Nx-1, 0] + h**2 * delta + h**2 * kb * n[Nx-1, 0] * w[Nx-1, 0])

    # c{i-1,j} terms
    below = list(- (coeff2 * w[1:Nx-1, 0] * uw[1:Nx-1, 0]) - D * w[1:Nx-1, 0] + (D / 4) * (w[2:Nx, 0] - w[:Nx-2, 0]))
    below.append(- 2 * D * w[Nx-1, 0])    # no flux bouwdary conditions

    # c{i+1,j} terms
    above = list( (coeff2 * w[1:Nx-1, 0] * uw[1:Nx-1, 0]) - D * w[1:Nx-1, 0] - (D / 4) * (w[2:Nx, 0] - w[:Nx-2, 0]))
    above.insert(0, - 2 * D * w[0, 0])    # no flux bouwdary conditions

    B0 = sparse.diags([above, diagonals, below], [1, 0, -1])

    return B0

# ------------------------------------------------------------------------

# c{i,Ny-1} terms
def BN_matrix(uw, vw, w, w0, n, D, delta, kb, Nx, Ny, h, dt):
    
    coeff1 = h**2/dt
    coeff2 = h / 2
    
    # c{ij} terms
    diagonals = list(coeff1 * (w[1:Nx-1, Ny-1] - w0[1:Nx-1, Ny-1]) + coeff1 * w[1:Nx-1, Ny-1] + coeff2 * w[1:Nx-1, Ny-1] * (uw[2:Nx, Ny-1] - uw[:Nx-2, Ny-1]) + coeff2 * w[1:Nx-1, Ny-1] * (3*vw[1:Nx-1, Ny-1] - 4*vw[1:Nx-1, Ny-2] + vw[1:Nx-1, Ny-3]) \
        + coeff2 * uw[1:Nx-1, Ny-1] * (w[2:Nx, Ny-1] - w[:Nx-2, Ny-1]) + coeff2 * vw[1:Nx-1, Ny-1] * (3*w[1:Nx-1, Ny-1] - 4*w[1:Nx-1, Ny-2] + w[1:Nx-1, Ny-3]) \
        + 4 * D * w[1:Nx-1, Ny-1] + h**2 * delta + h**2 * kb * n[1:Nx-1, Ny-1] * w[1:Nx-1, Ny-1])
    diagonals.insert(0, coeff1 * (w[0, Ny-1] - w0[0, Ny-1]) + coeff1 * w[0, Ny-1] + coeff2 * w[0, Ny-1] * (-3*uw[0, Ny-1] + 4*uw[1, Ny-1] - uw[2, Ny-1]) + coeff2 * w[0, Ny-1] * (3*vw[0, Ny-1] - 4*vw[0, Ny-2] + vw[0, Ny-3]) \
        + coeff2 * uw[0, Ny-1] * (-3*w[0, Ny-1] + 4*w[1, Ny-1] - w[2, Ny-1]) + coeff2 * vw[0, Ny-1] * (3*w[0, Ny-1] - 4*w[0, Ny-2] + w[0, Ny-3]) \
        + 4 * D * w[0, Ny-1] + h**2 * delta + h**2 * kb * n[0, Ny-1] * w[0, Ny-1])
    diagonals.append(coeff1*(w[Nx-1, Ny-1] - w0[Nx-1, Ny-1]) + coeff1 * w[Nx-1, Ny-1] + coeff2 * w[Nx-1, Ny-1] * (3*uw[Nx-1, Ny-1] - 4*uw[Nx-2, Ny-1] + uw[Nx-3, Ny-1]) + coeff2 * w[Nx-1, Ny-1] * (3*vw[Nx-1, Ny-1] - 4*vw[Nx-1, Ny-2] + vw[Nx-1, Ny-3]) \
        + coeff2 * uw[Nx-1, Ny-1] * (3*w[Nx-1, Ny-1] - 4*w[Nx-2, Ny-1] + w[Nx-3, Ny-1]) + coeff2 * vw[Nx-1, Ny-1] * (3*w[Nx-1, Ny-1] - 4*w[Nx-1, Ny-2] + w[Nx-1, Ny-3]) \
        + 4 * D * w[Nx-1, Ny-1] + h**2 * delta + h**2 * kb * n[Nx-1, Ny-1] * w[Nx-1, Ny-1])

    # c{i-1,j} terms
    below = list(- (coeff2 * w[1:Nx-1, Ny-1] * uw[1:Nx-1, Ny-1]) - D * w[1:Nx-1, Ny-1] + (D / 4) * (w[2:Nx, Ny-1] - w[:Nx-2, Ny-1]))
    below.append(- 2 * D * w[Nx-1, Ny-1])    # no flux bouwdary conditions

    # c{i+1,j} terms
    above = list( (coeff2 * w[1:Nx-1, Ny-1] * uw[1:Nx-1, Ny-1]) - D * w[1:Nx-1, Ny-1] - (D / 4) * (w[2:Nx, Ny-1] - w[:Nx-2, Ny-1]))
    above.insert(0, - 2 * D * w[0, Ny-1])    # no flux bouwdary conditions

    BN = sparse.diags([above, diagonals, below], [1, 0, -1])

    return BN

# ------------------------------------------------------------------------

# n{ij-1} terms
def D1_matrix(j, vw, w, D, Nx, h):
    # off-diagonal block matrix for A for j = 1, ... n-2

    coeff = h / 2
 
    diagonals = - (coeff * vw[:Nx, j] * w[:Nx, j]) - D * w[:Nx, j] + (D / 4) * (w[:Nx, j+1] - w[:Nx, j-1])
    
    D1 = sparse.diags([diagonals], [0])

    return D1

# ------------------------------------------------------------------------

# n_{ij+1} terms
def D2_matrix(j, vw, w, D, Nx, h):
    # off-diagonal block matrix for A for j=1, ... n-2 

    coeff = h / 2

    diagonals = (coeff * vw[:Nx, j] * w[:Nx, j]) - D * w[:Nx, j] - (D / 4) * (w[:Nx, j+1] - w[:Nx, j-1])
    
    D2 = sparse.diags([diagonals], [0])

    return D2


# ------------------------------------------------------------------------

def vegf_solver(cell, water, solute, bound_solute, Nx, Ny, h, dt):
    """
    
    Solves the 2D advection-reaction-diffusion equation for the concentration of a solute 

    Parameters
    ----------
    cells : Cell object
    water : Water object
    solute : Solute object
    n : cell distribution at current time step
    uw : cell x-velocity at current time step
    vw : cell y-velocity at current time step
    c : solute concentration at previous time step
    w : water distribution at current time step
    w0 : water distribution at previous time step
    D : solute diffusion coefficient
    delta : solute degradation rate
    alpha : solute production rate
    kappa: solute uptake rate
    Nx : number of grid points in x-direction
    Ny : number of grid points in y-direction
    h : grid spacing
    dt : time step

    Returns
    -------
    c : solute concentration at current time step
    
    """

    # uwpack variables
    n = cell.distribution
    uw = water.x_velocity
    vw = water.y_velocity
    c = solute.distribution
    c0 = solute.previous_distribution
    cb = bound_solute.distribution
    w = water.distribution
    w0 = water.previous_distribution
    D = solute.diffusion_rate
    delta = solute.degradation_rate
    alpha = solute.production_rate
    kappa = solute.uptake_rate
    K = solute.uptake_constant
    phi = water.available_volume_fraction
    m = water.matrix_volume_fraction
    kb = solute.binding_rate
    ku = bound_solute.unbinding_rate


    # create block matrix A

    A = sparse.lil_matrix((Nx*Ny, Nx*Ny))

    # fill in the diagonal blocks
    A[0:Nx, 0:Nx] = B0_matrix(uw, vw, w, w0, n, D, delta, kb, Nx, h, dt)

    # first off-diagonal
    diagonals0 = -2 * D * w[:, 0]
    A[0:Nx, Nx:Nx*2] = sparse.diags([diagonals0], [0])

    for i in range(1, Ny-1):   
        A[i*Nx:(i+1)*Nx, i*Nx:(i+1)*Nx] = B_matrix(i, uw, vw, w, w0, n, D, delta, kb, Nx, h, dt)
        A[i*Nx:(i+1)*Nx, (i-1)*Nx:i*Nx] = D1_matrix(i, vw, w, D, Nx, h)
        A[i*Nx:(i+1)*Nx, (i+1)*Nx:(i+2)*Nx] = D2_matrix(i, vw, w, D, Nx, h)

    A[(Ny-1)*Nx:Ny*Nx, (Ny-1)*Nx:Ny*Nx] = BN_matrix(uw, vw, w, w0, n, D, delta, kb, Nx, Ny, h, dt)

    # last off-diagonal
    diagonalsN = -2 * D * w[:, Ny-1]
    A[(Ny-1)*Nx:Ny*Nx, (Ny-2)*Nx:(Ny-1)*Nx] = sparse.diags([diagonalsN], [0])


    # convert to CSR format
    A = A.tocsr()

    # create RHS vector 
    b2d = (h**2 * c[:] * w[:]/ dt) + (h**2 * alpha * n[:]) * (phi - n[:]) - ((h**2 * kappa * n[:] * c0[:] * w[:]) / (K + c0[:])) + (h**2 * ku * w[:] * m * cb[:]) 

    b1d = b2d.flatten(order='F')

    # solve linear system
    c_new = spsolve(A, b1d)

    c = c_new.reshape(Nx, Ny, order='F')


    #-- solve for bound solute, cb 
    invt = 1/dt
    coeff = kb / m

    diagonals = (invt + delta) * np.ones((Nx*Ny))
    Ab = sparse.diags([diagonals], [0])
    Ab = sparse.csr_matrix(Ab)

    e2d = cb * (invt - (ku * w)) + coeff * n * c * w
    e1d = e2d.flatten(order='F')

    cb_new = spsolve(Ab, e1d)

    cb = cb_new.reshape(Nx, Ny, order='F')

    return c, cb


