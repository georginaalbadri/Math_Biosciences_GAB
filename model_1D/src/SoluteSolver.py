import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# -------------------------------------------------------------------------------------

def solute_solver_no_flux(solute, bound_solute, cell, water, phi, nx, dx, dt): 

    """
    Solve the 1D solute transport equation with no flux boundary conditions.

    Parameters
    ----------
    solute : Solute dataclass
    cell : Cell dataclass
    water : Water dataclass
    c0 : initial solute concentration (numpy array)
    n : current cell distribution (numpy array)
    w : current water concentration (numpy array)
    w0 : initial water concentration (numpy array)
    vw : current water velocity (numpy array)
    D : solute diffusion coefficient (float)
    alpha : production rate (float)
    delta : degradation rate (float)
    kappa : uptake rate (float)
    K : solute half saturation constant (float)
    kb : solute binding rate (float)
    ku : solute unbinding rate (float)
    phi : available volume fraction (float)
    nx : number of grid points (int)
    dx : grid spacing (float)
    dt : time step (float)    
    Returns
    -------
    c : current solute concentration (numpy array)    
    
    """
    c = solute.distribution
    c0 = solute.previous_distribution
    cb = bound_solute.distribution
    n = cell.distribution
    w = water.distribution
    w0 = water.previous_distribution
    vw = water.velocity
    m = water.matrix_volume_fraction
    D = solute.diffusion_rate * np.ones((nx))
    alpha = solute.production_rate
    delta = solute.degradation_rate
    kappa = solute.uptake_rate
    K = solute.uptake_constant
    kb = solute.binding_rate
    ku = bound_solute.unbinding_rate
    dx2 = dx * dx

    #-- matrix A (Ac = B)

    diagonals = list(((w[2:nx-2] - w0[2:nx-2]) / dt) + (w[2:nx-2] / dt) + (w[2:nx-2] / (2*dx)) * (vw[3:nx-1] - vw[1:nx-3]) \
            + (vw[2:nx-2] / (2*dx)) * (w[3:nx-1] - w[1:nx-3]) + ((D[3:nx-1] * w[3:nx-1]) / (4 * dx2)) + ((D[1:nx-3] * w[1:nx-3]) / (4 * dx2)) + delta \
                + (kb * n[2:nx-2] * w[2:nx-2]))
    diagonals.insert(0, ((w[1] - w0[1]) / dt) + (w[1] / dt) + (w[1] / (2*dx)) * (vw[2] - vw[0]) \
            + (vw[1] / (2*dx)) * (w[2] - w[0]) + ((D[2] * w[2]) / (4 * dx2)) + delta + (kb * n[1] * w[1]))
    diagonals.insert(0, ((w[0] - w0[0]) / dt) + (w[0] / dt) + (w[0] / (2*dx)) * (- 3* vw[0] + 4*vw[1] - vw[2]) \
            + (2 * D[0] * w[0] / dx2) + delta + (kb * n[0] * w[0]))
    diagonals.append( ((w[nx-2] - w0[nx-2]) / dt) + (w[nx-2] / dt) + ((D[nx-3] * w[nx-3]) / (4 * dx2)) + delta + (kb * n[nx-2] * w[nx-2]))
    diagonals.append((1/ dt) + (2 * D[nx - 1] / dx2) + delta + kb * n[nx-1] * w[nx-1])

    below = list(- ((w[2:nx-2] * vw[2:nx-2]) / (2 * dx)) )
    below.insert(0, - ((w[1] * vw[1]) / (2 * dx)) )
    below.append(- ((D[nx-3] * w[nx-3]) / (4 * dx2)) )
    below.append(- 2 * D[nx - 1] / dx2 )

    above = list( ((w[2:nx-2] * vw[2:nx-2]) / (2 * dx)) )
    above.insert(0, ((w[1] * vw[1]) / (2 * dx)))
    above.insert(0, - 2* D[0] * w[0] / dx2)
    above.append(0) #is this right?

    below2 = list( - (D[1:nx-3] * w[1:nx-3]) / (4 * dx2)  )
    below2.append(0)
    below2.append(0)

    above2 = list(- (D[3:nx-1] * w[3:nx-1]) / (4 * dx2) )
    above2.insert(0, - (D[2] * w[2]) / (4 * dx2) )
    above2.insert(0, 0)

    A = sparse.diags([diagonals, below, below2, above, above2], [0, -1, -2, 1, 2])

    A = sparse.csr_matrix(A)

    #-- source terms B
    B = np.zeros((nx))
    B[:nx] =  ((w[:nx] * c0[:nx]) / dt) + (alpha * n[:nx]) * (phi - n[:nx]) - ((kappa * n[:nx] * c0[:nx] * w[:nx]) / (K + c0[:nx]) ) + ku * w[:nx] * m * cb[:nx] 

    #-- solve matrix equation for c

    c = spsolve(A, B)

    
    return c

# -------------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------

def bound_solute_solver(bound_solute, solute, cell, water, nx, dt):

    delta = bound_solute.degradation_rate
    kb = solute.binding_rate
    ku = bound_solute.unbinding_rate
    m = water.matrix_volume_fraction
    w = water.distribution
    n = cell.distribution
    c = solute.distribution
    cb = bound_solute.distribution

    invt = 1/dt
    coeff = kb / m

    diagonals = (invt + delta) * np.ones((nx))
    Ab = sparse.diags([diagonals], [0])
    Ab = sparse.csr_matrix(Ab)

    Bb = np.zeros((nx))
    Bb[:nx] = cb[:nx] * (invt - (ku * w[:nx])) + coeff * n[:nx] * c[:nx] * w[:nx]

    cb = spsolve(Ab, Bb)

    return cb

# -------------------------------------------------------------------------------------



# -------------------------------------------------------------------------------------------

def solute_solver_diffusive_flux(solute, bound_solute, cell, water, phi, nx, dx, dt): 
    """
    Solve the solute transport equation using a matrix solver.
    
    Parameters
    ----------
    solute : Solute dataclass
    cell : Cell dataclass
    water : Water dataclass
    c0 : previous solute distribution (numpy array)
    n : cell volume fraction (numpy array)
    w : water volume fraction (numpy array)
    w0 : previous water volume fraction (numpy array)
    vw : water velocity (numpy array)
    D : solute diffusion coefficient (float)
    delta : solute degradation rate (float)
    alpha : solute production rate (float)
    phi : available volume fraction (float)
    nx : number of grid points (int)
    dx : grid spacing (float)
    dt : time step (float)

    Returns
    -------
    c : solute distribution (numpy array)
    
    """
    c = solute.distribution
    c0 = solute.previous_distribution
    cb = bound_solute.distribution
    n = cell.distribution
    w = water.distribution
    w0 = water.previous_distribution
    m = water.matrix_volume_fraction
    vw = water.velocity
    D = solute.diffusion_rate
    alpha = solute.production_rate
    delta = solute.degradation_rate
    kappa = solute.uptake_rate
    K = solute.half_saturation_constant
    kb = solute.binding_rate
    ku = bound_solute.unbinding_rate
    dx2 = dx * dx
    
    dx2 = dx**2

    #-- matrix A (Ac = B)

    diagonals = list(((w[1:nx-1] - w0[1:nx-1]) / dt) + (w[1:nx-1] / dt) + (w[1:nx-1] / (2*dx)) * (vw[2:nx] - vw[:nx-2]) \
            + (vw[1:nx-1] / (2*dx)) * (w[2:nx] - w[:nx-2]) + (2 * D * w[1:nx-1] / (dx2)) + delta + (kb * n[1:nx-1] * w[1:nx-1]))

    diagonals.insert(0, ((w[0] - w0[0]) / dt) + (w[0] / dt) + (w[0] / (2*dx)) * (- 3* vw[0] + 4*vw[1] - vw[2]) + (vw[0] / (2*dx)) * (- 3* w[0] + 4 * w[1] - w[2]) \
            + (3 * D / (4 * dx2)) * (-3*w[0] + 4*w[1] - w[2]) + delta - (D * w[0] / dx2) + (kb * n[0] * w[0]))

    diagonals.append(((w[nx-1] - w0[nx-1]) / dt) + (w[nx-1] / dt) + (w[nx-1] / (2*dx)) * (3* vw[nx-1] - 4*vw[nx-2] + vw[nx-3]) + (vw[nx-1] / (2*dx)) * (3* w[nx-1] - 4*w[nx-2] + w[nx-3]) \
            + (3 * w[nx-1] * vw[nx-1] / (2 * dx)) - (3 * D)/(4 * dx2) * (3* w[nx-1] - 4*w[nx-2] + w[nx-3]) - (2 * D * w[nx-1] / dx2) + delta + (kb * n[nx-1] * w[nx-1]))

    below = list(- ((w[1:nx-1] * vw[1:nx-1]) / (2 * dx)) + (D / (4 * dx2)) * (w[2:nx] - w[:nx-2]) - (D / dx2) * w[1:nx-1]  )
    below.append(- ((2 * w[nx-1] * vw[nx-1]) / (dx)) + (D / dx2) * (3*w[nx-1] - w[nx-2] + w[nx-3]) + (5 * D / dx2) * w[nx-1] )

    above = list(((w[1:nx-1] * vw[1:nx-1]) / (2 * dx)) - (D / (4 * dx2)) * (w[2:nx] - w[:nx-2]) - (D / dx2) * w[1:nx-1]   )
    above.insert(0, (2 * w[0] * vw[0] / dx) - (D / dx2) * (-3*w[0] + 4*w[1] - w[2]) + (5 * D * w[0] / dx2))

    A = sparse.diags([diagonals, below, above], [0, -1, 1])
   
    A = sparse.csr_matrix(A)

    # add off-diagonal elements arising from no flux boundary conditions 

    A[0, 2] = - (w[0] * vw[0] / (2 * dx)) + (D / (4 * dx2) * ( -3*w[0] + 4*w[1] - w[2])) - (4 * D * w[0] / dx2)

    A[0, 3] = D * w[0] / dx2

    A[nx-1, nx-3] = (w[nx-1] * vw[nx-1] / (2 * dx)) - (D / (4 * dx2) * ( 3*w[nx-1] - 4*w[nx-2] + w[nx-3])) - (4 * D * w[nx-1] / dx2)

    A[nx-1, nx-4] = D * w[nx-1] / dx2


    #-- source terms B
    B = np.zeros((nx))

    B[:nx] =  ((w[:nx] * c0[:nx]) / dt) + (alpha * n[:nx]) * (phi - n[:nx]) - ((kappa * n[:nx] * c0[:nx] * w[:nx]) / (K + c0[:nx]) ) + ku * w[:2*nx] * m * cb[:2*nx]


    #-- solve matrix equation for c

    c = spsolve(A, B)
    
    return c


# -------------------------------------------------------------------------------------------