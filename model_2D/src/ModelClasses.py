import numpy as np
from dataclasses import dataclass
import random




@dataclass
class Grid:

    nx: int = 200
    T : float = 1e5
    L : float = 3e-3
    H : float = 3e-3
    h: float = 1/nx
    ny: int = int((H/L)/h)


@dataclass
class CellType:

    diffusion_rate: float = 1e-14
    initial_volume_fraction: float = 0.05
    chemotaxis_strength: float = 0.1
    aggregation_strength: float = 0.0
    matrix_drag : float = 1e10
    matrix_traction : float = 0.0
    contact_inhibition : float = 0.0
    bound_chemotaxis_strength: float = 0


    def uniform_cell_distribution(self, water, grid):

        distribution = np.ones((grid.nx, grid.ny)) * self.initial_volume_fraction
        water.distribution = water.available_volume_fraction - distribution

        return distribution
        
    def noisy_cell_distribution(self, water, grid):

        np.random.seed(45) #fix noise distribution to be the same every time
        distribution = self.initial_volume_fraction + np.random.normal(0, 0.1*self.initial_volume_fraction, size = (int(grid.nx), int(grid.ny)))
        #distribution = np.repeat(distribution, 2, axis = 0)
        #distribution = np.repeat(distribution, 2, axis = 1)
        water.distribution = water.available_volume_fraction - distribution

        return distribution

    def SA_cell_distribution(self, water, grid, seed=0, noise_level=0.1, spacing=1):

        np.random.seed(seed)
        distribution = self.initial_volume_fraction + np.random.normal(0, noise_level*self.initial_volume_fraction, (int(grid.nx/spacing), int(grid.ny/spacing)))
        distribution = np.repeat(distribution, 5, axis = 0)
        distribution = np.repeat(distribution, spacing, axis = 1)

        water.distribution = water.available_volume_fraction - distribution

        return distribution


@dataclass
class Water:

    matrix_drag: float = 5e7
    matrix_volume_fraction: float = 0.03
    available_volume_fraction: float = 1 - matrix_volume_fraction

    assert matrix_volume_fraction + available_volume_fraction == 1, "Matrix and available volume fractions do not sum to 1"


@dataclass
class Solute:

    diffusion_rate: float = 1e-11
    production_rate: float = 1e-12
    degradation_rate: float = 1e-4
    uptake_rate: float = 0.0
    uptake_constant: float = 1e-7
    initial_solute_concentration: float = 0.0
    binding_rate: float = 0

    def uniform_solute_distribution(self, water, grid):

        distribution = np.ones((grid.nx, grid.ny)) * self.initial_solute_concentration
        water.distribution = water.available_volume_fraction - distribution

        return distribution


@dataclass
class BoundSolute:

    initial_solute_concentration: float = 0
    unbinding_rate: float = 0
    degradation_rate: float = 0

    def uniform_solute_distribution(self, water, grid):

        distribution = np.ones((grid.nx, grid.ny)) * self.initial_solute_concentration
        water.distribution = water.available_volume_fraction - distribution

        return distribution