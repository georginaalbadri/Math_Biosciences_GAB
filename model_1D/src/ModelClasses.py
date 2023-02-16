import numpy as np
from dataclasses import dataclass
import random




@dataclass
class Geometry:

    T : float = 1e5
    L: float = 1e-3
        

@dataclass
class Grid:

    nx: int = 500
    dx: float = 1/nx


@dataclass
class CellType:

    diffusion_rate: float = 1e-14
    initial_volume_fraction: float = 0.05
    chemotaxis_strength: float = 0.1
    aggregation_strength: float = 0.0
    matrix_drag : float = 1e10
    matrix_traction : float = 0.0
    contact_inhibition : float = 0.0
    chemotaxis_strength_bound: float = 0.1


    def uniform_cell_distribution(self, water, grid):

        distribution = np.ones(grid.nx) * self.initial_volume_fraction
        water.distribution = water.available_volume_fraction - distribution

        return distribution
        

    def noisy_cell_distribution(self, water, grid):

        np.random.seed(28) #fix noise distribution to be the same every time
        distribution = self.initial_volume_fraction + np.random.normal(0, 0.1*self.initial_volume_fraction, int(grid.nx/5))
        distribution = np.repeat(distribution, 5)
        water.distribution = water.available_volume_fraction - distribution

        return distribution

    def SA_cell_distribution(self, water, grid, seed=0, noise_level=0.1, spacing=1):

        np.random.seed(seed)
        distribution = self.initial_volume_fraction + np.random.normal(0, noise_level*self.initial_volume_fraction, int(grid.nx/spacing))
        distribution = np.repeat(distribution, spacing)

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
    binding_rate: float = 1.5e-3

    def uniform_solute_distribution(self, water, grid):

        distribution = np.ones(grid.nx) * self.initial_solute_concentration
        water.distribution = water.available_volume_fraction - distribution

        return distribution


@dataclass
class BoundSolute:

    initial_solute_concentration: float = 0
    unbinding_rate: float = 3.6e-3
    degradation_rate: float = 1e-4

    def uniform_solute_distribution(self, water, grid):

        distribution = np.ones(grid.nx) * self.initial_solute_concentration
        water.distribution = water.available_volume_fraction - distribution

        return distribution