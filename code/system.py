# --- system.py ---
"""
Defines the System class and provides default system configurations.
"""
import numpy as np

class System:
    """
    Encapsulates dynamics parameters for all agents.
    Attributes:
        Aii (np.ndarray): Self-dynamics matrix.
        Aij (np.ndarray): Coupling matrix (neighbor influence).
        Bi (np.ndarray): Control input vector.
        noise_bounds (np.ndarray): Bounds for uniform process noise.
    """
    def __init__(self, Aii: np.ndarray, Aij: np.ndarray, Bi: np.ndarray, noise_bounds: np.ndarray):
        self.Aii = Aii
        self.Aij = Aij
        self.Bi = Bi
        self.noise_bounds = noise_bounds


def create_default_system() -> System:
    """
    Factory for a default 2D system with simple consensus gains.
    Returns:
        System: with Aii = 0.9*I, Aij = 0.1*I, Bi = 0, noise_bounds = [0.01, 0.01]
    """
    Aii = np.eye(2) * 0.9
    Aij = np.eye(2) * 0.1
    Bi = np.zeros(2)
    noise_bounds = np.array([0.01, 0.01])
    return System(Aii, Aij, Bi, noise_bounds)

# Global neighbor mapping and desired distances for square formation
neighbor_mapping = {
    0: [1, 3],
    1: [0, 2],
    2: [1, 3],
    3: [0, 2],
}

desired_distances = {
    0: [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
    1: [np.array([-1.0, 0.0]), np.array([0.0, 1.0])],
    2: [np.array([0.0, -1.0]), np.array([-1.0, 0.0])],
    3: [np.array([0.0, -1.0]), np.array([1.0, 0.0])],
}
