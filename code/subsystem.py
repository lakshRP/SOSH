
# --- subsystem.py ---
"""
Defines the Subsystem class: state update, residuals, detection, and attack injection.
"""
import numpy as np

class Subsystem:
    """
    Represents a single agent in the MAS with detection and attack capabilities.
    """
    def __init__(self, system, initial_state: np.ndarray):
        self.system = system
        self.state = np.array(initial_state, dtype=float)
        self.ideal_state = self.state.copy()
        self.residuals = []
        self.attack_fn = None
        self.is_detected = False
        self.detection_counter = 0

    def update_state(self, step: int):
        """
        Updates the agent's state with consensus dynamics, noise, and optional attack.
        Records residual against the ideal state.
        Args:
            step (int): current timestep, passed to attack_fn.
        """
        # Noise
        noise = np.random.uniform(-self.system.noise_bounds,
                                  self.system.noise_bounds)
        # Neighbors' states provided externally via global positions list
        # Coupling placeholder (to be filled by main driver)
        coupling = np.zeros_like(self.state)

        # Dynamics update
        self.state = self.system.Aii @ self.state + coupling + self.system.Bi + noise
        # Attack injection
        if self.attack_fn:
            self.state += self.attack_fn(step)

    def compute_residual(self):
        """
        Residual between actual state and ideal (noise-free) state.
        Returns:
            float: Euclidean norm.
        """
        res = np.linalg.norm(self.state - self.ideal_state)
        self.residuals.append(res)
        return res

    def add_attack(self, start: int, end: int, magnitude: np.ndarray):
        """
        Attaches a constant attack function active in [start, end).
        """
        def attack_fn(step: int) -> np.ndarray:
            if start <= step < end:
                return magnitude
            return np.zeros_like(magnitude)
        self.attack_fn = attack_fn

    def update_ideal(self, neighbor_states):
        """
        Noise-free, attack-free dynamics for ideal trajectory.
        Args:
            neighbor_states (list[np.ndarray]): latest neighbor states.
        Returns:
            np.ndarray: updated ideal_state.
        """
        coupling = np.zeros_like(self.ideal_state)
        # ... same coupling logic as update_state but without noise/attack
        self.ideal_state = self.system.Aii @ self.ideal_state + coupling + self.system.Bi
        return self.ideal_state
