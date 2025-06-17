# --- hallucination.py ---
"""
Functions to compute hallucinated states using second-order approximations.
"""
import numpy as np

def compute_hallucinated_state(system, current_state, neighbor_states, desired_distances):
    """
    Predictive second-order Taylor update given last state.
    """
    # Linear term
    linear = system.Aii @ current_state
    # Nonlinear correction (placeholder quadratic term)
    quad = 0.5 * (current_state ** 2)
    return linear + quad


def update_ideal_state(system, ideal_state, neighbor_states, desired_distances):
    """
    Clean update without noise or attack.
    """
    return system.Aii @ ideal_state
