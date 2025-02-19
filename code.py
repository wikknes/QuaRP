#!/usr/bin/env python3
"""
Parameter Sweeps and Adaptive Feedback for Quantum State Refinement

This script demonstrates two key extensions to a basic feedback protocol:
1) Parameter sweeps over feedback strength (k_fb) and noise (gamma) to
   generate a heatmap of final fidelities.
2) An enhanced (adaptive) feedback method, where k_fb is dynamically adjusted
   based on the current fidelity, aiming to maintain high fidelity longer.

Author: Vignesh
Date: 19/02/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, Qobj, sigmaz

# =============================================================================
# 1. Basic Feedback Simulation with Noise
# =============================================================================

def run_feedback_simulation(rho_initial: Qobj, rho_target: Qobj,
                            dt: float, n_steps: int,
                            k_fb: float, gamma: float) -> np.ndarray:
    """
    Simulate the evolution of a quantum state under a constant-feedback protocol
    with dephasing noise (gamma).

    Args:
        rho_initial (Qobj): Initial density matrix.
        rho_target (Qobj): Target density matrix.
        dt (float): Time step per iteration.
        n_steps (int): Number of iterations.
        k_fb (float): Constant feedback strength.
        gamma (float): Dephasing noise rate.

    Returns:
        fidelities (np.ndarray): Fidelity to rho_target at each time step.
    """
    # Dephasing noise operator
    L = np.sqrt(gamma) * sigmaz()

    fidelities = []
    rho = rho_initial.copy()

    for _ in range(n_steps):
        # Feedback Hamiltonian: H_fb = i * k_fb * (rho_target*rho - rho*rho_target)
        H_fb = 1j * k_fb * (rho_target * rho - rho * rho_target)

        # Unitary evolution from feedback
        U = (-1j * H_fb * dt).expm()
        rho = U * rho * U.dag()

        # Dephasing noise via Euler step
        noise_term = dt * (L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L))
        rho = rho + noise_term

        # Normalize
        rho = rho / rho.tr()

        # Record fidelity
        fid = (rho_target * rho).tr().real
        fidelities.append(fid)

    return np.array(fidelities)

# =============================================================================
# 2. Adaptive Feedback Simulation
# =============================================================================

def run_adaptive_feedback_simulation(rho_initial: Qobj, rho_target: Qobj,
                                     dt: float, n_steps: int,
                                     k_fb0: float, gamma: float,
                                     alpha: float) -> np.ndarray:
    """
    Simulate a quantum state under an *adaptive* feedback protocol where
    the feedback strength k_fb is dynamically adjusted based on the current
    fidelity: k_fb(t) = k_fb0 * (1 + alpha * [1 - fidelity]).

    Args:
        rho_initial (Qobj): Initial density matrix.
        rho_target (Qobj): Target density matrix.
        dt (float): Time step per iteration.
        n_steps (int): Number of iterations.
        k_fb0 (float): Base feedback strength.
        gamma (float): Dephasing noise rate.
        alpha (float): Adaptation rate (0 -> no adaptation; higher -> stronger adaptation).

    Returns:
        fidelities (np.ndarray): Fidelity to rho_target at each time step.
    """
    L = np.sqrt(gamma) * sigmaz()

    fidelities = []
    rho = rho_initial.copy()

    for _ in range(n_steps):
        # Current fidelity
        current_fid = (rho_target * rho).tr().real

        # Adapt feedback strength: k_fb(t) = k_fb0 * (1 + alpha*(1 - fidelity))
        k_fb_t = k_fb0 * (1.0 + alpha * (1.0 - current_fid))

        # Feedback Hamiltonian
        H_fb = 1j * k_fb_t * (rho_target * rho - rho * rho_target)

        # Unitary evolution from feedback
        U = (-1j * H_fb * dt).expm()
        rho = U * rho * U.dag()

        # Dephasing noise
        noise_term = dt * (L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L))
        rho = rho + noise_term

        # Normalize
        rho = rho / rho.tr()

        fidelities.append((rho_target * rho).tr().real)

    return np.array(fidelities)

# =============================================================================
# 3. Parameter Sweep (k_fb vs gamma)
# =============================================================================

def parameter_sweep(rho_initial: Qobj, rho_target: Qobj,
                    dt: float, n_steps: int,
                    k_fb_values: np.ndarray,
                    gamma_values: np.ndarray) -> np.ndarray:
    """
    Sweep over feedback strength (k_fb) and noise (gamma), returning the final fidelity
    for each combination.

    Args:
        rho_initial (Qobj): Initial density matrix.
        rho_target (Qobj): Target density matrix.
        dt (float): Time step per iteration.
        n_steps (int): Number of iterations.
        k_fb_values (np.ndarray): Array of k_fb values to test.
        gamma_values (np.ndarray): Array of gamma values to test.

    Returns:
        fidelity_map (np.ndarray): 2D array of shape (len(k_fb_values), len(gamma_values))
                                  containing final fidelities.
    """
    fidelity_map = np.zeros((len(k_fb_values), len(gamma_values)))

    for i, k_fb in enumerate(k_fb_values):
        for j, gamma in enumerate(gamma_values):
            fidelities = run_feedback_simulation(rho_initial, rho_target,
                                                 dt, n_steps, k_fb, gamma)
            fidelity_map[i, j] = fidelities[-1]  # Final fidelity

    return fidelity_map

# =============================================================================
# 4. Plotting Functions
# =============================================================================

def plot_heatmap(k_fb_values, gamma_values, fidelity_map):
    """
    Plot a heatmap of final fidelities as a function of (k_fb, gamma).
    """
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(gamma_values, k_fb_values)
    c = plt.pcolormesh(X, Y, fidelity_map, cmap='viridis', shading='auto')
    plt.colorbar(c, label='Final Fidelity')
    plt.xlabel("Noise Rate (gamma)")
    plt.ylabel("Feedback Strength (k_fb)")
    plt.title("Final Fidelity Heatmap (Parameter Sweep)")
    plt.show()

def plot_fidelity_curves(times, fidelities_list, labels_list, title="Fidelity vs Time"):
    """
    Plot multiple fidelity curves on the same figure.
    """
    plt.figure(figsize=(8, 6))
    for fid, lbl in zip(fidelities_list, labels_list):
        plt.plot(times, fid, label=lbl, lw=2)
    plt.xlabel("Time")
    plt.ylabel("Fidelity to Target State")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# =============================================================================
# 5. Main Demonstration
# =============================================================================

def main():
    # Simulation settings
    dt = 0.1
    n_steps = 100
    times = np.linspace(0, dt*n_steps, n_steps)

    # Define target state: |+> = (|0> + |1>)/sqrt(2)
    from qutip import basis
    plus_state = (basis(2, 0) + basis(2, 1)).unit()
    rho_target = plus_state * plus_state.dag()

    # Initial state: |0>
    rho_initial = basis(2, 0) * basis(2, 0).dag()

    # --- 1) Parameter Sweep Example ---
    k_fb_values = np.linspace(0.1, 2.0, 10)   # 10 points from 0.1 to 2.0
    gamma_values = np.linspace(0.01, 0.1, 10) # 10 points from 0.01 to 0.1

    fidelity_map = parameter_sweep(rho_initial, rho_target,
                                   dt, n_steps, k_fb_values, gamma_values)

    plot_heatmap(k_fb_values, gamma_values, fidelity_map)

    # --- 2) Compare Constant vs Adaptive Feedback ---

    # Choose a single pair (k_fb=1.0, gamma=0.05) for demonstration
    k_fb_fixed = 1.0
    gamma_fixed = 0.05

    # Run constant feedback
    fidelities_const = run_feedback_simulation(rho_initial, rho_target,
                                               dt, n_steps, k_fb_fixed, gamma_fixed)

    # Run adaptive feedback with some adaptation rate alpha
    alpha = 2.0  # The higher alpha is, the stronger the adaptation
    fidelities_adapt = run_adaptive_feedback_simulation(rho_initial, rho_target,
                                                        dt, n_steps,
                                                        k_fb_fixed, gamma_fixed,
                                                        alpha)

    # Plot both curves
    plot_fidelity_curves(times,
                         [fidelities_const, fidelities_adapt],
                         ["Constant Feedback", f"Adaptive Feedback (alpha={alpha})"],
                         title="Constant vs Adaptive Feedback")

if __name__ == "__main__":
    main()
