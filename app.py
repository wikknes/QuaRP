#!/usr/bin/env python3
"""
Benchmarking Adaptive Quantum State Refinement with Statistical Analysis

This script performs multiple simulation runs of a feedback-controlled
quantum state refinement protocol with noise. It then computes the average
fidelity and standard deviation at each time point, comparing the feedback
protocol against a baseline (no feedback). The resulting plots include error
bars (or shaded regions) to indicate statistical reliability.

Requirements:
    - Python 3.x
    - QuTiP
    - NumPy
    - Matplotlib

Author: [Your Name]
Date: [Today's Date]
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, Qobj, sigmaz

# =============================================================================
# Simulation Functions with Stochastic Noise
# =============================================================================
def run_feedback_simulation_stochastic(rho_initial: Qobj, rho_target: Qobj,
                                         dt: float, n_steps: int,
                                         k_fb: float, gamma: float,
                                         noise_strength: float) -> np.ndarray:
    """
    Run one simulation run of the feedback-controlled evolution with noise.
    Here, we add a small random perturbation at each step (stochastic noise)
    to simulate variations in experimental conditions.

    Parameters:
        rho_initial (Qobj): The starting density matrix.
        rho_target (Qobj): The target density matrix.
        dt (float): Time step.
        n_steps (int): Number of iterations.
        k_fb (float): Feedback strength.
        gamma (float): Deterministic dephasing noise strength.
        noise_strength (float): Additional stochastic noise level.

    Returns:
        fidelities (np.ndarray): Array of fidelity values over time.
    """
    # Dephasing noise operator (deterministic part)
    L = np.sqrt(gamma) * sigmaz()

    fidelities = []
    rho = rho_initial.copy()

    for step in range(n_steps):
        # 1. Compute feedback Hamiltonian: H_fb = i * k_fb * (rho_target * rho - rho * rho_target)
        H_fb = 1j * k_fb * (rho_target * rho - rho * rho_target)

        # 2. Compute the unitary evolution due to feedback:
        U = (-1j * H_fb * dt).expm()
        rho = U * rho * U.dag()

        # 3. Add deterministic dephasing noise via Euler step:
        noise_term_det = dt * (L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L))
        rho = rho + noise_term_det

        # 4. Add an extra stochastic noise term:
        # Here we add a small random Hermitian perturbation.
        random_matrix = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2))
        random_herm = (random_matrix + random_matrix.conj().T) / 2  # make it Hermitian
        noise_term_stoch = dt * noise_strength * Qobj(random_herm)
        rho = rho + noise_term_stoch

        # 5. Renormalize the density matrix (ensure trace equals 1)
        rho = rho / rho.tr()

        # 6. Record fidelity: F = Tr(rho_target * rho)
        fid = (rho_target * rho).tr().real
        fidelities.append(fid)

    return np.array(fidelities)

def run_baseline_simulation_stochastic(rho_initial: Qobj, rho_target: Qobj,
                                         dt: float, n_steps: int,
                                         gamma: float,
                                         noise_strength: float) -> np.ndarray:
    """
    Run one simulation run for the baseline (non-adaptive evolution) with noise.
    Here the system is not controlled by feedback, but noise (both deterministic and
    stochastic) still acts on it.

    Returns:
        fidelities (np.ndarray): Array of fidelity values over time.
    """
    L = np.sqrt(gamma) * sigmaz()

    fidelities = []
    rho = rho_initial.copy()

    for step in range(n_steps):
        # Baseline: no feedback Hamiltonian (free evolution)
        # Only noise evolves the state.
        noise_term_det = dt * (L * rho * L.dag() - 0.5 * (L.dag() * L * rho + rho * L.dag() * L))
        rho = rho + noise_term_det

        # Add stochastic noise as well:
        random_matrix = (np.random.randn(2, 2) + 1j * np.random.randn(2, 2))
        random_herm = (random_matrix + random_matrix.conj().T) / 2
        noise_term_stoch = dt * noise_strength * Qobj(random_herm)
        rho = rho + noise_term_stoch

        # Renormalize:
        rho = rho / rho.tr()

        fid = (rho_target * rho).tr().real
        fidelities.append(fid)

    return np.array(fidelities)

# =============================================================================
# Benchmarking: Running Multiple Simulation Runs
# =============================================================================
def run_multiple_simulations(rho_initial: Qobj, rho_target: Qobj, dt: float,
                             n_steps: int, k_fb: float, gamma: float,
                             noise_strength: float, num_runs: int, use_feedback: bool):
    """
    Run multiple simulation runs (Monte Carlo trajectories) and compute mean fidelity
    and standard deviation over time.

    Parameters:
        use_feedback (bool): If True, run feedback simulation; else, run baseline.

    Returns:
        mean_fid (np.ndarray): Mean fidelity at each time step.
        std_fid (np.ndarray): Standard deviation of fidelity at each time step.
    """
    all_fidelities = []
    for _ in range(num_runs):
        if use_feedback:
            fid = run_feedback_simulation_stochastic(rho_initial, rho_target,
                                                     dt, n_steps, k_fb, gamma, noise_strength)
        else:
            fid = run_baseline_simulation_stochastic(rho_initial, rho_target,
                                                     dt, n_steps, gamma, noise_strength)
        all_fidelities.append(fid)
    all_fidelities = np.array(all_fidelities)  # shape: (num_runs, n_steps)
    mean_fid = np.mean(all_fidelities, axis=0)
    std_fid = np.std(all_fidelities, axis=0)
    return mean_fid, std_fid

# =============================================================================
# Main Function: Benchmark and Statistical Analysis
# =============================================================================
def main():
    # Simulation parameters
    dt = 0.01         # Time step
    n_steps = 1000    # Number of iterations
    k_fb = .25        # Feedback strength constant for adaptive method
    gamma = 0.05      # Deterministic dephasing noise strength
    noise_strength = 0.02  # Stochastic noise strength (random fluctuations)
    num_runs = 50     # Number of simulation runs for statistical analysis

    # Define the target state: |+> = (|0> + |1>)/sqrt(2)
    plus_state = (basis(2, 0) + basis(2, 1)).unit()
    rho_target = plus_state * plus_state.dag()

    # Define an initial state (you could also vary this over multiple runs)
    rho_initial = basis(2, 0) * basis(2, 0).dag()

    # Run simulations for the feedback (adaptive) method
    mean_fb, std_fb = run_multiple_simulations(rho_initial, rho_target, dt, n_steps,
                                                 k_fb, gamma, noise_strength, num_runs, True)
    # Run simulations for the baseline (non-adaptive) method
    mean_base, std_base = run_multiple_simulations(rho_initial, rho_target, dt, n_steps,
                                                     k_fb, gamma, noise_strength, num_runs, False)

    times = np.linspace(0, n_steps*dt, n_steps)

    # Plotting: mean fidelity with error bands (mean +/- std)
    plt.figure(figsize=(10, 6))

    plt.plot(times, mean_fb, 'b-', lw=2, label='Feedback Simulation')
    plt.fill_between(times, mean_fb - std_fb, mean_fb + std_fb, color='b', alpha=0.3)

    plt.plot(times, mean_base, 'r--', lw=2, label='Baseline (No Feedback)')
    plt.fill_between(times, mean_base - std_base, mean_base + std_base, color='r', alpha=0.3)

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Fidelity to Target State", fontsize=14)
    plt.title("Benchmark: Fidelity Evolution with Statistical Analysis", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('app_benchmark_result.png')
    print("Successfully ran benchmark simulation with the following parameters:")
    print(f"  Time step (dt): {dt}")
    print(f"  Number of steps: {n_steps}")
    print(f"  Feedback strength (k_fb): {k_fb}")
    print(f"  Dephasing noise (gamma): {gamma}")
    print(f"  Stochastic noise: {noise_strength}")
    print(f"  Number of runs: {num_runs}")
    print("\nResults summary:")
    print(f"  Final fidelity with feedback: {mean_fb[-1]:.4f} ± {std_fb[-1]:.4f}")
    print(f"  Final fidelity without feedback: {mean_base[-1]:.4f} ± {std_base[-1]:.4f}")
    print(f"  Improvement: {(mean_fb[-1] - mean_base[-1]):.4f} ({(mean_fb[-1] - mean_base[-1])/mean_base[-1]*100:.1f}%)")
    print("\nPlot saved as 'app_benchmark_result.png'")

if __name__ == "__main__":
    main()