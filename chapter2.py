from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi
from scipy.interpolate import interp1d
from tueplots.constants.color import palettes

tue_col_1 = palettes.tue_plot[0]
tue_col_2 = palettes.tue_plot[1]

# ----------------------------------
# CPU Problem
# ----------------------------------


@dataclass
class CPUHeatProblem1D:
    """
    Encapsulates the 1D CPU heat problem, now independent of linpde_gp.
    The ground truth solution is computed numerically.
    """

    # Geometry
    width: float
    domain: tuple[float, float]

    # Material & Heat Properties
    kappa: float
    TDP: float

    # Ground Truth Solution (numerically computed)
    solution: Callable[[np.ndarray], np.ndarray]

    # PDE right-hand-side function
    q_total: Callable[[np.ndarray], np.ndarray]

    # Synthetic DTS sensor data
    X_dts: np.ndarray
    y_dts: np.ndarray
    dts_noise_std: float


def create_cpu_problem() -> CPUHeatProblem1D:
    """
    Factory function to set up the CPU heat problem using NumPy and SciPy.
    """
    # -- Geometry
    width, height, depth = 16.28, 9.19, 0.37
    domain = (0.0, width)
    V = width * height * depth

    # -- Material property
    kappa = 1.56 * 10.0

    # -- Heat sources
    TDP = 95.0
    N_cores_x, core_width, core_offset_x, core_distance_x = 3, 2.5, 1.95, 0.35
    core_centers_xs = (
        core_offset_x
        + (core_width + core_distance_x) * np.arange(N_cores_x, dtype=np.double)
        + core_width / 2.0
    )

    # -- Heat source function (re-implemented with NumPy)
    def create_core_heat_source_fn(rel_heights=[0.9, 0.75, 1.0]):
        """Creates a callable heat source function using interpolation."""
        xs, ys = [domain[0]], [0.0]
        eps = core_distance_x / 3
        for cx, h in zip(core_centers_xs, rel_heights):
            xs.extend(
                [
                    cx - core_width / 2 - eps,
                    cx - core_width / 2,
                    cx + core_width / 2,
                    cx + core_width / 2 + eps,
                ]
            )
            ys.extend([0.0, h, h, 0.0])
        xs.append(domain[1])
        ys.append(0.0)

        # Normalize the distribution using the trapezoidal rule
        norm_const = np.trapz(ys, xs)
        ys_normalized = np.array(ys) / norm_const

        # Return a callable interpolation function
        return interp1d(xs, ys_normalized, bounds_error=False, fill_value=0.0)

    q_src_dist = create_core_heat_source_fn()

    def q_dot_V_src(x):
        return (TDP / (depth * height)) * q_src_dist(x)

    q_dot_V_sink = -TDP / V

    def q_total(x):
        return q_dot_V_src(x) + q_dot_V_sink

    # -- Numerical Ground Truth Solution --
    def create_numerical_solution(u0=60.0, du0=0.0):
        """Solves -k u'' = q_total for u using numerical integration."""
        x_dense = np.linspace(domain[0], domain[1], 2001)

        # We solve u'' = -q_total(x) / kappa
        rhs = -q_total(x_dense) / kappa
        du_dx = spi.cumulative_trapezoid(rhs, x_dense, initial=0.0) + du0
        u_values = spi.cumulative_trapezoid(du_dx, x_dense, initial=0.0) + u0

        # Return a callable interpolation function
        return interp1d(
            x_dense,
            u_values,
            bounds_error=False,
            fill_value=(u_values[0], u_values[-1]),
        )

    solution_fn = create_numerical_solution(u0=60.0, du0=0.0)

    # -- Synthetic DTS Data --
    dts_noise_std = 0.5
    noise_rng = np.random.default_rng(33215)
    noise_dts = noise_rng.normal(scale=dts_noise_std, size=len(core_centers_xs))
    y_dts = solution_fn(core_centers_xs) + noise_dts

    return CPUHeatProblem1D(
        width=width,
        domain=domain,
        kappa=kappa,
        TDP=TDP,
        solution=solution_fn,
        q_total=q_total,
        X_dts=core_centers_xs,
        y_dts=y_dts,
        dts_noise_std=dts_noise_std,
    )


# -----------------------------
# Plotting utilities
# -----------------------------


def plot_gp_belief_and_pde(
    gp,
    problem: CPUHeatProblem1D,
    X_grid: np.ndarray,
    conditions: list | None = None,
    n_samples: int = 0,
    seed: int = 0,
    title: str | None = None,
):
    """
    Generates a two-panel plot, now with support for visualizing integral conditions.
    """
    fig, (ax_u, ax_pde) = plt.subplots(
        ncols=2, sharex=True, figsize=(12, 3.5), constrained_layout=True
    )
    if conditions is None:
        conditions = []

    # -- Prepare GP data for plotting --
    X_plot = np.asarray(X_grid).reshape(-1, 1)
    gp_eval = gp(jnp.asarray(X_plot))
    mu = np.asarray(gp_eval.mu).flatten()
    Sigma = np.asarray(gp_eval.Sigma)
    std = np.sqrt(Sigma.diagonal())

    # --- 1. Left Panel: Belief over Temperature u(x) ---
    ax_u.set(
        ylabel="Temperature (°C)",
        title="Belief over Temperature $u(x)$",
        xlim=problem.domain,
    )
    ax_u.grid(True)
    ax_u.plot(
        X_grid, problem.solution(X_grid), color="k", lw=2, ls="-", label="True Solution"
    )
    ax_u.fill_between(
        X_grid,
        mu - 1.96 * std,
        mu + 1.96 * std,
        color="C0",
        alpha=0.2,
        label="95% Credible Interval",
    )
    ax_u.plot(X_grid, mu, color="C0", lw=2, label="GP Mean")
    if n_samples > 0:
        samples = gp_eval.sample(jax.random.key(seed), n_samples)
        ax_u.plot(X_grid, samples.T, color="C0", lw=1.0, alpha=0.5)

    # --- 2. Right Panel: PDE Balance Check ---
    ax_pde.set(
        xlabel="x-position (mm)",
        ylabel="Heat Flow",
        title=r"PDE Balance: $-\kappa u''(x)$ vs $\dot{q}_V(x)$",
    )
    ax_pde.grid(True)
    ax_pde.plot(
        X_grid,
        problem.q_total(X_grid),
        color="C1",
        lw=2,
        label=r"Heat Source $\dot{q}_V(x)$",
    )

    h = np.median(np.diff(X_grid))
    d2_mu = (np.roll(mu, -1) - 2 * mu + np.roll(mu, 1)) / h**2
    d2_mu[[0, -1]] = d2_mu[[1, -2]]
    mean_Lu = -problem.kappa * d2_mu
    N = len(X_grid)
    D = np.zeros((N, N))
    D[range(1, N - 1), range(0, N - 2)] = 1
    D[range(1, N - 1), range(1, N - 1)] = -2
    D[range(1, N - 1), range(2, N)] = 1
    D /= h**2
    D[0, 0:3] = [1, -2, 1]
    D[-1, -3:] = [1, -2, 1]
    Sigma_d2 = D @ Sigma @ D.T
    var_d2 = np.diag(Sigma_d2)
    std_Lu = problem.kappa * np.sqrt(np.maximum(var_d2, 0))
    ax_pde.plot(X_grid, mean_Lu, color="C0", lw=2, label=r"GP-implied $-\kappa u''(x)$")
    ax_pde.fill_between(
        X_grid, mean_Lu - 1.96 * std_Lu, mean_Lu + 1.96 * std_Lu, color="C0", alpha=0.2
    )

    if n_samples > 0:
        d2_samples = np.array(
            [
                (np.roll(sample, -1) - 2 * sample + np.roll(sample, 1)) / h**2
                for sample in samples
            ]
        )
        d2_samples[:, [0, -1]] = d2_samples[:, [1, -2]]
        Lu_samples = -problem.kappa * d2_samples
        ax_pde.plot(
            X_grid,
            Lu_samples.T,
            color="C0",
            lw=1.0,
            alpha=0.5,
        )

    # -- Plot all provided conditions --
    for cond in conditions:
        cond_name = type(cond.op).__name__
        if cond_name == "PDEObservation":
            ax_pde.scatter(
                cond.X,
                cond.y_vec,
                s=40,
                marker="x",
                color=tue_col_1,
                label="PDE Collocation Points",
                zorder=5,
            )
        elif cond_name == "SensorObservation":
            ax_u.errorbar(
                cond.X,
                cond.y_vec,
                yerr=np.sqrt(cond.op.sigma2),
                fmt="o",
                color=tue_col_2,
                label="Sensor Observations",
                zorder=5,
            )
            ax_pde.scatter(
                cond.X,
                np.zeros_like(cond.X),
                marker="o",
                color=tue_col_2,
                s=40,
                label="Sensor Locations",
            )
        elif cond_name == "BoundaryObservation":
            ax_u.errorbar(
                cond.X,
                cond.y_vec,
                yerr=np.sqrt(cond.op.sigma2),
                fmt="o",
                color=tue_col_2,
                label="Dirichlet Condition",
                zorder=5,
            )

        else:
            print(f"Warning: Unknown condition type for plotting: {cond_name}")

    ax_u.legend(loc="lower left")
    ax_pde.legend(loc="upper right")
    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()
