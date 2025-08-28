# UTILITY FUNCTIONS FOR CHAPTER 1

import ipywidgets as widgets
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation

# ---------------------------------
# % Animation of damped oscillator
# ---------------------------------


def animate_damped_oscillator(
    damped_oscillator_fn,
    t_anim,
    theta,
    fps=24,
    n_coils=12,
    amp=0.11,
    trace_len=200,
):
    y_anim = damped_oscillator_fn(t_anim, theta)
    t_anim_np = np.asarray(t_anim)
    y_anim_np = np.asarray(y_anim)

    def _spring_points(
        x_left: float, x_right: float, n_coils: int = n_coils, amp: float = amp
    ):
        xs = np.linspace(x_left, x_right, n_coils * 2 + 1)
        ys = np.zeros_like(xs)
        ys[1:-1:2] = amp
        ys[2:-1:2] = -amp
        ys[0] = ys[-1] = 0.0
        return xs, ys

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.suptitle("Damped Harmonic Oscillator — Mass Motion & Displacement", fontsize=14)

    A = theta.A
    ax_left.set_xlim(-A * 1.8 - 0.4, A * 1.8 + 0.4)
    ax_left.set_ylim(-0.8, 0.8)
    ax_left.set_xlabel("Position")
    ax_left.set_yticks([])
    ax_left.grid(True, axis="x")
    wall_x = -A * 1.8 - 0.2
    ax_left.plot([wall_x, wall_x], [-0.8, 0.8], linewidth=4)

    mass_w, mass_h = 0.28, 0.32
    mass = plt.Rectangle(
        (y_anim_np[0] - mass_w / 2, -mass_h / 2), mass_w, mass_h, ec="black", fc="C0"
    )
    ax_left.add_patch(mass)
    (spring_line,) = ax_left.plot([], [], lw=2)
    (trace_line,) = ax_left.plot([], [], lw=1, alpha=0.6)
    trace_buf = []

    ax_right.set_xlim(t_anim_np[0], t_anim_np[-1])
    ax_right.set_ylim(-A * 1.2, A * 1.2)
    ax_right.set_xlabel("Time t")
    ax_right.set_ylabel("Displacement y(t)")
    ax_right.grid(True)
    ax_right.plot(t_anim_np, y_anim_np, alpha=0.4, label="True displacement")
    (marker_dot,) = ax_right.plot([], [], "o", label="Current position")
    y0, y1 = ax_right.get_ylim()
    (time_line,) = ax_right.plot(
        [t_anim_np[0], t_anim_np[0]], [y0, y1], linestyle="--", alpha=0.6
    )
    ax_right.legend(loc="upper right")

    def init():
        spring_line.set_data([], [])
        trace_line.set_data([], [])
        marker_dot.set_data([], [])
        y0, y1 = ax_right.get_ylim()
        time_line.set_data([t_anim_np[0], t_anim_np[0]], [y0, y1])
        return mass, spring_line, trace_line, marker_dot, time_line

    def animate(i):
        x_pos = y_anim_np[i]
        mass.set_xy((x_pos - mass_w / 2, -mass_h / 2))
        xs, ys = _spring_points(wall_x, x_pos - mass_w / 2)
        spring_line.set_data(xs, ys)
        trace_buf.append(x_pos)
        if len(trace_buf) > trace_len:
            del trace_buf[0]
        trace_line.set_data(np.array(trace_buf), np.zeros(len(trace_buf)))
        marker_dot.set_data([t_anim_np[i]], [y_anim_np[i]])
        y0, y1 = ax_right.get_ylim()
        time_line.set_data([t_anim_np[i], t_anim_np[i]], [y0, y1])
        return mass, spring_line, trace_line, marker_dot, time_line

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(t_anim_np),
        interval=1000 / fps,
        blit=True,
    )
    plt.close(fig)
    return HTML(anim.to_jshtml())


# ---------------------------------------
# % Plotting GPs and interactive widgets
# ---------------------------------------

plt.ioff()


def plot_gp(
    ax,
    gp_or_dist,
    X_grid,
    *,
    n_samples: int = 0,
    seed: int = 0,
    mean_label: str = "Mean",
    shade_k: float = 2.0,
    shade_label: str = "\pm2\sigma",
    true_curve=None,  # (X_true, y_true)
    obs=None,  # (X_obs, y_obs)
    obs_style=None,
    sample_style=None,
    mean_style=None,
    band_alpha: float = 0.2,
):
    """
    Draws GP mean, ±ksigma band, optional samples, optional true curve, and observations.
    Works with either a GP (callable returning a dist) or a distribution directly.
    """
    # Get a distribution at X_grid
    if hasattr(gp_or_dist, "mu") and hasattr(gp_or_dist, "Sigma"):
        dist = gp_or_dist
    else:
        dist = gp_or_dist(X_grid)  # assume GP-like callable

    Xr = np.asarray(X_grid).ravel()
    mu = np.asarray(dist.mu).reshape(-1)
    std = np.sqrt(np.asarray(dist.Sigma).diagonal())

    # Styles
    if mean_style is None:
        mean_style = dict(lw=2, label=mean_label)
    if sample_style is None:
        sample_style = dict(lw=1.0, alpha=0.8)
    if obs_style is None:
        obs_style = dict(s=60, marker="x", color="red", zorder=5, label="Observed")

    # Plot uncertainty band + mean
    ax.fill_between(
        Xr, mu - shade_k * std, mu + shade_k * std, alpha=band_alpha, label=shade_label
    )
    ax.plot(Xr, mu, **mean_style)

    # Optional samples
    if n_samples > 0:
        S = dist.sample(jax.random.PRNGKey(seed), num_samples=n_samples)
        ax.plot(Xr, np.asarray(S).T, **sample_style)

    # Optional ground truth
    if true_curve is not None:
        X_true, y_true = true_curve
        X_true, y_true = np.asarray(X_true).ravel(), np.asarray(y_true).ravel()
        ax.plot(
            X_true,
            y_true,
            lw=2,
            alpha=0.7,
            color="black",
            linestyle="--",
            label="True Signal",
        )

    # Optional observations
    if obs is not None:
        X_obs, y_obs = obs
        X_obs, y_obs = np.asarray(X_obs).ravel(), np.asarray(y_obs).ravel()
        ax.scatter(X_obs, y_obs, **obs_style)

    return mu, std  # handy for metrics etc.


class InteractiveGPPlotter:
    def __init__(
        self,
        plot_type: str,
        *,
        kernel_registry,
        gp_cls=None,
        mean_fn=None,
        training_data=None,
        test_data=None,
    ):
        assert plot_type in {"kernel", "prior", "posterior"}
        self.plot_type = plot_type
        self.kernel_registry = kernel_registry
        self.gp_cls = gp_cls
        self.mean_fn = mean_fn

        self.X_train, self.y_train = training_data if training_data else (None, None)
        self.X_test, self.y_test = test_data if test_data else (None, None)

        if self.plot_type == "posterior":
            if any(
                v is None
                for v in (self.X_train, self.y_train, self.X_test, self.y_test)
            ):
                raise ValueError("posterior plot requires training_data and test_data.")

        self.widgets = self._create_widgets()
        self.controls = self._layout_widgets()

        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.ui = widgets.VBox([self.controls, self.fig.canvas])

        self._link_widgets_to_redraw()
        self.redraw()

    def _create_widgets(self) -> dict[str, widgets.Widget]:
        style = {"description_width": "110px"}
        w = {
            "kernel": widgets.Dropdown(
                options=list(self.kernel_registry.keys()),
                value=next(iter(self.kernel_registry.keys())),
                description="Kernel",
                style=style,
            ),
            "variance": widgets.FloatLogSlider(
                value=1.0, base=10, min=-2, max=2, description="Variance", style=style
            ),
            "lengthscale": widgets.FloatLogSlider(
                value=1.0,
                base=10,
                min=-2,
                max=2,
                description="Lengthscale",
                style=style,
            ),
            "period": widgets.FloatSlider(
                value=np.pi,
                min=0.2,
                max=8.0,
                step=0.05,
                description="Period",
                style=style,
            ),
        }
        if self.plot_type == "kernel":
            w["xmax"] = widgets.FloatSlider(
                value=5.0, min=1.0, max=15, step=0.1, description="x_max", style=style
            )
        elif self.plot_type == "prior":
            w.update(
                {
                    "samples": widgets.IntSlider(
                        value=5, min=0, max=30, description="# Samples", style=style
                    ),
                    "seed": widgets.IntSlider(
                        value=0, min=0, max=100, description="Seed", style=style
                    ),
                    "xmin": widgets.FloatSlider(
                        value=-5.0,
                        min=-15,
                        max=0,
                        step=0.1,
                        description="x_min",
                        style=style,
                    ),
                    "xmax": widgets.FloatSlider(
                        value=5.0,
                        min=0,
                        max=15,
                        step=0.1,
                        description="x_max",
                        style=style,
                    ),
                }
            )
        else:
            w.update(
                {
                    "noise": widgets.FloatLogSlider(
                        value=0.1,
                        base=10,
                        min=-3,
                        max=0,
                        description="σ_noise",
                        style=style,
                    ),
                    "samples": widgets.IntSlider(
                        value=3, min=0, max=30, description="# Samples", style=style
                    ),
                    "seed": widgets.IntSlider(
                        value=0, min=0, max=100, description="Seed", style=style
                    ),
                    "n_used": widgets.IntSlider(
                        value=min(10, len(self.X_train)),
                        min=0,
                        max=len(self.X_train),
                        description="# Points",
                        style=style,
                    ),
                    "metrics": widgets.HTML(
                        value="<pre style='margin:0'>RMSE: –\nMLPD: –</pre>"
                    ),
                }
            )
        return w

    def _layout_widgets(self) -> widgets.Widget:
        w = self.widgets
        kernel_box = widgets.VBox(
            [w["kernel"], w["variance"], w["lengthscale"], w["period"]]
        )
        if self.plot_type == "kernel":
            return widgets.HBox([kernel_box, widgets.VBox([w["xmax"]])])
        if self.plot_type == "prior":
            sampling_box = widgets.VBox([w["samples"], w["xmin"], w["xmax"], w["seed"]])
            return widgets.HBox([kernel_box, sampling_box])
        data_box = widgets.VBox([w["noise"], w["n_used"], w["samples"], w["seed"]])
        metrics_box = widgets.VBox([widgets.HTML("<b>Metrics</b>"), w["metrics"]])
        return widgets.HBox([kernel_box, data_box, metrics_box])

    def _link_widgets_to_redraw(self):
        for name, widget in self.widgets.items():
            if name == "metrics":
                continue
            widget.observe(self.redraw, names="value")

    def _get_kernel(self):
        KernelClass = self.kernel_registry[self.widgets["kernel"].value]
        params = {
            "variance": float(self.widgets["variance"].value),
            "lengthscale": float(self.widgets["lengthscale"].value),
            "period": float(self.widgets["period"].value),
        }
        # show/hide period control if the kernel suggests it
        show_period = (
            "period" in KernelClass.__name__.lower()
            or "periodic" in self.widgets["kernel"].value.lower()
        )
        self.widgets["period"].layout.display = "" if show_period else "none"
        return KernelClass.from_params(params)

    @staticmethod
    def _compute_metrics(mu, std, y_true):
        eps = 1e-12
        rmse = float(np.sqrt(np.mean((mu - y_true) ** 2)))
        var = std**2 + eps
        mlpd = float(
            np.mean(-0.5 * np.log(2 * np.pi * var) - 0.5 * ((y_true - mu) ** 2) / var)
        )
        return rmse, mlpd

    # --- redraw
    def redraw(self, *_):
        ax = self.ax
        ax.cla()

        if self.plot_type == "kernel":
            self._plot_kernel(ax)
        elif self.plot_type == "prior":
            self._plot_prior(ax)
        else:
            self._plot_posterior(ax)

        ax.grid(True, alpha=0.5)
        ax.legend(loc="upper right")
        self.fig.canvas.draw_idle()

    # --- individual plot modes
    def _plot_kernel(self, ax: plt.Axes):
        w = self.widgets
        kernel = self._get_kernel()
        xmax = float(w["xmax"].value)
        X = jnp.linspace(-xmax, xmax, 500)[:, None]
        ax.plot(X, kernel(X, jnp.array([[0.0]])), lw=2, label="k(x, 0)")
        ax.set(
            ylim=(-0.1, float(w["variance"].value) * 1.1),
            title=f"'{w['kernel'].value}' Kernel Function",
            xlabel="x",
            ylabel="k(x,0)",
        )

    def _plot_prior(self, ax: plt.Axes):
        w = self.widgets
        kernel = self._get_kernel()
        gp = self.gp_cls(m=self.mean_fn, k=kernel)
        X_grid = jnp.linspace(float(w["xmin"].value), float(w["xmax"].value), 400)[
            :, None
        ]
        mu, std = plot_gp(
            ax,
            gp,
            X_grid,
            n_samples=int(w["samples"].value),
            seed=int(w["seed"].value),
            mean_label="Prior Mean",
            shade_k=2.0,
            shade_label="±2σ",
        )
        ax.set(title=f"GP Prior — {w['kernel'].value}", xlabel="x", ylabel="f(x)")

    def _plot_posterior(self, ax: plt.Axes):
        w = self.widgets
        kernel = self._get_kernel()
        gp = self.gp_cls(m=self.mean_fn, k=kernel)

        X_grid = jnp.linspace(float(self.X_test.min()), float(self.X_test.max()), 400)[
            :, None
        ]
        dist = gp(X_grid)

        n = int(w["n_used"].value)
        obs = None
        if n > 0:
            X_obs = self.X_train[:n]
            y_obs = self.y_train[:n]

            gp = gp.condition(
                y=jnp.asarray(y_obs),
                X=jnp.asarray(X_obs),
                sigma2=float(w["noise"].value) ** 2,
            )
            dist = gp(X_grid)
            obs = (X_obs, y_obs)

        mu, std = plot_gp(
            ax,
            dist,
            X_grid,
            n_samples=int(w["samples"].value),
            seed=int(w["seed"].value),
            mean_label="Posterior Mean",
            true_curve=(self.X_test, self.y_test),
            obs=obs,
        )
        # metrics
        y_interp = np.interp(
            np.asarray(X_grid).ravel(), self.X_test.ravel(), self.y_test.ravel()
        )
        rmse, mlpd = self._compute_metrics(mu, std, y_interp)
        self.widgets[
            "metrics"
        ].value = f"<pre style='margin:0'>RMSE: {rmse:.4f}\nMLPD: {mlpd:.4f}</pre>"
        ax.set(title=f"GP Posterior — {w['kernel'].value}", xlabel="x", ylabel="f(x)")


# ------------------------------------------------
# Optimization widget
# ------------------------------------------------


class OptimizationWidget:
    """Manages the UI for fitting and plotting."""

    def __init__(
        self,
        tuner,
        kernel_registry: dict,
        param_init_registry: dict,
        test_data: tuple,
        gp_cls,
        extrapolate: float | bool = False,
    ):
        self.tuner = tuner
        self.gp_cls = gp_cls
        self.kernel_registry = kernel_registry
        self.param_init_registry = param_init_registry
        self.X_test, self.y_test = test_data

        self.kernel_sel = widgets.Dropdown(
            options=list(self.kernel_registry.keys()), description="Kernel"
        )
        self.fit_button = widgets.Button(
            description="Fit Hyperparameters", button_style="success"
        )

        self.fig, self.ax = plt.subplots(figsize=(9, 4.5))
        self.ui = widgets.VBox(
            [widgets.HBox([self.kernel_sel, self.fit_button]), self.fig.canvas]
        )
        self.fit_button.on_click(self._on_fit_clicked)

        # Initial empty plot
        self.ax.text(
            0.5,
            0.5,
            "Select a kernel and click 'Fit' to begin.",
            ha="center",
            va="center",
        )
        self.ax.grid(True)

        self.extrapolate = extrapolate

    def _format_results_text(self, kernel_name, fit_results):
        """Formats the optimization results into a string for plotting."""
        lines = [f"--- Optimized: {kernel_name} ---"]
        final_params = fit_results["final_params"]
        log_params = fit_results["optimized_params_log"]

        for key, val in final_params.items():
            if key in ["noise", "sigma2"]:
                continue
            log_key = "log_" + key if key in ["variance", "lengthscale"] else key
            log_val_str = (
                f"(log={log_params.get(log_key, 'N/A'):.2f})"
                if log_key in log_params
                else ""
            )
            lines.append(f"{key:<12s} = {val:.3f} {log_val_str}")

        lines.append(
            f"{'noise':<12s} = {final_params['noise']:.3f}  (log={log_params['log_noise']:.2f})"
        )
        lines.append(f"Final Neg MLL = {fit_results['result'].fun:.2f}")
        return "\n".join(lines)

    def _on_fit_clicked(self, _):
        # 1. Update the plot to show an "Optimizing..." message
        self.ax.cla()
        self.ax.text(0.5, 0.5, "Optimizing...", ha="center", va="center", fontsize=14)
        self.fig.canvas.draw_idle()

        # 2. Run the optimization
        kernel_name = self.kernel_sel.value
        fit_results = self.tuner.fit(
            kernel_name, self.kernel_registry, self.param_init_registry
        )

        # 3. Build the optimized GP posterior
        final_params = fit_results["final_params"]
        sigma2_opt = final_params["sigma2"]
        k_opt = self.kernel_registry[kernel_name].from_params(final_params)

        gp_prior_opt = self.gp_cls(m=lambda x: jnp.zeros(x.shape[0]), k=k_opt)
        post_gp = gp_prior_opt.condition(
            y=self.tuner.y, X=self.tuner.X, sigma2=sigma2_opt
        )

        # 4. Clear the axes and draw the final plot
        self.ax.cla()
        X_grid = (
            jnp.linspace(
                self.X_test.min() - self.extrapolate,
                self.X_test.max() + self.extrapolate,
                500,
            )[:, None]
            if self.extrapolate
            else self.X_test[:, None]
        )
        plot_gp(
            self.ax,
            post_gp,
            X_grid,
            n_samples=3,
            obs=(self.tuner.X, self.tuner.y),
            true_curve=(self.X_test, self.y_test),
        )
        self.ax.set_title(f"Posterior with Optimized '{kernel_name}' Kernel")

        # 5. Add the formatted results text to the plot
        results_text = self._format_results_text(kernel_name, fit_results)
        self.ax.text(
            0.02,
            0.98,
            results_text,
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontfamily="monospace",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.7),
        )

        self.ax.legend(loc="upper right")
        self.ax.grid(True)
