"""Simple PID simulation helper (plot-only step).

Plots available impressions per hour and a linear delivery plan
that spreads `budget` equally over `runtime_hours`.
"""
from pathlib import Path

import csv
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PID:
    Kp: float
    Ki: float
    Kd: float
    dt: float

    integral: float = 0.0
    prev_error: float = 0.0
    derivative: float = 0.0
    current_error: float = 0.0

    def update(self, error: float) -> float:
        self.integral += error * self.dt
        self.derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        self.current_error = error
        out = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative
        self.prev_error = error
        return out


def simulate_pid_with_opps_schedule(
    budget: int,
    runtime_hours: float,
    opps_schedule: List[float],
    pid_params: Tuple[float, float, float] = (0.8, 0.05, 0.01),
    stochastic: bool = False,
):
    """Simulate PID-controlled selection over an arbitrary opps schedule.

    Returns (hours, opps, desired, prob, selected, cumulative_selected)
    """
    steps = len(opps_schedule)
    dt = runtime_hours / steps

    pid = PID(Kp=pid_params[0], Ki=pid_params[1], Kd=pid_params[2], dt=dt)

    hours = []
    opps = []
    desired_list = []
    probs = []
    selected_list = []
    cumulative = []

    current_selected = 0.0
    rng = np.random.default_rng(42)

    for i in range(steps):
        t = (i + 1) * dt
        available = float(opps_schedule[i])

        target_frac = t / runtime_hours
        target_selected = target_frac * budget

        error = target_selected - current_selected

        pid_out = pid.update(error)

        # Desired impressions this step (bounded by available)
        desired = max(0.0, min(available, pid_out))

        prob = desired / available if available > 0 else 0.0
        prob = max(0.0, min(1.0, prob))

        if stochastic:
            selected = float(rng.binomial(int(round(available)), prob))
        else:
            selected = prob * available

        current_selected += selected

        hours.append(t)
        opps.append(available)
        desired_list.append(desired)
        probs.append(prob)
        selected_list.append(selected)
        cumulative.append(current_selected)

    return hours, opps, desired_list, probs, selected_list, cumulative


def simulate_pid_with_opps_schedule_sigmoid(
    budget: int,
    runtime_hours: float,
    opps_schedule: List[float],
    theta: float = 0.2,
    stochastic: bool = False,
):
    """Simulate selection following the sigmoid target curve.

    Target cumulative (fraction): y = 1 - (1 - x) * exp(-theta * x), x = t / runtime_hours
    Probability is computed per-step so selected impressions track the sigmoid curve.
    Returns (hours, opps, desired, prob, selected, cumulative)
    """
    steps = len(opps_schedule)
    dt = runtime_hours / steps

    hours = []
    opps = []
    desired_list = []
    probs = []
    selected_list = []
    cumulative = []

    current_selected = 0.0
    rng = np.random.default_rng(42)

    for i in range(steps):
        t = (i + 1) * dt
        available = float(opps_schedule[i])

        x_norm = t / runtime_hours
        target_sigmoid = budget * (1 - (1 - x_norm) * np.exp(-theta * x_norm))
        # linear cumulative target for reference (not used for probability calc)
        target_lin = (t / runtime_hours) * budget

        # desired to meet sigmoid target at this time
        desired = max(0.0, target_sigmoid - current_selected)
        # clamp desired to available impressions this step
        desired_step = min(desired, available)

        prob = desired_step / available if available > 0 else 0.0
        prob = max(0.0, min(1.0, prob))

        if stochastic:
            selected = float(rng.binomial(int(round(available)), prob))
        else:
            selected = prob * available

        current_selected += selected

        hours.append(t)
        opps.append(available)
        desired_list.append(desired_step)
        probs.append(prob)
        selected_list.append(selected)
        cumulative.append(current_selected)

    return hours, opps, desired_list, probs, selected_list, cumulative


def save_csv(path: str, hours, opps, desired, probs, selected, cumulative):
    out = Path(path)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour", "available", "desired", "prob", "selected", "cumulative"])
        for h, a, d, p, s, c in zip(hours, opps, desired, probs, selected, cumulative):
            writer.writerow([f"{h:.6f}", f"{a:.6f}", f"{d:.6f}", f"{p:.6f}", f"{s:.6f}", f"{c:.6f}"])
    print(f"Wrote CSV to {out.absolute()}")


def plot_pid_results(
    hours,
    opps,
    probs,
    cumulative,
    budget,
    runtime_hours,
    out_path: str,
    title: str = "PID Tracking - Linear Target",
):
    times = np.array(hours)
    opps_arr = np.array(opps)
    probs_arr = np.array(probs)
    cum_arr = np.array(cumulative)

    target_frac = times / runtime_hours
    target_selected = target_frac * budget

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(times, probs_arr, label="Selection Probability", color="C0", marker="o")
    ax1.set_ylabel("Probability")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(times, cum_arr, label="Cumulative Selected", color="C1")
    ax2.plot(times, target_selected, label="Target Cumulative", color="C2", linestyle="--")
    ax2.set_ylabel("Cumulative Selections")
    ax2.set_xlabel("Time (hours)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    ax3 = ax2.twinx()
    ax3.fill_between(times, 0, opps_arr, color="C4", alpha=0.12)
    ax3.plot(times, opps_arr, color="C4", alpha=0.6, label="Available Impressions")
    ax3.set_ylabel("Available Impressions")
    ax3.set_ylim(0, opps_arr.max() * 1.1)

    if title:
        fig.suptitle(title)

    out = Path(out_path)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved plot to {out.absolute()}")


def plot_pid_results_sigmoid(
    hours,
    opps,
    probs,
    cumulative,
    budget,
    runtime_hours,
    theta,
    out_path: str,
    title: str = "PID Tracking - Sigmoid Target",
):
    times = np.array(hours)
    opps_arr = np.array(opps)
    probs_arr = np.array(probs)
    cum_arr = np.array(cumulative)

    # Sigmoid target: y = 1 - (1 - x) * exp(-theta * x)
    x_norm = times / runtime_hours
    target_selected = budget * (1 - (1 - x_norm) * np.exp(-theta * x_norm))

    target_linear = x_norm * budget

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(times, probs_arr, label="Selection Probability", color="C0", marker="o")
    ax1.set_ylabel("Probability")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(times, cum_arr, label="Cumulative Selected", color="C1")
    ax2.plot(times, target_linear, label="Target Cumulative (Linear)", color="C2", linestyle="--")
    ax2.set_ylabel("Cumulative Selections")
    ax2.set_xlabel("Time (hours)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left")

    ax3 = ax2.twinx()
    ax3.fill_between(times, 0, opps_arr, color="C4", alpha=0.12)
    ax3.plot(times, opps_arr, color="C4", alpha=0.6, label="Available Impressions")
    ax3.set_ylabel("Available Impressions")
    ax3.set_ylim(0, opps_arr.max() * 1.1)

    if title:
        fig.suptitle(title)

    out = Path(out_path)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved plot to {out.absolute()}")


def plot_hourly_impressions_comparison(
    hours,
    selected_pid,
    selected_sigmoid,
    out_path: str,
    title: str = "Hourly Selected Impressions Comparison",
):
    """Plot hourly impressions for both PID and Sigmoid methods side by side."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(hours))
    width = 0.35
    
    ax.bar(x - width/2, selected_pid, width, label="PID", alpha=0.8)
    ax.bar(x + width/2, selected_sigmoid, width, label="Sigmoid", alpha=0.8)
    
    ax.set_xlabel("Hour")
    ax.set_ylabel("Selected Impressions")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}" for h in hours])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    out = Path(out_path)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved hourly comparison plot to {out.absolute()}")


def main():
    # Extend campaign to 24 hours with total target 24,000
    budget = 24_000
    runtime_hours = 24

    # Base hourly pattern (10-hour pattern from previous run)
    base_pattern = [700, 900, 900, 1300, 1300, 1400, 1400, 1300, 1300, 900, 900, 700]
    repeats = (runtime_hours + len(base_pattern) - 1) // len(base_pattern)
    available = (base_pattern * repeats)[:runtime_hours]

    # Quick visual: available and naive linear per-hour delivery
    hours = np.arange(1, runtime_hours + 1)
    delivery_per_hour = budget / runtime_hours

    delivery = [delivery_per_hour] * len(available)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(hours, available, width=0.6, alpha=0.6, label="Available impressions")
    ax.plot(hours, delivery, color="C1", marker="o", linewidth=2, label="Planned delivery (linear)")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Impressions")
    ax.set_xticks(hours)
    ax.set_title("Available impressions and linear delivery (24,000 over 24 hours)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = Path("pid_simulation.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved plot to {out.absolute()}")

    # Run PID simulation on the same schedule
    pid_params = (1.0472, 0.00, 0.2742)
    hours_s, opps_s, desired_s, probs_s, selected_s, cum_s = simulate_pid_with_opps_schedule(
        budget=budget,
        runtime_hours=runtime_hours,
        opps_schedule=available,
        pid_params=pid_params,
        stochastic=False,
    )

    save_csv("pid_simulation_pid.csv", hours_s, opps_s, desired_s, probs_s, selected_s, cum_s)
    plot_pid_results(hours_s, opps_s, probs_s, cum_s, budget, runtime_hours, out_path="pid_simulation_pid.png")

    # Run sigmoid simulation
    theta = 0.4
    hours_s2, opps_s2, desired_s2, probs_s2, selected_s2, cum_s2 = simulate_pid_with_opps_schedule_sigmoid(
        budget=budget,
        runtime_hours=runtime_hours,
        opps_schedule=available,
        theta=theta,
        stochastic=False,
    )

    save_csv("pid_simulation_sigmoid.csv", hours_s2, opps_s2, desired_s2, probs_s2, selected_s2, cum_s2)
    plot_pid_results_sigmoid(hours_s2, opps_s2, probs_s2, cum_s2, budget, runtime_hours, theta=theta, out_path="pid_simulation_sigmoid.png")

    # Plot hourly impressions comparison
    plot_hourly_impressions_comparison(hours, selected_s, selected_s2, out_path="hourly_impressions_comparison.png")

    print(f"Sigmoid final selected = {cum_s2[-1]:.2f} (target {budget})")
    print(f"PID final selected = {cum_s[-1]:.2f} (target {budget})")


if __name__ == "__main__":
    main()
