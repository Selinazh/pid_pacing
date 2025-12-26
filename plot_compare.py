"""Plot linear vs exponential delivery patterns and save comparison image.

Budget: 10,000
Runtime: 10 hours

Exponential formula: f(t) = 1 - exp(-theta * t)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_theta_for_epsilon(epsilon: float, runtime_hours: float) -> float:
	"""Return theta such that exp(-theta * runtime_hours) = epsilon.

	This makes f(runtime_hours) = 1 - epsilon (i.e. within `epsilon` of full delivery).
	"""
	if not (0 < epsilon < 1):
		raise ValueError("epsilon must be between 0 and 1")
	return -np.log(epsilon) / runtime_hours


def plot_compare_linear_vs_exp(budget=10_000, runtime_hours=10, theta=0.2, n_points=201, out_path="ideal_vs_exp.png", normalize_exp=True):
	"""Plot linear (ideal) vs the custom exponential-like approach.

	Formula used (with x normalized to [0,1]):
		y = 1 - (1 - x) * exp(-theta * x)

	Here `times` are in hours; we compute x = times / runtime_hours.
	The resulting fraction `y` is multiplied by `budget` to get cumulative spend.
	The `normalize_exp` parameter is accepted for backward compatibility but is
	ignored because this formula already yields y(1)=1 (full delivery at runtime).
	"""
	times = np.linspace(0, runtime_hours, n_points)

	frac_linear = times / runtime_hours
	x_norm = frac_linear
	# new formula: y = 1 - (1 - x) * exp(-theta * x)
	frac_exp = 1 - (1 - x_norm) * np.exp(-theta * x_norm)

	spends_linear = frac_linear * budget
	spends_exp = frac_exp * budget

	fig, ax = plt.subplots(figsize=(9, 6))
	ax.plot(times, spends_linear, label="Linear (ideal)", color="C0", lw=2)
	ax.plot(times, spends_exp, label=(f"Exp approach (theta={theta}, normalized)" if normalize_exp else f"Exp approach (theta={theta})"), color="C2", lw=2)

	# Mark start/end (for reference)
	ax.scatter([0, runtime_hours], [0, budget], color="k", s=20)

	ax.set_title("Delivery Patterns — Cumulative Spend vs Time")
	ax.set_xlabel("Time (hours)")
	ax.set_ylabel("Cumulative Spend (currency)")
	ax.grid(alpha=0.25)
	ax.legend()

	out = Path(out_path)
	fig.tight_layout()
	fig.savefig(out)
	plt.close(fig)
	print(f"Saved comparison plot to {out.absolute()}")


if __name__ == "__main__":
	plot_compare_linear_vs_exp()
