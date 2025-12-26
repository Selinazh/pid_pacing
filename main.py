"""Generate an ideal ad delivery digraph-like plot.

Budget: 10,000
Runtime: 10 hours

This script draws cumulative spend vs time as a linear line from (0,0) to (1,1)
and adds directed arrows between nodes to represent delivery direction.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_ideal_delivery(budget=10_000, runtime_hours=10, n_nodes=11, out_path="ideal_delivery.png"):
	# nodes evenly spaced from 0..1 (fraction of runtime/budget)
	frac = np.linspace(0, 1, n_nodes)
	times = frac * runtime_hours
	spends = frac * budget

	fig, ax = plt.subplots(figsize=(8, 6))
	ax.plot(times, spends, linestyle="-", color="C0")
	ax.scatter(times, spends, color="C1", zorder=5)

	# Draw directed edges (arrows) between consecutive nodes
	for i in range(len(times) - 1):
		x0, y0 = times[i], spends[i]
		x1, y1 = times[i + 1], spends[i + 1]
		ax.annotate(
			"",
			xy=(x1, y1),
			xytext=(x0, y0),
			arrowprops=dict(arrowstyle="->", color="gray", lw=1),
			size=12,
		)

	ax.set_title("Ideal Ad Delivery (linear) — cumulative spend vs time")
	ax.set_xlabel("Time (hours)")
	ax.set_ylabel("Cumulative Spend (currency)")
	ax.grid(alpha=0.3)

	# Annotate start and end
	ax.annotate("Start", xy=(times[0], spends[0]), xytext=(5, -20), textcoords="offset points")
	ax.annotate("End", xy=(times[-1], spends[-1]), xytext=(5, -20), textcoords="offset points")

	out = Path(out_path)
	fig.tight_layout()
	fig.savefig(out)
	plt.close(fig)
	print(f"Saved ideal delivery plot to {out.absolute()}")


if __name__ == "__main__":
	plot_ideal_delivery()

