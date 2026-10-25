import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
STORM_JSON = os.path.join(RESULTS_DIR, "storm_sweep.json")
AUTOMATON_JSON = os.path.join(RESULTS_DIR, "automaton_sweep.json")
THETA_PSEUDO_CRITICAL = 0.39  # paper's estimated threshold, in units of pi

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


def load(path):
    if not os.path.exists(path):
        print(f"[!] Skipping, not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def p_round_from_shot(p_shot, rounds):
    '''Per-round logical error rate from the per-shot failure probability.'''
    if p_shot >= 0.5:
        return 0.5
    return (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / rounds)) / 2.0


def point(entry):
    '''
    Returns (p_round, error bar, is_upper_limit) for one result entry.
    Zero-failure points are returned as an upper limit (rule of three, ~95%).
    '''
    n, fails, rounds = entry["num_shots"], entry["num_fails"], entry["rounds"]
    if fails == 0:
        return p_round_from_shot(3.0 / n, rounds), 0.0, True
    p_shot = fails / n
    err_shot = np.sqrt(p_shot * (1.0 - p_shot) / n)
    if p_shot < 0.5:
        # d(p_round)/d(p_shot) = (1/rounds) * (1 - 2 p_shot)^(1/rounds - 1)
        err = (1.0 / rounds) * (1.0 - 2.0 * p_shot) ** (1.0 / rounds - 1.0) * err_shot
    else:
        err = err_shot
    return p_round_from_shot(p_shot, rounds), err, False


def group(data, line_of, x_of):
    '''{line value: sorted list of (x, p_round, err, upper_limit)}'''
    lines = {}
    for e in data:
        y, err, ul = point(e)
        lines.setdefault(line_of(e), []).append((x_of(e), y, err, ul))
    return {k: sorted(v) for k, v in lines.items()}


def colors(n):
    return [plt.get_cmap("tab10")(i % 10) for i in range(n)]


def draw(ax, pts, color, marker, linestyle, label):
    x, y, err, ul = (np.array(c) for c in zip(*pts))
    ul = ul.astype(bool)
    ax.plot(x, y, linestyle=linestyle, color=color, linewidth=1, label=label)
    if (~ul).any():
        ax.errorbar(x[~ul], y[~ul], yerr=err[~ul], fmt=marker, color=color, markersize=4, capsize=0)
    if ul.any():
        ax.plot(x[ul], y[ul], "v", markerfacecolor="none", color=color, markersize=5)


def has_upper_limits(groups):
    return any(p[3] for pts in groups.values() for p in pts)


def add_upper_limit_legend_entry(ax):
    ax.plot([], [], "v", markerfacecolor="none", color="gray", label="0 failures (upper limit)")


def add_qubit_axis(ax, distances):
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(distances)
    top.set_xticklabels([str(2 * d ** 2 - 1) for d in distances])
    top.set_xlabel("Total no. qubits")


def save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"[+] Saved {path}")


def label_panels(ax1, ax2):
    ax1.text(-0.12, 1.12, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="bottom")
    ax2.text(-0.12, 1.12, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="bottom")


def plot_storm(data):
    '''Temporal storm: logical error rate vs correlation length (a) and vs distance (b).'''
    by_d = group(data, lambda e: e["distance"], lambda e: e["corr_len"])
    by_xi = group(data, lambda e: e["corr_len"], lambda e: e["distance"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for (d, pts), c in zip(sorted(by_d.items()), colors(len(by_d))):
        draw(ax1, pts, c, "o", "-", f"$d = {d}$")
    for (xi, pts), c in zip(sorted(by_xi.items()), colors(len(by_xi))):
        draw(ax2, pts, c, "s", "--", f"$\\xi = {xi:g}$")
    if has_upper_limits(by_d):
        add_upper_limit_legend_entry(ax1)

    ax1.set_yscale("log")
    ax1.set_xlabel("Correlation length, $\\xi$")
    ax1.set_ylabel("Logical error rate (per round)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, ncol=2)

    xi_ticks = [1, 5, 10, 15, 20, 28]
    top = ax1.twiny()
    top.set_xlim(ax1.get_xlim())
    top.set_xticks(xi_ticks)
    top.set_xticklabels([f"{1 - np.exp(-1 / x):.2f}" for x in xi_ticks])  # exact gap, eq. 79
    top.set_xlabel("Spectral gap, $\\Delta$")

    ax2.set_yscale("log")
    ax2.set_xlabel("Code distance, $d$")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, ncol=2)
    add_qubit_axis(ax2, sorted(by_d.keys()))

    label_panels(ax1, ax2)
    fig.tight_layout()
    save(fig, "storm_memory.png")


def plot_automaton(data):
    '''QCA bath: logical error rate vs rotation angle (a) and vs distance (b).'''
    theta_of = lambda e: round(e["theta"] / np.pi, 4)
    by_d = group(data, lambda e: e["distance"], theta_of)
    by_theta = group(data, theta_of, lambda e: e["distance"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for (d, pts), c in zip(sorted(by_d.items()), colors(len(by_d))):
        draw(ax1, pts, c, "o", "-", f"$d = {d}$")
    for (th, pts), c in zip(sorted(by_theta.items()), colors(len(by_theta))):
        draw(ax2, pts, c, "s", "--", f"$\\theta/\\pi = {th:g}$")
    if has_upper_limits(by_d):
        add_upper_limit_legend_entry(ax1)

    ax1.axvline(THETA_PSEUDO_CRITICAL, color="gray", ls=":", lw=1)
    ax1.set_yscale("log")
    ax1.set_xlabel("Controlled-rotation angle, $\\theta/\\pi$")
    ax1.set_ylabel("Logical error rate (per round)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, ncol=2)

    ax2.set_yscale("log")
    ax2.set_xlabel("Code distance, $d$")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(loc="lower left", frameon=True, ncol=2)
    add_qubit_axis(ax2, sorted(by_d.keys()))

    label_panels(ax1, ax2)
    fig.tight_layout()
    save(fig, "automaton_memory.png")


def main():
    storm = load(STORM_JSON)
    automaton = load(AUTOMATON_JSON)
    if storm is None and automaton is None:
        raise FileNotFoundError(f"No result files found in {RESULTS_DIR}")
    if storm:
        plot_storm(storm)
    if automaton:
        plot_automaton(automaton)
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()