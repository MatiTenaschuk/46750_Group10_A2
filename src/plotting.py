"""Figures of Question 1 (Assignment 2)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Colours: one fixed colour per model (validated categorical palette), neutral ink for references
COLORS = {"R": "#2a78d6", "I": "#eb6834", "C": "#1baf7a"}
STYLES = {"R": "-", "I": "--", "C": "-"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_import_vs_cap(results, com, path):
    """Aggregated import of R, I and C against the cap (top) and internal price of C (bottom).

    results : {"R": res_R, "I": res_I, "C": res_C}, outputs of CommunityModel.solve()
    com     : CommunityData (prices, tariffs, cap, discount)
    path    : file name without extension; saves path.png and path.pdf
    """
    hours = np.arange(len(com.energy_price))
    edges = np.arange(len(hours) + 1) - 0.5        # hour t is drawn over [t - 0.5, t + 0.5]
    cap = com.contract.capacity_limit_kW
    beta = com.contract.tariff_discount_ratio
    p = com.energy_price

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.0), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})

    # Shade the hours where the import without contract exceeds the cap (the congestion to remove)
    congested = results["R"]["P_imp"] > cap + 1e-6
    for ax in (ax1, ax2):
        for t in hours[congested]:
            ax.axvspan(t - 0.5, t + 0.5, color="#f1efe9", zorder=0, lw=0)

    # --- Top: aggregated import at the connection point against the cap
    ax1.stairs(cap, edges, baseline=None, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax1.text(23.6, cap[-1], " cap", color=INK, va="center", ha="left", fontsize=9)
    labels = {"R": r"$\mathcal{M}^R$ (no contract)", "I": r"$\mathcal{M}^I$ (individual)",
              "C": r"$\mathcal{M}^C$ (community)"}
    for name in ("R", "I", "C"):
        ax1.stairs(results[name]["P_imp"], edges, baseline=None, color=COLORS[name], lw=2,
                   ls=STYLES[name], label=labels[name], zorder=3)
    ax1.set_ylabel("Import at the connection\npoint [kW]", color=INK_2)
    ax1.set_ylim(0, None)
    handles, _ = ax1.get_legend_handles_labels()
    handles.append(Patch(facecolor="#f1efe9", label=r"hours where $\mathcal{M}^R$ exceeds the cap"))
    ax1.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax1.set_title("The community contract keeps the import at the cap; without contract it peaks at "
                  f"{results['R']['P_imp'].max():.1f} kW", fontsize=10, loc="left", color=INK)

    # --- Bottom: internal price of C against the day-ahead price and the discounted import price
    p_imp = p + (1 - beta) * com.import_tariff
    lam = results["C"]["price"]
    ax2.stairs(p, edges, baseline=None, color=INK_2, lw=1.2, label=r"day-ahead price $p_t$")
    ax2.stairs(p_imp, edges, baseline=None, color=INK_2, lw=1.2, ls=(0, (4, 3)),
               label=r"discounted import price $p_t + (1-\beta)\tau^{imp}$")
    ax2.stairs(lam, edges, baseline=None, color=COLORS["C"], lw=2,
               label=r"internal price $\lambda_t$ of $\mathcal{M}^C$")
    # Congestion rent: internal price above the discounted import price
    for t in hours[lam > p_imp + 1e-6]:
        ax2.fill_between([t - 0.5, t + 0.5], p_imp[t], lam[t], color=COLORS["C"], alpha=0.25, lw=0)
    ax2.fill_between([], [], color=COLORS["C"], alpha=0.25, lw=0, label="congestion rent")
    ax2.set_ylabel("Price [DKK/kWh]", color=INK_2)
    ax2.set_xlabel("Hour of the day", color=INK_2)
    ax2.set_ylim(0, None)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)