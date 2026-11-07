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

def plot_sensitivity(df_beta, df_cap, base_beta, base_f, path):
    """Value of the contract (top) and congestion removed (bottom) against the discount (left) and the cap (right).

    df_beta, df_cap : outputs of sensitivity.run_sweep for the discount ("beta") and the cap factor ("f")
    base_beta, base_f : base-case values, marked with a vertical line
    path : file name without extension; saves path.png and path.pdf
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex="col", sharey="row",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    labels = {"I": r"$\mathcal{M}^I$ (individual)", "C": r"$\mathcal{M}^C$ (community)"}
    panels = [(df_beta, "beta", base_beta, r"Tariff discount $\beta$ [-]"),
              (df_cap, "f", base_f, r"Cap energy factor $f$ [-]  (cap $= f\,E^R/24$)")]

    for col, (df, x, base, xlabel) in enumerate(panels):
        ax_v, ax_d = axes[0, col], axes[1, col]
        for m in ("I", "C"):
            d = df[df["model"] == m].sort_values(x)
            ax_v.plot(d[x], d["contract_value"], color=COLORS[m], ls=STYLES[m], lw=2, marker="o",
                      ms=4, label=labels[m])
            ax_d.plot(d[x], 100 * d["congestion_removed"], color=COLORS[m], ls=STYLES[m], lw=2,
                      marker="o", ms=4, label=labels[m])
        ax_v.axhline(0, color=INK, lw=1)
        for ax in (ax_v, ax_d):
            ax.axvline(base, color=INK_2, lw=1, ls=":")
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_2)
            ax.tick_params(colors=INK_2, labelsize=9)
        ax_v.text(base, ax_v.get_ylim()[1], " base case", color=INK_2, fontsize=8, va="top", ha="left")
        ax_d.set_xlabel(xlabel, color=INK_2)

    axes[0, 0].set_ylabel("Value of the contract\n" + r"$C^R - C^m$ [DKK]", color=INK_2)
    axes[1, 0].set_ylabel("Congestion removed\n" + r"$1 - X^m / X^R$ [%]", color=INK_2)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=9)
    axes[0, 0].text(0.0, 1.0, "  signing pays off above the black line", transform=axes[0, 0].transAxes,
                    color=INK_2, fontsize=8, va="bottom")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_summary(df_grid, df_beta, base_beta, base_f, cap_base_kW, path):
    """Sensitivity to the contract terms in two panels.

    (a) Value of the community contract over discount x cap, with the break-even lines of C and I.
    (b) Along the discount (base cap): what the community gains and what the DSO gives up in tariffs.

    df_grid : output of sensitivity.run_grid;  df_beta : output of sensitivity.run_sweep for "beta"
    base_beta, base_f : base case;  cap_base_kW : cap of the base case [kW] (cap = cap_base_kW * f / base_f)
    path : file name without extension; saves path.png and path.pdf
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.32})

    # --- (a) Map: value of the community contract over discount x cap
    grid = {m: df_grid[df_grid["model"] == m].pivot(index="f", columns="beta", values="contract_value")
            for m in ("I", "C")}
    x = 100 * grid["C"].columns.to_numpy()                          # discount [%]
    y = cap_base_kW * grid["C"].index.to_numpy() / base_f            # cap [kW]
    zC, zI = grid["C"].to_numpy(), grid["I"].to_numpy()
    cmap = LinearSegmentedColormap.from_list("value", ["#9a3a12", "#eb6834", "#f0efec", "#1baf7a", "#0b6b47"])
    lim = np.abs(zC).max()
    dx, dy = x[1] - x[0], y[1] - y[0]
    mesh = ax1.pcolormesh(np.append(x - dx / 2, x[-1] + dx / 2), np.append(y - dy / 2, y[-1] + dy / 2), zC,
                          cmap=cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    cbar = fig.colorbar(mesh, ax=ax1, pad=0.02)
    cbar.set_label("Value of the community contract [DKK]", color=INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_2, labelsize=8)
    ax1.contour(x, y, zC, levels=[0], colors=INK, linewidths=2)
    if zI.min() < 0 < zI.max():
        ax1.contour(x, y, zI, levels=[0], colors=INK, linewidths=1.5, linestyles="dashed")
    ax1.plot(100 * base_beta, cap_base_kW, marker="o", ms=7, color=INK, ls="none")
    ax1.text(100 * base_beta + 2.5, cap_base_kW, "base case", fontsize=8.5, va="center", color=INK,
             bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
    box = dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85)
    ax1.text(12, 7.9, "not worth\nsigning", fontsize=8.5, ha="center", va="center", color=INK, bbox=box)
    ax1.text(52, 12.2, "worth signing together", fontsize=8.5, ha="center", va="center", color=INK, bbox=box)
    ax1.text(90, 14.1, "alone too", fontsize=8, ha="center", va="center", color=INK, bbox=box)
    ax1.set_xlabel("Discount on the grid tariffs [%]", color=INK_2)
    ax1.set_ylabel("Cap [kW]", color=INK_2)
    ax1.set_title("(a) The tighter the cap, the larger the discount needed", fontsize=10, loc="left", color=INK)

    # --- (b) Along the discount, at the base cap: community gain against DSO cost (both in DKK)
    d = {m: df_beta[df_beta["model"] == m].sort_values("beta") for m in ("I", "C")}
    ax2.axhline(0, color=INK, lw=1)
    ax2.plot(100 * d["C"]["beta"], d["C"]["contract_value"], color=COLORS["C"], lw=2,
             label=r"community gain, $\mathcal{M}^C$")
    ax2.plot(100 * d["I"]["beta"], d["I"]["contract_value"], color=COLORS["I"], lw=2, ls="--",
             label=r"community gain, $\mathcal{M}^I$")
    ax2.plot(100 * d["C"]["beta"], d["C"]["tariff_revenue_lost"], color=INK_2, lw=2, ls=":",
             label=r"DSO tariff revenue lost, $\mathcal{M}^C$")
    ax2.axvline(100 * base_beta, color=INK_2, lw=1, ls=(0, (1, 3)))
    ax2.text(100 * base_beta - 1, 0.04, "base case", transform=ax2.get_xaxis_transform(), color=INK_2, fontsize=8, ha="right")
    ax2.set_xlabel("Discount on the grid tariffs [%]", color=INK_2)
    ax2.set_ylabel("DKK per day", color=INK_2)
    ax2.set_title(f"(b) At {cap_base_kW:.1f} kW, the discount moves money, not load",
                  fontsize=10, loc="left", color=INK)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=9)
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_settlement(df, path):
    """Gain of every member under the community contract, settled at the internal price.

    Bars: gain against no contract (R); dots: gain against its own contract (I). Members grouped by
    disutility slope, from the most flexible to the most rigid.
    df   : output of settlement.settle
    path : file name without extension; saves path.png and path.pdf
    """
    order = ["none", "PV", "battery", "PV + battery"]
    d = df.assign(_a=df["assets"].map({a: k for k, a in enumerate(order)})).sort_values(["slope", "_a"])
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    colors = np.where(d["gain_vs_R"] >= 0, COLORS["C"], COLORS["I"])
    ax.barh(y, d["gain_vs_R"], color=colors, height=0.7, label="gain against no contract ($\\mathcal{M}^R$)")
    ax.plot(d["gain_vs_I"], y, "o", ms=6, color=INK, label="gain against its own contract ($\\mathcal{M}^I$)")
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m}  ({a})" for m, a in zip(d["member"], d["assets"])], fontsize=8.5)
    for k in range(1, 4):                                # separate the four slope classes
        ax.axhline(y[4 * k] + 0.5, color=GRID, lw=1)
    for k, s in enumerate(sorted(d["slope"].unique())):
        ax.text(1.0, y[4 * k] - 1.5, f"  slope {s:g} DKK/kWh", transform=ax.get_yaxis_transform(),
                color=INK_2, fontsize=8, va="center")
    ax.set_xlabel("Gain under the community contract [DKK per day]  (> 0: better off)", color=INK_2)
    n_lose = int((d["gain_vs_R"] < -1e-6).sum())
    ax.set_title(f"Settled at the internal price, {n_lose} of {len(d)} members lose against no contract",
                 fontsize=10, loc="left", color=INK)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncol=2, frameon=False, fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.8)
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
