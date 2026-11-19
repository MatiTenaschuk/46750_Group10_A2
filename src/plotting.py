"""Figures of Question 1 (Assignment 2)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write files, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter, MaxNLocator
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
    
def plot_Q3_part1_results(profile_results, slope_results, nominal_report, folder="results/q3_step1/figures", show=False, note=None, font_size=11, figure_width=7.2, paper=True):
    """Visualize costs, constraint exposure, draws and member outcomes.

    The defaults target a full-width LaTeX figure (~7.2 inches). Use PDF at
    width=\\textwidth. font_size and figure_width control the physical typography
    and canvas width. paper=False includes explanatory headers and footnotes;
    paper=True leaves that context to the LaTeX caption. Scenario numbers start
    at zero, matching the source reports. Both experiments use separate draws.

    Colors identify separate experiments, not paired scenarios. All draws are
    shown; means/frequencies use scenario probabilities. IR means private cost
    above the same-realization individual-contract benchmark. Cap exceedance is
    permitted by the soft contract and charged a penalty. With five scenarios,
    empirical frequencies have 20-percentage-point resolution. This function
    compares the specified samples/bounds; it does not rank equally sized risks.
    Asset markers use the current member-ID suffixes _PV, _B and _PVB.
    """
    reports = [profile_results, slope_results]
    names = ["Profile", "Slope"]
    colors = ["#267A9B", "#C06434"]
    nominal_cost = float(nominal_report["summary"]["mean_cost_DKK"])
    ids = list(profile_results["members"])
    if set(ids) != set(slope_results["members"]):
        raise ValueError("Both reports must contain the same members.")
    for report in reports:
        rows = report["scenarios"]
        weights = np.array([r["probability"] for r in rows], dtype=float)
        if not rows or np.any(weights < 0) or not np.isfinite(weights).all() or not np.isclose(weights.sum(), 1.):
            raise ValueError("Nonempty reports with valid scenario probabilities required.")
        if any(set(r["members"]) != set(ids) for r in rows):
            raise ValueError("Each draw must contain every member.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    style = {"font.family": "DejaVu Serif", "font.size": font_size,
             "axes.titlesize": font_size, "axes.labelsize": font_size,
             "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
             "legend.fontsize": font_size - 2, "pdf.fonttype": 42,
             "ps.fonttype": 42, "axes.spines.top": False,
             "axes.spines.right": False, "axes.edgecolor": "#A6ADB4",
             "text.color": "#25313C", "axes.labelcolor": "#25313C",
             "xtick.color": "#46515C", "ytick.color": "#46515C",
             "axes.labelpad": 3, "savefig.facecolor": "white"}


    def tidy(ax, title, ylabel=None):
        ax.set_title(title, loc="left", fontweight="bold", pad=6)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)

    def save(fig, name):
        fig.canvas.draw()
        for ext in ("png", "pdf"):
            path = folder / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.035)
            paths[f"{name}_{ext}"] = path

    footer = "Separate joint draws; prices and optimistic response rule held fixed. Frequencies describe these samples only."
    if note:
        footer += "\n" + note
    with plt.rc_context(style):
        fig, grid = plt.subplots(3, 2, figsize=(figure_width, 8.3), layout="constrained")
        fig.set_constrained_layout_pads(w_pad=.03, h_pad=.035, wspace=.04, hspace=.055)
        # Keep A-F in reading order while retaining the original panel code.
        axes = np.asarray(grid).reshape(2, 3)
        if not paper:
            fig.suptitle(f"Fixed prices under preference uncertainty\nNominal: {nominal_cost:.2f} DKK", fontsize=font_size + 1)
            fig.supxlabel(footer, fontsize=font_size - 2)

        # A: every scenario, probability-weighted means, and nominal reference.
        ax = axes[0, 0]
        tidy(ax, "A  Community cost", "Cost (DKK/day)")
        for j, report in enumerate(reports):
            vals = [r["cost_DKK"] for r in report["scenarios"]]
            jitter = np.linspace(-.12, .12, len(vals))
            ax.scatter(j + jitter, vals, s=44, color=colors[j], alpha=.85, zorder=3)
            mean = report["summary"]["mean_cost_DKK"]
            ax.scatter(j + .24, mean, marker="D", s=60, color=colors[j], edgecolor="white", zorder=4)

        ax.axhline(nominal_cost, color="#53616E", linestyle="--", linewidth=1.1, label="Nominal")
        ax.set_xticks([0, 1], names)
        ax.set_xlim(-.45, 1.65)
        ax.scatter([], [], marker="D", color="#53616E", label="Mean")
        ax.legend(frameon=False, fontsize=font_size - 2, loc="upper left")

        # B: average economic components; payments are internal transfers.
        ax = axes[0, 1]
        tidy(ax, "B  Mean cost components", "Cost (DKK/day)")
        bottom = np.zeros(2)
        components = [("mean_energy_DKK", "Energy", "#8BB7C4"), ("mean_tariffs_DKK", "Tariffs", "#D6E0E4"), ("mean_own_cost_DKK", "PV + disutility", "#626F86"), ("mean_penalty_DKK", "Cap penalty", "#D98A5B")]
        for key, label, color in components:
            vals = np.array([r["summary"][key] for r in reports])
            ax.bar([0, 1], vals, bottom=bottom, width=.58, label=label, color=color)
            bottom += vals
        for j, value in enumerate(bottom):
            ax.text(j, value + max(bottom) * .025, f"{value:.2f}", ha="center", fontsize=font_size - 1)
        ax.set_ylim(0, max(bottom) * 1.48)
        ax.set_xticks([0, 1], names)
        ax.legend(frameon=False, fontsize=font_size - 2, ncol=2, loc="upper left")

        # C: preserve both small profile imbalances and very large slope deficits.
        ax = axes[0, 2]
        tidy(ax, "C  Manager balance", "Balance (DKK, symlog)")
        ax.set_yscale("symlog", linthresh=5)
        ax.set_yticks([-500, -100, -10, 0, 10], ["-500", "-100", "-10", "0", "10"])
        ax.axhline(0, color="#53616E", linewidth=1)
        for j, report in enumerate(reports):
            x = np.arange(len(report["scenarios"])) + (j - .5) * .18
            y = [r["balance_DKK"] for r in report["scenarios"]]
            ax.scatter(x, y, s=44, color=colors[j], label=names[j], zorder=3)
            for xx, yy in (zip(x, y) if len(y) <= 6 else []):
                ax.annotate(f"{yy:+.1f}", (xx, yy), xytext=(0, 7 if yy >= 0 else -12), textcoords="offset points", ha="center", fontsize=font_size - 2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.set_xlabel("Scenario")
        ax.margins(y=.22)

        # D: separate equality failures from manager deficits and member losses.
        ax = axes[1, 0]
        tidy(ax, "D  Violation frequencies", "Scenario probability")
        keys = ["BB_frequency", "deficit_frequency", "IR_any_frequency", "cap_frequency"]
        for j, report in enumerate(reports):
            vals = [report["summary"][k] for k in keys]
            bars = ax.bar(np.arange(4) + (j - .5) * .36, vals, width=.34, color=colors[j], label=names[j])
            for bar, value in zip(bars, vals):
                ax.annotate(f"{value:.0%}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 4 + 11 * j), textcoords="offset points", ha="center", fontsize=font_size - 2)
        ax.set_xticks(range(4), ["BB", "Deficit", "Any IR", "Cap"])
        ax.set_ylim(0, 1.30)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1))


        # E: zero cap violations versus large contractual exposure under slope uncertainty.
        ax = axes[1, 1]
        tidy(ax, "E  Cap exceedance", "Exceedance (kWh/day)")
        for j, report in enumerate(reports):
            vals = [r["exceedance_kWh"] for r in report["scenarios"]]
            bars = ax.bar(np.arange(len(vals)) + (j - .5) * .36, vals, width=.34, color=colors[j])
            if len(vals) <= 6:
                ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=3, fontsize=font_size - 2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        ax.set_xlabel("Scenario")
        ax.margins(y=.22)

        # F: magnitudes matter as well as frequencies. No averaging away member losses.
        ax = axes[1, 2]
        tidy(ax, "F  Mean positive violations", "DKK per day")
        for j, report in enumerate(reports):
            vals = [report["summary"]["mean_deficit_DKK"], report["summary"]["mean_total_IR_violation_DKK"]]
            bars = ax.bar(np.arange(2) + (j - .5) * .36, vals, width=.34, color=colors[j])
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in vals], padding=4, fontsize=font_size - 2)
        ax.set_xticks([0, 1], ["Manager deficit", "Total IR loss"])
        ax.margins(y=.25)
        ax.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, names)],
                  frameon=False, ncol=2, loc="upper right", fontsize=font_size - 2)
        save(fig, "part1_overview")

        # Detailed figure: draw-response association and per-member IR exposure.
        fig2, axes2 = plt.subplots(2, 2, figsize=(figure_width, 7.4),
                                   gridspec_kw={"height_ratios": [1, 1.8]}, layout="constrained")
        fig2.set_constrained_layout_pads(w_pad=.03, h_pad=.035, wspace=.035, hspace=.045)
        if not paper:
            fig2.suptitle("Member responses and participation", fontsize=font_size + 1)
        markers = {"Load only": "o", "PV": "s", "Battery": "^", "PV + battery": "D"}
        def asset(i):
            return "PV + battery" if i.endswith("_PVB") else "PV" if i.endswith("_PV") else "Battery" if i.endswith("_B") else "Load only"
        for j, report in enumerate(reports):
            ax = axes2[0, j]
            key = "theta_profile" if j == 0 else "theta_cL"
            tidy(ax, f"{'A' if j == 0 else 'B'}  {names[j]} response", "Load change (kWh/day)")
            for group, marker in markers.items():
                points = [d for r in report["scenarios"] for i, d in r["members"].items() if asset(i) == group]
                if points:
                    ax.scatter([d[key] for d in points], [d["delta_load_kWh"] for d in points], marker=marker, color=colors[j], s=30, alpha=.6, linewidths=.4, edgecolors="white", label=group)
            ax.axhline(0, color="#53616E", linewidth=.8)
            ax.axvline(1, color="#53616E", linestyle="--", linewidth=.8)
            ax.set_xlabel(r"Profile exponent $\theta$" if j == 0 else r"Disutility multiplier $\theta$")
            if j == 0:
                ax.legend(frameon=False, fontsize=font_size - 2, ncol=2,
                          handletextpad=.3, columnspacing=.6, loc="upper right")
        for j, (key, title) in enumerate([("IR_frequency", "C  Member IR frequency"), ("mean_IR_violation_DKK", "D  Mean positive IR gap")]):
            ax = axes2[1, j]
            data = np.array([[r["members"][i][key] for r in reports] for i in ids])
            scale = 1.0 if j == 0 else max(float(data.max()), 1e-12)
            im = ax.imshow(data, cmap="Blues" if j == 0 else "YlOrRd", norm=Normalize(0, scale), aspect="auto")
            ax.set_title(title, loc="left", fontweight="bold", pad=6)
            ax.set_xticks([0, 1], names)
            ax.set_yticks(range(len(ids)), ids)
            ax.tick_params(length=0)
            for a in range(len(ids)):
                for b in range(2):
                    value = data[a, b]
                    text = f"{value:.0%}" if j == 0 else f"{value:.3f}"
                    ax.text(b, a, text, ha="center", va="center", fontsize=font_size - 2, color="white" if value > .6 * scale else "#25313C")
            colorbar = fig2.colorbar(im, ax=ax, fraction=.045, pad=.04)
            if j == 0:
                colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                colorbar.set_label("Mean positive gap (DKK/day)", fontsize=font_size - 2)
        if not paper:
            fig2.supxlabel("IR benchmark: each draw's individual contract.\n" + (note or "Separate draws; prices and response rule fixed."), fontsize=font_size - 2)
        save(fig2, "part1_members")
        if show:
            plt.show()
        else:
            plt.close(fig)
            plt.close(fig2)
    return paths
