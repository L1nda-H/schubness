"""Figures for the gamma next-steps analysis.

python make_gamma_figures.py results/gamma_next_steps
reads   <root>/experiment, <root>/real, <root>/simulations, <root>/umap
writes  <root>/figures/*.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated reference palette (light mode), fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 8,
    }
)

ZHENG_POPS = ["b.cells", "cd14.monocytes", "naive.cytotoxic", "regulatory.t"]
POP_NAMES = {
    "b.cells": "B cells",
    "cd14.monocytes": "CD14 monocytes",
    "naive.cytotoxic": "Naive cytotoxic T",
    "regulatory.t": "Regulatory T",
    "v6.5 mouse 2i+LIF": "2i+LIF",
    "v6.5 mouse serum+LIF": "serum+LIF",
    "Dgcr8 knockout mouse serum+LIF": "Dgcr8 KO serum+LIF",
}
POP_COLOR = {p: SERIES[i] for i, p in enumerate(ZHENG_POPS)}
KUMAR_POPS = ["v6.5 mouse 2i+LIF", "v6.5 mouse serum+LIF", "Dgcr8 knockout mouse serum+LIF"]
POP_COLOR.update({p: SERIES[4 + i] for i, p in enumerate(KUMAR_POPS)})
FOCUS = ["Zhengmix4eq", "Zhengmix4uneq", "Zhengmix8eq", "Kumar"]
FOCUS_COLOR = {d: SERIES[i] for i, d in enumerate(FOCUS)}


def _end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(
        text, (x, y), xytext=(4, dy), textcoords="offset points",
        va="center", fontsize=8, color=INK2,
    )
    ax.plot([x], [y], "o", ms=4, color=color, mec=SURFACE, mew=1)


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fig_purity_trajectories(real, out):
    traj = pd.read_csv(real / "all_purity_trajectory.csv")
    datasets = ["Zhengmix4eq", "Zhengmix4uneq", "Kumar"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, name in zip(axes, datasets):
        t = traj[traj["dataset"] == name]
        pops = [p for p in POP_COLOR if p in set(t["source_label"])]
        for pop in pops:
            g = t[t["source_label"] == pop]
            size = int(g["population_size"].iloc[0])
            ax.step(g["gamma"], g["delta_same_per_cell"], where="post", color=POP_COLOR[pop],
                    label="{} (n={})".format(POP_NAMES.get(pop, pop), size))
        ax.legend(loc="lower left", fontsize=7.5)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xlim(0, 1)
        ax.set_title(name)
        ax.set_xlabel("γ")
    axes[0].set_ylabel("Same-label out-edges gained per cell")
    fig.suptitle(
        "Correction moves same-label edges from dense to sparse populations",
        x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.04,
    )
    fig.subplots_adjust(wspace=0.12)
    _save(fig, out / "fig1_purity_trajectories.png")


def fig_fragility(real, out):
    frag = pd.read_csv(real / "all_fragility.csv")
    frag["mid"] = (frag["gamma_lo"].clip(lower=0) + frag["gamma_hi"]) / 2
    frag["net_per_1000"] = 1000 * frag["net_delta_same"] / frag["swaps"]
    panels = [
        ("median_log_distance_ratio", "Median log(d_il / d_ij)", "Early swaps are near-ties in distance"),
        ("median_log_sigma_ratio", "Median log(σ_l / σ_j)", "…not sharper contrasts in scale"),
        ("removed_hub_fraction", "Share of swaps dropping a hub", "Early swaps mostly cut hub edges"),
        ("net_per_1000", "Net same-label edges per 1,000 swaps", "…and carry most of the purity gain"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, (col, ylabel, title) in zip(axes, panels):
        for name in FOCUS:
            f = frag[frag["dataset"] == name]
            ax.plot(f["mid"], f[col], "-o", ms=4, color=FOCUS_COLOR[name], label=name, mec=SURFACE, mew=1)
        ax.set_xlabel("γ* bin centre")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1)
        if col in ("median_log_sigma_ratio", "removed_hub_fraction"):
            ax.set_ylim(bottom=0)
        if col == "net_per_1000":
            ax.axhline(0, color=AXIS, lw=1)
    axes[0].legend(loc="upper left")
    fig.subplots_adjust(wspace=0.38)
    _save(fig, out / "fig2_fragility_by_gamma_star.png")


def fig_swap_plane(exp_real, out, name="Zhengmix4uneq"):
    swaps = pd.read_csv(exp_real / name / "swaps.csv.gz")
    fig, ax = plt.subplots(figsize=(6, 4.4))
    neutral = swaps[swaps["transition"].isin(["same->same", "cross->cross"])]
    ax.scatter(neutral["log_sigma_ratio"], neutral["log_distance_ratio"], s=2, color=NEUTRAL, alpha=0.35, lw=0, label="label-neutral swap", rasterized=True)
    for t, color, text in (("cross->same", SERIES[0], "cross → same (purity gain)"), ("same->cross", SERIES[1], "same → cross (purity loss)")):
        g = swaps[swaps["transition"] == t]
        ax.scatter(g["log_sigma_ratio"], g["log_distance_ratio"], s=5, color=color, alpha=0.7, lw=0, label="{} · {}".format(text, len(g)), rasterized=True)
    xmax = np.quantile(swaps["log_sigma_ratio"], 0.995)
    xs = np.linspace(0, xmax, 50)
    for g in (0.25, 0.5, 1.0):
        ax.plot(xs, g * xs / 2, color=MUTED, lw=1, ls="--")
        ax.annotate("γ*={:g}".format(g), (xmax, g * xmax / 2), xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(0, 0.6 * xmax)
    ax.set_xlabel("log(σ_l / σ_j)  — how much sparser the incoming neighbour is")
    ax.set_ylabel("log(d_il / d_ij)  — distance penalty")
    ax.set_title("{}: every swap sits on the line γ* = 2·Δlog d / Δlog σ".format(name))
    ax.legend(loc="upper left", markerscale=3)
    _save(fig, out / "fig3_swap_plane_{}.png".format(name))


def fig_resolution_confound(exp, real, out):
    res1 = pd.read_csv(exp / "all_gamma_summaries.csv")
    fk = pd.read_csv(real / "all_fixed_k_scores.csv").groupby(["dataset", "gamma"])["ari"].mean().reset_index()
    datasets = ["Kumar", "KumarTCC", "Zhengmix4uneq", "Zhengmix4eq"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=False)
    for ax, name in zip(axes, datasets):
        r = res1[res1["dataset"] == name]
        f = fk[fk["dataset"] == name]
        ax.plot(r["gamma"], r["ari_mean"], "-o", color=SERIES[1], ms=4, mec=SURFACE, mew=1, label="Leiden, resolution 1")
        ax.plot(f["gamma"], f["ari"], "-o", color=SERIES[0], ms=4, mec=SURFACE, mew=1, label="Leiden, resolution tuned to true K")
        for _, row in r.iterrows():
            ax.annotate("{:.1f}".format(row["clusters_mean"]), (row["gamma"], row["ari_mean"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ax.set_title(name)
        ax.set_xlabel("γ")
        ax.set_xlim(-0.03, 1.03)
    axes[0].set_ylabel("ARI (mean of 5 seeds)")
    axes[0].legend(loc="lower left")
    fig.text(0.01, -0.06, "Numbers above the orange points: mean cluster count at resolution 1.", color=MUTED, fontsize=8)
    _save(fig, out / "fig4_resolution_confound.png")


def fig_worst_population(real, out):
    rec = pd.read_csv(real / "all_population_recovery.csv")
    summ = pd.read_csv(real / "dataset_summary.csv")
    rows = []
    for name, g in rec.groupby("dataset"):
        by = g.groupby(["gamma", "label"])["f1"].mean().unstack()
        s = summ[summ["dataset"] == name].iloc[0]
        rows.append(
            {
                "dataset": name,
                "d_global": s["global_purity_g1"] - s["global_purity_g0"],
                "d_worst": s["worst_population_purity_g1"] - s["worst_population_purity_g0"],
                "d_res1": s["ari_res1_g1"] - s["ari_res1_g0"],
                "d_min_f1": by.loc[1.0].min() - by.loc[0.0].min(),
            }
        )
    d = pd.DataFrame(rows)
    d = d[~d["dataset"].str.startswith("SimKumar")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, xlabel in (
        (axes[0], "d_global", "Δ global same-label fraction (γ 0 → 1)"),
        (axes[1], "d_worst", "Δ worst-population same-label fraction (γ 0 → 1)"),
    ):
        ax.axhline(0, color=AXIS, lw=1)
        ax.axvline(0, color=AXIS, lw=1)
        ax.scatter(d[col], d["d_min_f1"], s=40, color=SERIES[0], edgecolor=SURFACE, lw=1.5, zorder=3)
        nudge = {"Kumar": (-5, 8, "right"), "KohTCC": (5, 8, "left"), "KumarTCC": (5, -9, "left"), "TrapnellTCC": (5, 7, "left"), "Koh": (5, -9, "left")}
        for _, row in d.iterrows():
            dx, dy, ha = nudge.get(row["dataset"], (5, 3, "left"))
            ax.annotate(row["dataset"], (row[col], row["d_min_f1"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2, ha=ha)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("Δ worst-population F1 (fixed-K Leiden)")
    axes[0].set_title("Global purity rises almost everywhere…")
    axes[1].set_title("…the worst-off population predicts the damage")
    _save(fig, out / "fig5_worst_population.png")
    return d


def fig_population_mechanism(real, sims, out):
    prof = pd.read_csv(real / "all_population_profile.csv")
    prof = prof[~prof["dataset"].str.startswith("SimKumar")]
    sim_prof = pd.read_csv(sims / "sim_profile.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.subplots_adjust(wspace=0.3)
    ax = axes[0]
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(1, color=AXIS, lw=1)
    ax.scatter(prof["neighbour_sigma_ratio"], prof["delta_same_per_cell"], s=20 + 300 * prof["population_fraction"], color=SERIES[0], alpha=0.75, edgecolor=SURFACE, lw=1.2, zorder=3)
    callouts = {
        ("Zhengmix4uneq", "naive.cytotoxic"): (8, 0, "left"),
        ("Zhengmix8eq", "naive.cytotoxic"): (8, 6, "left"),
        ("Zhengmix4eq", "regulatory.t"): (8, 4, "left"),
        ("Koh", "H7_derived_Sclrtm"): (8, 0, "left"),
    }
    for (ds, lab), (dx, dy, ha) in callouts.items():
        row = prof[(prof["dataset"] == ds) & (prof["label"] == lab)]
        if len(row):
            row = row.iloc[0]
            ax.annotate("{} · {}".format(ds, POP_NAMES.get(lab, lab)), (row["neighbour_sigma_ratio"], row["delta_same_per_cell"]), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=INK2, va="center", ha=ha)
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 4]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Neighbouring cells' σ / population's own σ (log scale)")
    ax.set_ylabel("Same-label out-edges gained per cell (γ 0 → 1)")
    ax.set_title("Real data: all Duo populations\n(dot size ∝ share of cells)")

    ax = axes[1]
    sp = sim_prof[sim_prof["scenario"] == "pair"]
    ax.axhline(0, color=AXIS, lw=1)
    for i, (ratio, g) in enumerate(sp.groupby("size_ratio")):
        for pop, marker in (("A", "o"), ("B", "s")):
            h = g[g["label"] == pop].sort_values("spread_ratio")
            ax.plot(h["spread_ratio"], h["delta_same_per_cell"], marker=marker, ms=6, color=SERIES[i], mec=SURFACE, mew=1, ls="-" if pop == "A" else ":", label="pop {} · A is {:g}× the size of B".format(pop, ratio))
    ax.set_xlabel("Spread of A relative to B (>1: A is sparser)")
    ax.set_ylabel("Same-label out-edges gained per cell (γ 0 → 1)")
    ax.set_title("Simulation: the denser\npopulation always loses")
    ax.legend(loc="best")
    _save(fig, out / "fig6_density_mechanism.png")


def fig_simulations(sims, out):
    grid = pd.read_csv(sims / "sim_grid.csv")
    pair = grid[grid["scenario"] == "pair"]
    spreads = sorted(pair["spread_ratio"].unique())
    fig, axes = plt.subplots(2, len(spreads), figsize=(11, 5.6), sharex=True)
    for j, spread in enumerate(spreads):
        for i, (ratio, g) in enumerate(pair[pair["spread_ratio"] == spread].groupby("size_ratio")):
            m = g.groupby("gamma").mean(numeric_only=True)
            label = "A {:g}× size of B".format(ratio)
            axes[0, j].plot(m.index, m["purity_A"], color=SERIES[i], label=label + " · pop A")
            axes[0, j].plot(m.index, m["purity_B"], color=SERIES[i], ls=":", label=label + " · pop B")
            axes[1, j].plot(m.index, m["ari_fixed_k"], color=SERIES[i], label=label)
        dens = "A denser" if spread < 1 else ("A sparser" if spread > 1 else "equal density")
        axes[0, j].set_title("A spread {:g}× B ({})".format(spread, dens))
        axes[1, j].set_xlabel("γ")
    axes[0, 0].set_ylabel("kNN same-label fraction")
    axes[1, 0].set_ylabel("ARI, fixed-K Leiden")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    axes[1, 0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Two-population simulations (50-D Gaussians, 3 replicates)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out / "fig7_simulations_pair.png")

    imb = grid[grid["scenario"] == "imbalance"]
    # Prefer the dedicated high-replicate imbalance run when it exists.
    imb_path = sims.parent / "simulations_imbalance" / "sim_grid.csv"
    if imb_path.exists():
        imb = pd.read_csv(imb_path)
    n_imb = imb["replicate"].nunique()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for i, (cfg, g) in enumerate(imb.groupby("config", sort=False)):
        m = g.groupby("gamma")[["purity_A", "f1_A", "ari_fixed_k"]].agg(["mean", "std"])
        label = "A:B = " + cfg.split("= ")[1]
        for ax, col in zip(axes, ("purity_A", "f1_A", "ari_fixed_k")):
            mean, sd = m[col]["mean"], m[col]["std"]
            ax.fill_between(mean.index, mean - sd, mean + sd, color=SERIES[i], alpha=0.15, lw=0)
            ax.plot(mean.index, mean, color=SERIES[i], label=label)
    axes[0].set_title("kNN purity of A (the denser side)")
    axes[1].set_title("Recovery of A (best-match F1, fixed K)")
    axes[2].set_title("ARI, fixed-K Leiden")
    for ax in axes:
        ax.set_xlabel("γ")
    axes[2].legend(loc="lower left")
    fig.suptitle("A slightly denser than B and overlapping it: balanced vs 1:10 (mean ± sd, {} replicates)".format(n_imb), x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.05)
    fig.tight_layout()
    _save(fig, out / "fig7b_simulations_imbalance.png")

    grad = grid[grid["scenario"] == "gradient"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, (halo, g) in enumerate(grad.groupby("halo_spread")):
        m = g.groupby("gamma").mean(numeric_only=True)
        lab = "no gradient (control)" if halo == 1 else "halo spread {:g}×".format(halo)
        axes[0].plot(m.index, m["ari_mean"], color=SERIES[i], label=lab)
        axes[0].plot(m.index, m["ari_fixed_k"], color=SERIES[i], ls=":")
        axes[1].plot(m.index, m["clusters_mean"], color=SERIES[i], label=lab)
        axes[2].plot(m.index, m["within_compartment_fraction_P"], color=SERIES[i], label=lab)
    axes[0].set_title("ARI: resolution 1 (solid) vs fixed K (dotted)")
    axes[1].set_title("Clusters at resolution 1 (true K = 2)")
    axes[2].set_title("P's edges staying in core or halo")
    for ax in axes:
        ax.set_xlabel("γ")
    axes[1].legend(loc="upper left")
    fig.suptitle("Density-gradient simulation: dense core + sparse halo in population P (halo 1× = no gradient)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.05)
    fig.tight_layout()
    _save(fig, out / "fig8_simulations_gradient.png")


def fig_umap(umap_dir, out):
    datasets = ["Zhengmix4eq", "Zhengmix4uneq", "Kumar"]
    gammas = [0.0, 0.5, 1.0]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, name in enumerate(datasets):
        for j, g in enumerate(gammas):
            ax = axes[i, j]
            data = np.load(umap_dir / "{}_gamma{:.2f}.npz".format(name, g), allow_pickle=True)
            emb, labels = data["embedding"], data["labels"].astype(str)
            for pop in [p for p in POP_COLOR if p in set(labels)]:
                m = labels == pop
                ax.scatter(emb[m, 0], emb[m, 1], s=2 if len(labels) > 1000 else 6, color=POP_COLOR[pop], lw=0, label=POP_NAMES.get(pop, pop), rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title("{} · γ = {:g}".format(name, g), fontsize=9)
        axes[i, 2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=4)
    fig.suptitle("UMAP of each D_γ kNN graph (same PCA, graph changes only)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, out / "fig9_umap_by_gamma.png")


def main(root):
    root = Path(root)
    exp, real, sims = root / "experiment", root / "real", root / "simulations"
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_purity_trajectories(real, out)
    fig_fragility(real, out)
    fig_swap_plane(real, out)
    fig_resolution_confound(exp, real, out)
    fig_worst_population(real, out)
    fig_population_mechanism(real, sims, out)
    fig_simulations(sims, out)
    fig_umap(root / "umap", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/gamma_next_steps")
