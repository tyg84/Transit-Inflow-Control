import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MaxNLocator



def lighten_color(color, amount=0.5):
    """
    Lighten a matplotlib color by mixing it with white.

    amount=0 -> original color
    amount=1 -> white
    """
    c = np.array(to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * c + amount * white)


def build_stage_pax_history(case_name, max_iter=100):
    """
    For each iteration:
    - compute current max LB
    - compute number of passengers with current max LB
    - maintain:
        best_max_lb_so_far
        best_pax_count_with_that_best_lb_so_far

    Returns
    -------
    df : pd.DataFrame
        columns:
            iteration
            actual_max_lb
            actual_pax_at_max_lb
            best_max_lb
            best_pax_at_best_lb
    """
    rows = []

    current_best_lb = np.inf
    current_best_pax = np.inf

    for iter_idx in range(max_iter):
        file_name = f'output/{case_name}/left_behind_log_iteration_{iter_idx}.csv'
        if not os.path.exists(file_name):
            break

        lb_log = pd.read_csv(file_name, low_memory=False)
        lb_values = lb_log['left_behind_times'].to_numpy()

        actual_max_lb = int(np.max(lb_values))
        actual_pax_at_max_lb = int(np.sum(lb_values == actual_max_lb))

        if actual_max_lb < current_best_lb:
            current_best_lb = actual_max_lb
            current_best_pax = actual_pax_at_max_lb
        elif actual_max_lb == current_best_lb:
            current_best_pax = min(current_best_pax, actual_pax_at_max_lb)

        rows.append({
            "iteration": iter_idx,
            "actual_max_lb": actual_max_lb,
            "actual_pax_at_max_lb": actual_pax_at_max_lb,
            "best_max_lb": int(current_best_lb),
            "best_pax_at_best_lb": int(current_best_pax),
        })

    return pd.DataFrame(rows)


def plot_stagewise_best_pax_convergence(
    case_name,
    save_fig=False,
    save_path="img/stagewise_best_pax_convergence.pdf",
    fontsize=15,
    base_color="tab:red",
    background_base="tab:blue",
    linewidth=2.2,
    marker_size=18,
):
    """
    Plot one vertically stacked subplot for each best-max-LB stage.

    x-axis: iteration
    y-axis in each subplot: best number of passengers with that LB stage so far

    Each stage gets a separate background shade.
    """
    df = build_stage_pax_history(case_name)
    if df.empty:
        print(f"No iteration files found for case: {case_name}")
        return

    plt.rcParams.update({
        "font.size": fontsize,
        "font.family": "serif",
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
    })

    stages = sorted(df["best_max_lb"].unique(), reverse=True)
    n_stage = len(stages)

    fig, axes = plt.subplots(
        n_stage, 1,
        figsize=(10, 2.0 * n_stage + 0.5),
        sharex=True,
        constrained_layout=True
    )
    fig.supylabel(
        "Best record count at the maximum in each $W_{\\text{Max}}$ phase",
        fontsize=fontsize
    )

    if n_stage == 1:
        axes = [axes]

    # create a gradient of background colors
    # top panel darker, lower panels lighter
    bg_amounts = np.linspace(0.88, 0.97, n_stage)

    for i, stage in enumerate(stages):
        ax = axes[i]
        stage_df = df[df["best_max_lb"] == stage].copy()

        x = stage_df["iteration"].to_numpy()
        y = stage_df["best_pax_at_best_lb"].to_numpy()

        # background color
        bg_color = lighten_color(background_base, amount=bg_amounts[i])
        ax.set_facecolor(bg_color)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # curve
        ax.plot(
            x, y,
            color=base_color,
            linewidth=linewidth,
            zorder=3
        )
        ax.scatter(
            x, y,
            color=base_color,
            s=marker_size,
            zorder=4
        )

        # optional: fill under curve
        ax.fill_between(
            x, y, np.min(y) if len(y) > 0 else 0,
            color=base_color,
            alpha=0.10,
            zorder=2
        )

        # y label as stage name
        # ax.set_ylabel(r"$W_{\text{Max}}$"+f"={stage}")

        # clean style
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # make y range tight but readable
        y_min = max(0, y.min() - max(1, int(0.05 * max(y.max(), 1))))
        y_max = y.max() + max(1, int(0.08 * max(y.max(), 1)))
        if y_min == y_max:
            y_max = y_min + 1
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        # small title text inside each panel
        ax.text(
            0.98, 0.88,
            r"$W_{\text{Max}}$" + f" = {stage}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize * 0.9,
            fontweight="bold"
        )
    axes[-1].set_xlabel("Iteration")

    # overall title
    # fig.suptitle(
    #     "Stage-wise convergence of best passenger count",
    #     fontsize=fontsize + 1
    # )

    if save_fig:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    case_name = "reference"
    plot_stagewise_best_pax_convergence(
        case_name=case_name,
        save_fig=True,
        save_path="img/stagewise_best_pax_convergence_reference.pdf"
    )
