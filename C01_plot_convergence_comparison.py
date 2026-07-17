import argparse
import os
import numpy as np
import pandas as pd


MAX_ITER = 100


def collect_convergence(case_name, max_iter=MAX_ITER):
    """
    Collect best-so-far max left-behind time for a given case.
    """
    best_list = []
    current_best = np.inf

    for iter in range(max_iter):
        file_name = f'output/{case_name}/left_behind_log_iteration_{iter}.csv'

        if os.path.exists(file_name):
            lb_log = pd.read_csv(file_name, low_memory=False)
            max_lb = np.max(lb_log['left_behind_times'])
            if case_name == 'BYO' and iter == 0:
                lb_log = pd.read_csv(f'output/reference/left_behind_log_iteration_0.csv', low_memory=False)
                max_lb = np.max(lb_log['left_behind_times']) ### BYO miss first, use reference

            current_best = min(current_best, max_lb)
            best_list.append(current_best)
        else:
            break

    return np.array(best_list)


def collect_all_convergence(max_iter=MAX_ITER):
    methods = {
        "Proposed": "reference",
        "MLR": "equity_efficiency_MLR",
        "DE": "DE",
        "BO": "BYO",
    }
    results = {
        label: collect_convergence(folder, max_iter)
        for label, folder in methods.items()
    }
    result_length = max(len(values) for values in results.values())
    source_data = pd.DataFrame({"Iteration": np.arange(result_length)})
    for label, values in results.items():
        source_data[label] = pd.Series(values)
    return source_data


def plot_all_convergence(save_fig=True,
                         save_path="img/convergence_comparison.pdf",
                         source_data_path=None,
                         fontsize=16):

    import matplotlib.pyplot as plt

    # ----------------------------
    # Collect data
    # ----------------------------
    if source_data_path:
        source_data = pd.read_csv(source_data_path)
    else:
        source_data = collect_all_convergence(MAX_ITER)
        source_data_path = "output/convergence_comparison_source_data.csv"
        source_data.to_csv(source_data_path, index=False)

    # ----------------------------
    # Plot style configuration
    # ----------------------------
    plt.rcParams.update({
        "font.size": fontsize,
        "font.family": "serif",
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize * 0.85,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    line_styles = {
        "Proposed": dict(color="black", linewidth=2.8),
        "MLR": dict(color="#666666", linestyle="--", linewidth=2.2),
        "DE": dict(color="#2F6DB0", linestyle="-.", linewidth=2.2),
        "BO": dict(color="#8FAED0", linestyle=":", linewidth=2.6),
    }

    # ----------------------------
    # Plot curves
    # ----------------------------
    for label in ("Proposed", "MLR", "DE", "BO"):
        values = source_data[label].dropna().to_numpy()
        iterations = source_data.loc[source_data[label].notna(), "Iteration"].to_numpy()

        ax.plot(
            iterations,
            values,
            label=label,
            **line_styles.get(label, {})
        )

    # ----------------------------
    # Labels
    # ----------------------------
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Best $W_{\mathrm{Max}}$")

    ax.legend(frameon=False, loc="upper right")
    ax.grid(False)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--source-data")
    args = parser.parse_args()

    if args.collect_only:
        output_path = args.source_data or "output/convergence_comparison_source_data.csv"
        collect_all_convergence(MAX_ITER).to_csv(output_path, index=False)
        print(f"Source data saved to {output_path}")
    else:
        source_data_path = args.source_data
        default_source = "output/convergence_comparison_source_data.csv"
        if source_data_path is None and os.path.exists(default_source):
            source_data_path = default_source
        plot_all_convergence(save_fig=True, source_data_path=source_data_path)
