import os
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt




def plot_convergence(
    best_max_lb_list,
    actual_lb_list,
    save_fig = False,
    fontsize=14,
    best_color='black',
    actual_color='tab:blue',
    linewidth_best=2.5,
    linewidth_actual=1.8,
    alpha_actual=0.35,
    save_path="convergence_curve.pdf"
):
    """
    Plot convergence curve for optimization process.


    Parameters
    ----------
    best_max_lb_list : list
        Best lower bound values so far.
    actual_lb_list : list
        Current lower bound values at each iteration.
    fontsize : int
        Global font size.
    best_color : str
        Color of best curve.
    actual_color : str
        Color of actual curve.
    linewidth_best : float
        Line width of best curve.
    linewidth_actual : float
        Line width of actual curve.
    alpha_actual : float
        Transparency of actual curve.
    save_path : str
        Output pdf file name.
    """


    # Convert to numpy arrays
    best = np.array(best_max_lb_list)
    actual = np.array(actual_lb_list)


    iterations = np.arange(1, len(best) + 1)


    # Scientific style configuration
    plt.rcParams.update({
        "font.size": fontsize,
        "font.family": "serif",
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "legend.fontsize": fontsize * 0.9,
        "xtick.labelsize": fontsize * 0.9,
        "ytick.labelsize": fontsize * 0.9,
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,  # editable text in illustrator
    })


    fig, ax = plt.subplots(figsize=(7.5, 4.5))


    # Plot actual LB (background, transparent)
    ax.plot(
        iterations,
        actual,
        color=actual_color,
        linewidth=linewidth_actual,
        alpha=alpha_actual,
        label="Current max left behind"
    )


    # Plot best LB (solid)
    ax.plot(
        iterations,
        best,
        color=best_color,
        linewidth=linewidth_best,
        label="Best max left behind"
    )


    # Labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max left behind")


    # Remove top/right spines for cleaner look
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)


    ax.legend(frameon=False)
    ax.grid(False)


    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()








def plot_max_lb_convergence(case_name,save_fig):
    best_max_lb_list = []
    actual_lb_list = []
    current_max_lb = np.inf
    for iter in range(100):
        file_name = f'output/{case_name}/left_behind_log_iteration_{iter}.csv'
        if os.path.exists(file_name):
            lb_log = pd.read_csv(file_name, low_memory=False)
            max_lb = np.max(lb_log['left_behind_times'])
            if max_lb <= current_max_lb:
                best_max_lb_list.append(max_lb)
                current_max_lb = max_lb
            else:
                best_max_lb_list.append(current_max_lb)
            actual_lb_list.append(max_lb)
        else:
            break


    plot_convergence(
        best_max_lb_list,
        actual_lb_list,
        save_fig=save_fig,
        fontsize=16,
        best_color="black",
        actual_color="tab:orange",
        save_path="img/left_behind_convergence_reference.pdf",
    )






if __name__ == '__main__':
    case_name = 'reference'
    plot_max_lb_convergence(case_name, save_fig=True)
