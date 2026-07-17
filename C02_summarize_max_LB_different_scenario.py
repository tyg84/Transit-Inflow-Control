import os
import numpy as np
import pandas as pd


MAX_ITER = 100


def get_initial_max_lb(scenario):
    """
    Get initial max left-behind time from reference model iteration 0.
    """
    if scenario == "100_percent_demand":
        case_name = "reference"
    else:
        case_name = f"reference_{scenario}"

    file_name = f'output/{case_name}/left_behind_log_iteration_0.csv'

    if os.path.exists(file_name):
        lb_log = pd.read_csv(file_name, low_memory=False)
        return np.max(lb_log['left_behind_times'])
    else:
        return np.nan


def collect_convergence(case_name, max_iter=MAX_ITER):
    """
    Collect best-so-far max left-behind time.
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

    if len(best_list) == 0:
        return np.nan

    return best_list[-1]


def summarize_all_scenarios():

    scenarios = [
        "100_percent_demand",  # reference
        "110_percent_demand",
        "90_percent_demand",
    ]

    methods = {
        "Proposed": "reference",
        "MLR": "rule_based",
        "DE": "DE",
        "BYO": "BYO"
    }

    summary_rows = []

    for scenario in scenarios:

        if scenario == "":
            scenario_name = "reference"
        else:
            scenario_name = f"reference_{scenario}"

        row = {"Scenario": scenario_name}

        # -------------------------
        # Initial max LB
        # -------------------------
        row["Initial"] = get_initial_max_lb(scenario)

        # -------------------------
        # Final best for each method
        # -------------------------
        for method_name, method_folder in methods.items():

            if scenario == "100_percent_demand":
                case_name = f"{method_folder}"
            else:
                case_name = f"{method_folder}_{scenario}"

            final_best = collect_convergence(case_name)

            row[method_name] = final_best

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    save_path = "output/final_best_max_left_behind_summary.csv"
    summary_df.to_csv(save_path, index=False)

    print(f"Summary saved to {save_path}")
    print(summary_df)


if __name__ == "__main__":
    summarize_all_scenarios()