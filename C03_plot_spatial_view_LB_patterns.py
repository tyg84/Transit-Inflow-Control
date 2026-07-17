import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
X_STRETCH = 1.35

stations = [
    {"station_name": "Xinzhuang",         "station_id": 101, "x": 5,   "y": 115},
    {"station_name": "Waihuan Rd",        "station_id": 102, "x": 12,  "y": 115},
    {"station_name": "Lianhua Rd",        "station_id": 103, "x": 20,  "y": 110},
    {"station_name": "Jinjiang Park",     "station_id": 104, "x": 25,  "y": 102},
    {"station_name": "South Station",     "station_id": 105, "x": 30,  "y": 94},
    {"station_name": "Xujiahui",          "station_id": 108, "x": 40,  "y": 80},
    {"station_name": "Changshu Rd",       "station_id": 110, "x": 48,  "y": 68},
    {"station_name": "Shanxi S Rd",       "station_id": 111, "x": 64,  "y": 68},
    {"station_name": "People's Square",   "station_id": 113, "x": 72,  "y": 52},
    {"station_name": "Station",           "station_id": 116, "x": 68,  "y": 36},
    {"station_name": "Circus World",      "station_id": 119, "x": 60,  "y": 28},
    {"station_name": "Gongkang Rd",       "station_id": 122, "x": 60,  "y": 20},
    {"station_name": "Gongfu Xincun",     "station_id": 125, "x": 60,  "y": 12},
    {"station_name": "Fujin Rd",          "station_id": 128, "x": 60,  "y": 4},

    {"station_name": "Hongqiao",          "station_id": 202, "x": 2,   "y": 52},
    {"station_name": "Songhong Rd",       "station_id": 204, "x": 10,  "y": 52},
    {"station_name": "Beixinjing",        "station_id": 205, "x": 18,  "y": 52},
    {"station_name": "Weining Rd",        "station_id": 206, "x": 26,  "y": 52},
    {"station_name": "Loushanguan Rd",    "station_id": 207, "x": 34,  "y": 52},
    {"station_name": "Zhongshan Park",    "station_id": 208, "x": 42,  "y": 52},
    {"station_name": "Jingansi",          "station_id": 210, "x": 55,  "y": 52},
    {"station_name": "People's Square",   "station_id": 113, "x": 72,  "y": 52},
    {"station_name": "Nanjing E Rd",      "station_id": 213, "x": 80,  "y": 62},
    {"station_name": "Lujiazui",          "station_id": 214, "x": 87,  "y": 71},
    {"station_name": "Century Av",        "station_id": 216, "x": 94,  "y": 80},
    {"station_name": "Longyang Rd",       "station_id": 219, "x": 101, "y": 89},
    {"station_name": "Guanglan Rd",       "station_id": 222, "x": 108, "y": 98},
    {"station_name": "Chuansha",          "station_id": 226, "x": 115, "y": 107},
    {"station_name": "Pudong Airport",    "station_id": 230, "x": 122, "y": 116},

    {"station_name": "South Station",     "station_id": 105, "x": 30,  "y": 94},
    {"station_name": "Longcao Rd",        "station_id": 303, "x": 36,  "y": 94},
    {"station_name": "Yishan Rd",         "station_id": 305, "x": 32,  "y": 80},
    {"station_name": "Zhongshan Park",    "station_id": 208, "x": 42,  "y": 52},
    {"station_name": "Caoyang Rd",        "station_id": 310, "x": 55,  "y": 42},
    {"station_name": "Station",           "station_id": 116, "x": 68,  "y": 36},
    {"station_name": "Hongkou Stadium",   "station_id": 316, "x": 76,  "y": 26},
    {"station_name": "Changjiang S Rd",   "station_id": 321, "x": 76,  "y": 18},
    {"station_name": "Shuichan Rd",       "station_id": 325, "x": 76,  "y": 10},
    {"station_name": "Jiangyang N Rd",    "station_id": 329, "x": 76,  "y": 2},

    {"station_name": "Songjiang Station", "station_id": 901, "x": 2,   "y": 96},
    {"station_name": "Dongjing",          "station_id": 906, "x": 8,   "y": 88},
    {"station_name": "Jiuting",           "station_id": 909, "x": 14,  "y": 80},
    {"station_name": "Hechuan Rd",        "station_id": 913, "x": 22,  "y": 80},
    {"station_name": "Yishan Rd",         "station_id": 305, "x": 32,  "y": 80},
    {"station_name": "Xujiahui",          "station_id": 108, "x": 40,  "y": 80},
    {"station_name": "Zhaojiabang Rd",    "station_id": 918, "x": 52,  "y": 80},
    {"station_name": "Jiashan Rd",        "station_id": 919, "x": 62,  "y": 80},
    {"station_name": "Madang Rd",         "station_id": 921, "x": 72,  "y": 80},
    {"station_name": "Lujiabang Rd",      "station_id": 922, "x": 82,  "y": 80},
    {"station_name": "Century Av",        "station_id": 216, "x": 94,  "y": 80},
    {"station_name": "Yanggao M Rd",      "station_id": 926, "x": 104, "y": 80},
    {"station_name": "Lantian Rd",        "station_id": 928, "x": 112, "y": 80},
    {"station_name": "Jinhai Rd",         "station_id": 932, "x": 120, "y": 80},
    {"station_name": "Caolu",             "station_id": 935, "x": 128, "y": 80},
]

line1 = [(5,115),(17,115),(20,110),(30,94),(40,80),(48,68),(72,68),(72,52),(72,44),(72,36),(68,36),(60,28),(60,4)]
line2 = [(2,52),(42,52),(55,52),(72,52),(80,62),(87,71),(94,80),(101,89),(108,98),(115,107),(122,116)]
line3 = [(76,2),(76,26),(68,36),(63,42),(55,42),(50,42),(42,52),(32,66),(32,80),(32,86),(36,90),(36,94),(30,94)]
line9 = [(2,96),(8,88),(14,80),(32,80),(40,80),(49,80),(60,80),(73,80),(84,80),(94,80),(105,80),(113,80),(121,80),(128,80)]

c1 = "#ff0000"
c2 = "#32cd32"
c3 = "#e6c200"
c9 = "#5aa0d8"

def stretch(points):
    return [(x * X_STRETCH, y) for x, y in points]

line1s = stretch(line1)
line2s = stretch(line2)
line3s = stretch(line3)
line9s = stretch(line9)


def plot_stations(plot_data, save_name, save_fig):
    plt.figure(figsize=(12, 8))

    # ---- plot metro lines ----
    plt.plot(*zip(*line1s), color=c1, linewidth=4.5, solid_capstyle="round", zorder=1)
    plt.plot(*zip(*line2s), color=c2, linewidth=4.5, solid_capstyle="round", zorder=1)
    plt.plot(*zip(*line3s), color=c3, linewidth=4.5, solid_capstyle="round", zorder=1)
    plt.plot(*zip(*line9s), color=c9, linewidth=4.5, solid_capstyle="round", zorder=1)

    # ---- merge station info with data ----
    station_df = pd.DataFrame(stations)
    merged = station_df.merge(plot_data, on="station_id", how="left")
    merged["left_behind_times"] = merged["left_behind_times"].fillna(0)
    max_lb = np.max(merged["left_behind_times"])
    # ---- FIXED size scaling (0 to 6) ----
    LB_MIN = 0
    LB_MAX = 7

    min_size = 80
    max_size = 1200


    # linear scaling
    merged["size"] = min_size + (
        (merged["left_behind_times"] - LB_MIN) / (LB_MAX - LB_MIN)
    ) * (max_size - min_size)

    # ---- plot stations ----
    for _, row in merged.iterrows():
        x = row["x"] * X_STRETCH
        y = row["y"]

        plt.scatter(
            x, y,
            s=row["size"],
            facecolors="#e6e6e6",
            edgecolors="#333333",
            linewidths=1.2,
            zorder=3,
            alpha=0.7
        )
        if row['left_behind_times'] >= 1 or row['left_behind_times']==max_lb:
            plt.text(x-1, y-6, str(int(row['left_behind_times'])), va='center', ha='left', fontsize=18, zorder=11,color='black')

    # ---- legend (FIXED 0–6 scale) ----
    legend_values = list(range(LB_MIN, LB_MAX + 1))
    used_legend = [0,3,6]
    handles = []

    for val in used_legend:
        size = min_size + ((val - LB_MIN) / (LB_MAX - LB_MIN)) * (max_size - min_size)
        handles.append(
            plt.scatter([], [], s=size, facecolors="#e6e6e6", edgecolors="#333333")
        )

    # ---- Manual legend ----
    legend_values = used_legend
    legend_x = 150  # starting x position in data coordinates
    legend_y = 34  # starting y position
    legend_y_list = [36, 43, 52]
    dy = 6  # vertical spacing between legend entries
    legend_scale = 1  # shrink factor for legend markers

    # Draw rectangle frame
    frame_padding = 2
    frame_width = 14
    frame_height = dy * len(legend_values) + 1 * frame_padding + 5
    rect = plt.Rectangle(
        (legend_x - frame_padding+0.5, legend_y - frame_padding+1),  # bottom-left corner
        frame_width,
        frame_height,
        linewidth=1.2,
        edgecolor='black',
        facecolor='white',
        zorder=10
    )
    plt.gca().add_patch(rect)

    # Add title
    plt.text(legend_x + frame_width / 2, legend_y + frame_height - 2 -30,
             "Max Left-behind Times",
             fontsize=18, ha='center', va='top', zorder=11)

    # Add each legend item
    for val, y in zip(legend_values, legend_y_list):
        size = min_size + ((val - LB_MIN) / (LB_MAX - LB_MIN)) * (max_size - min_size)
        size *= legend_scale  # shrink for legend

        plt.scatter(legend_x + 3, y, s=size, facecolors="#e6e6e6", edgecolors="#333333", zorder=11, alpha=0.7)
        plt.text(legend_x + 8, y, str(val), va='center', ha='left', fontsize=18, zorder=11)
    # ---- line labels ----
    plt.text(38 * X_STRETCH, 18, "Line 1", color=c1, fontsize=20)
    plt.text(4 * X_STRETCH, 38, "Line 2", color=c2, fontsize=20)
    plt.text(94 * X_STRETCH-10, 22, "Line 3", color=c3, fontsize=20)
    plt.text(60 * X_STRETCH, 92, "Line 9", color=c9, fontsize=20)


    # ---- North arrow ----
    north_x = 10   # adjust based on your layout
    north_y = 20   # arrow base position

    plt.annotate(
        '',
        xy=(north_x, north_y - 12),   # arrow head (up direction since y is inverted later)
        xytext=(north_x, north_y),    # arrow tail
        arrowprops=dict(facecolor='black', width=2, headwidth=10),
        zorder=12
    )

    plt.text(
        north_x, north_y - 13,
        'N',
        ha='center',
        va='bottom',
        fontsize=20,
        fontweight='bold',
        zorder=12
    )

    # ---- formatting ----
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    # ---- save as PDF ----
    # plt.show()
    if save_fig:
        plt.savefig(f"img/{save_name}.pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def process_res(res):
    target_time = [6*3600, 9*3600]
    res_used = res.loc[
        (res['arrival_time_at_platform']>=target_time[0]) & (res['arrival_time_at_platform']<=target_time[1])]
    res_used['station_id'] = res_used['boarding_platform'].apply(lambda x: x.split('_')[0]).astype(int)
    lb_res = res_used.groupby(['station_id'])['left_behind_times'].max().reset_index()

    return lb_res

if __name__ == "__main__":
    station_df = pd.read_csv("data/reference/platforms.csv")
    optimal_iter = 89
    ####
    initial_res = pd.read_csv("output/reference/left_behind_log_iteration_0.csv")
    initial_plot_data = process_res(initial_res)
    plot_stations(initial_plot_data, save_name = 'initial_max_lb_distribution', save_fig=True)


    final_res = pd.read_csv(f"output/reference/left_behind_log_iteration_{optimal_iter}.csv")
    final_plot_data = process_res(final_res)
    plot_stations(final_plot_data, save_name='final_max_lb_distribution', save_fig=True)
