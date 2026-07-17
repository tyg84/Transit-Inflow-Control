from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_REGULAR = "Times-Roman"
FONT_BOLD = "Times-Bold"

try:
    pdfmetrics.registerFont(
        TTFont("TimesNewRoman", "/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    )
    pdfmetrics.registerFont(
        TTFont("TimesNewRoman-Bold", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf")
    )
    FONT_REGULAR = "TimesNewRoman"
    FONT_BOLD = "TimesNewRoman-Bold"
except Exception:
    pass


def _read_counts(case_name, iteration, max_lb=7):
    path = Path(f"output/{case_name}/left_behind_log_iteration_{iteration}.csv")
    df = pd.read_csv(path, usecols=["left_behind_times"])
    total = len(df)
    values = df["left_behind_times"].astype(int)
    counts = {i: int((values == i).sum()) for i in range(max_lb + 1)}
    affected_total = int((values > 0).sum())
    return {
        "iteration": iteration,
        "total": total,
        "mean": float(values.mean()),
        "max": int(values.max()),
        "affected_total": affected_total,
        "ge4": int((values >= 4).sum()),
        "counts": counts,
    }


def _draw_panel(c, x0, y0, width, height, title, x_values, before, after, y_max, y_label):
    before_color = colors.HexColor("#4D4D4D")
    after_color = colors.HexColor("#2F7EBB")
    axis_color = colors.HexColor("#333333")
    grid_color = colors.HexColor("#D9D9D9")

    left_pad = 0.56 * inch
    bottom_pad = 0.48 * inch
    top_pad = 0.34 * inch
    right_pad = 0.08 * inch

    px0 = x0 + left_pad
    py0 = y0 + bottom_pad
    plot_w = width - left_pad - right_pad
    plot_h = height - bottom_pad - top_pad

    if title:
        c.setFont(FONT_BOLD, 12.2)
        c.setFillColor(colors.black)
        c.drawString(x0, y0 + height - 0.16 * inch, title)

    # Grid and y-axis labels
    c.setFont(FONT_REGULAR, 10.6)
    for tick in [0, y_max / 4, y_max / 2, 3 * y_max / 4, y_max]:
        y = py0 + plot_h * tick / y_max
        c.setStrokeColor(grid_color)
        c.setLineWidth(0.35)
        c.line(px0, y, px0 + plot_w, y)
        c.setFillColor(axis_color)
        c.drawRightString(px0 - 0.07 * inch, y - 2.5, f"{tick:.0f}")

    # Axes
    c.setStrokeColor(axis_color)
    c.setLineWidth(0.8)
    c.line(px0, py0, px0 + plot_w, py0)
    c.line(px0, py0, px0, py0 + plot_h)

    group_w = plot_w / len(x_values)
    bar_w = group_w * 0.28
    offset = bar_w * 0.58

    for idx, lb in enumerate(x_values):
        cx = px0 + group_w * (idx + 0.5)
        vals = [(before.get(lb, 0), before_color, -offset), (after.get(lb, 0), after_color, offset)]
        for val, color, dx in vals:
            bar_h = plot_h * val / y_max
            c.setFillColor(color)
            c.setStrokeColor(color)
            c.rect(cx + dx - bar_w / 2, py0, bar_w, bar_h, fill=1, stroke=0)

        c.setFillColor(axis_color)
        c.setFont(FONT_REGULAR, 10.4)
        c.drawCentredString(cx, py0 - 0.18 * inch, str(lb))

    c.setFont(FONT_REGULAR, 11.6)
    c.drawCentredString(px0 + plot_w / 2, y0 + 0.02 * inch, "Left-behind times")

    c.saveState()
    c.translate(x0 + 0.06 * inch, py0 + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, y_label)
    c.restoreState()


def _draw_legend(c, x, y):
    before_color = colors.HexColor("#4D4D4D")
    after_color = colors.HexColor("#2F7EBB")
    c.setFont(FONT_REGULAR, 11.2)
    c.setFillColor(before_color)
    c.rect(x, y, 0.13 * inch, 0.08 * inch, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(x + 0.18 * inch, y - 0.005 * inch, "Before")
    c.setFillColor(after_color)
    c.rect(x + 0.85 * inch, y, 0.13 * inch, 0.08 * inch, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.drawString(x + 1.03 * inch, y - 0.005 * inch, "After")


def _save_before_after_panel(path, title, x_values, before, after, y_max, y_label):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(3.35, 3.12))
    x = np.arange(len(x_values))
    width = 0.38
    before_values = [before.get(lb, 0) for lb in x_values]
    after_values = [after.get(lb, 0) for lb in x_values]

    ax.bar(x - width / 2, before_values, width, label="Before", color="black")
    ax.bar(x + width / 2, after_values, width, label="After", color="tab:blue")
    ax.set_xlabel("Left-behind times")
    ax.set_ylabel(y_label)
    ax.set_xticks(x, [str(lb) for lb in x_values])
    ax.set_ylim(0, y_max)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(pad=0.35)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out}")


def _draw_single_distribution(c, page_w, page_h, title, x_values, values, y_max, bar_color):
    axis_color = colors.HexColor("#333333")
    grid_color = colors.HexColor("#D9D9D9")

    left_pad = 0.62 * inch
    bottom_pad = 0.52 * inch
    top_pad = 0.42 * inch
    right_pad = 0.14 * inch

    px0 = left_pad
    py0 = bottom_pad
    plot_w = page_w - left_pad - right_pad
    plot_h = page_h - bottom_pad - top_pad

    c.setFont(FONT_BOLD, 13.4)
    c.setFillColor(colors.black)
    c.drawCentredString(page_w / 2, page_h - 0.24 * inch, title)

    c.setFont(FONT_REGULAR, 11.4)
    for tick in [0, y_max / 4, y_max / 2, 3 * y_max / 4, y_max]:
        y = py0 + plot_h * tick / y_max
        c.setStrokeColor(grid_color)
        c.setLineWidth(0.35)
        c.line(px0, y, px0 + plot_w, y)
        c.setFillColor(axis_color)
        c.drawRightString(px0 - 0.07 * inch, y - 2.8, f"{tick:.0f}")

    c.setStrokeColor(axis_color)
    c.setLineWidth(0.85)
    c.line(px0, py0, px0 + plot_w, py0)
    c.line(px0, py0, px0, py0 + plot_h)

    group_w = plot_w / len(x_values)
    bar_w = group_w * 0.54
    for idx, lb in enumerate(x_values):
        cx = px0 + group_w * (idx + 0.5)
        val = values.get(lb, 0)
        bar_h = plot_h * val / y_max
        c.setFillColor(bar_color)
        c.setStrokeColor(bar_color)
        c.rect(cx - bar_w / 2, py0, bar_w, bar_h, fill=1, stroke=0)

        c.setFillColor(axis_color)
        c.setFont(FONT_REGULAR, 11.2)
        c.drawCentredString(cx, py0 - 0.19 * inch, str(lb))

    c.setFont(FONT_REGULAR, 12.2)
    c.drawCentredString(px0 + plot_w / 2, 0.08 * inch, "Left-behind times")

    c.saveState()
    c.translate(0.12 * inch, py0 + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Share of records (%)")
    c.restoreState()


def _save_single_distribution(path, title, values, color_hex):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = 3.35 * inch, 3.05 * inch
    c = canvas.Canvas(str(out), pagesize=(page_w, page_h))
    c.setTitle(title)
    _draw_single_distribution(
        c,
        page_w,
        page_h,
        title,
        list(range(0, 8)),
        values,
        100,
        colors.HexColor(color_hex),
    )
    c.showPage()
    c.save()
    print(f"Figure saved to {out}")


def plot_left_behind_distribution_split(
    case_name="reference",
    before_iter=0,
    after_iter=89,
    before_path="img/left_behind_distribution_before_reference.pdf",
    after_path="img/left_behind_distribution_after_reference.pdf",
):
    before = _read_counts(case_name, before_iter)
    after = _read_counts(case_name, after_iter)

    before_all_pct = {k: v / before["total"] * 100 for k, v in before["counts"].items()}
    after_all_pct = {k: v / after["total"] * 100 for k, v in after["counts"].items()}

    _save_single_distribution(
        before_path,
        "Before control",
        before_all_pct,
        "#4D4D4D",
    )
    _save_single_distribution(
        after_path,
        "After control",
        after_all_pct,
        "#2F7EBB",
    )


def plot_left_behind_distribution_subfigures(
    case_name="reference",
    before_iter=0,
    after_iter=89,
    all_path="img/left_behind_distribution_all_reference.pdf",
    positive_path="img/left_behind_distribution_positive_reference.pdf",
):
    before = _read_counts(case_name, before_iter)
    after = _read_counts(case_name, after_iter)

    before_all_pct = {k: v / before["total"] * 100 for k, v in before["counts"].items()}
    after_all_pct = {k: v / after["total"] * 100 for k, v in after["counts"].items()}

    before_aff_pct = {
        k: (v / before["affected_total"] * 100 if before["affected_total"] else 0)
        for k, v in before["counts"].items()
        if k > 0
    }
    after_aff_pct = {
        k: (v / after["affected_total"] * 100 if after["affected_total"] else 0)
        for k, v in after["counts"].items()
        if k > 0
    }

    _save_before_after_panel(
        all_path,
        "",
        list(range(0, 8)),
        before_all_pct,
        after_all_pct,
        100,
        "Share of records (%)",
    )
    _save_before_after_panel(
        positive_path,
        "",
        list(range(1, 8)),
        before_aff_pct,
        after_aff_pct,
        70,
        "Conditional share (%)",
    )


def plot_left_behind_distribution(
    case_name="reference",
    before_iter=0,
    after_iter=89,
    save_path="img/left_behind_distribution_reference.pdf",
):
    before = _read_counts(case_name, before_iter)
    after = _read_counts(case_name, after_iter)

    before_all_pct = {k: v / before["total"] * 100 for k, v in before["counts"].items()}
    after_all_pct = {k: v / after["total"] * 100 for k, v in after["counts"].items()}

    before_aff_pct = {
        k: (v / before["affected_total"] * 100 if before["affected_total"] else 0)
        for k, v in before["counts"].items()
        if k > 0
    }
    after_aff_pct = {
        k: (v / after["affected_total"] * 100 if after["affected_total"] else 0)
        for k, v in after["counts"].items()
        if k > 0
    }

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = landscape((6.7 * inch, 3.55 * inch))
    c = canvas.Canvas(str(out), pagesize=(page_w, page_h))
    c.setTitle("Distribution of left-behind times")

    margin = 0.28 * inch
    panel_gap = 0.28 * inch
    panel_w = (page_w - 2 * margin - panel_gap) / 2
    panel_h = page_h - 0.70 * inch
    y0 = 0.26 * inch

    _draw_legend(c, page_w / 2 - 0.75 * inch, page_h - 0.26 * inch)

    _draw_panel(
        c,
        margin,
        y0,
        panel_w,
        panel_h,
        "(a) All passenger-platform records",
        list(range(0, 8)),
        before_all_pct,
        after_all_pct,
        100,
        "Share of records (%)",
    )

    _draw_panel(
        c,
        margin + panel_w + panel_gap,
        y0,
        panel_w,
        panel_h,
        "(b) Positive left-behind records",
        list(range(1, 8)),
        before_aff_pct,
        after_aff_pct,
        70,
        "Conditional share (%)",
    )

    c.showPage()
    c.save()
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    plot_left_behind_distribution_subfigures()
