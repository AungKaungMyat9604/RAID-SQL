#!/usr/bin/env python3
"""Render difficulty-classifier confusion matrix as PNG (dark theme, viva-ready)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_JSON = ROOT / "outputs" / "test_raid_v2_values" / "confusion_matrices.json"
DEFAULT_PNG = ROOT / "outputs" / "test_raid_v2_values" / "confusion_matrix.png"

# Match HTML deck heatmap (.heatmap in Final_Presentation.html)
COLORS = {
    "bg": "#0F172A",
    "panel": "#1E293B",
    "ink": "#F1F5F9",
    "muted": "#94A3B8",
    "accent": "#3B82F6",
    "diag_fg": "#FFFFFF",
    "high_bg": "#991B1B",
    "high_fg": "#FCA5A5",
    "mid_bg": "#92400E",
    "mid_fg": "#FDE68A",
    "low_bg": "#1E293B",
    "low_fg": "#94A3B8",
}


def cell_style(row: str, col: str, value: int) -> tuple[str, str]:
    """Background and text colour per cell (diagonal / off-diagonal severity)."""
    if row == col:
        return COLORS["accent"], COLORS["diag_fg"]
    if row == "NESTED" and col == "EASY":
        return COLORS["high_bg"], COLORS["high_fg"]
    if value >= 86:
        return COLORS["mid_bg"], COLORS["mid_fg"]
    return COLORS["low_bg"], COLORS["low_fg"]


def load_matrix(path: Path) -> tuple[list[str], list[list[int]], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    diff = data["difficulty_classifier"]
    classes: list[str] = diff["classes"]
    raw = diff["matrix_rows_predicted_cols_gold"]
    grid = [[int(raw[r][c]) for c in classes] for r in classes]
    return classes, grid, diff


def plot_matrix(
    classes: list[str],
    grid: list[list[int]],
    meta: dict,
    out_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = meta.get("diagonal_correct", sum(grid[i][i] for i in range(len(classes))))
    total = sum(sum(row) for row in grid)
    acc_pct = meta.get("accuracy_pct", round(100 * n / total, 1))

    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.axis("off")

    fig.suptitle(
        "Difficulty Classifier Confusion Matrix",
        fontsize=18,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.96,
    )
    ax.text(
        0.5,
        0.90,
        f"Predicted vs gold-proxy hardness · diagonal accuracy {acc_pct}% ({n}/{total})",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )

    # Table: row 0 = header, rows 1..3 = data, last row = column totals
    col_labels = ["Pred ↓ / Gold →", *classes, "Total"]
    row_totals = [sum(row) for row in grid]
    col_totals = [sum(grid[r][c] for r in range(len(classes))) for c in range(len(classes))]

    n_rows = len(classes) + 2
    n_cols = len(classes) + 2
    table_data: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]

    table_data[0][0] = col_labels[0]
    for j, c in enumerate(classes):
        table_data[0][j + 1] = c
    table_data[0][-1] = "Total"

    cell_colors: list[list[str]] = [[COLORS["panel"]] * n_cols for _ in range(n_rows)]
    cell_text_colors: list[list[str]] = [[COLORS["muted"]] * n_cols for _ in range(n_rows)]

    for i, pred in enumerate(classes):
        table_data[i + 1][0] = pred
        for j, gold in enumerate(classes):
            val = grid[i][j]
            table_data[i + 1][j + 1] = str(val)
            bg, fg = cell_style(pred, gold, val)
            cell_colors[i + 1][j + 1] = bg
            cell_text_colors[i + 1][j + 1] = fg
        table_data[i + 1][-1] = str(row_totals[i])
        cell_colors[i + 1][0] = COLORS["panel"]
        cell_text_colors[i + 1][0] = COLORS["ink"]
        cell_colors[i + 1][-1] = COLORS["panel"]
        cell_text_colors[i + 1][-1] = COLORS["muted"]

    table_data[-1][0] = "Total"
    for j in range(len(classes)):
        table_data[-1][j + 1] = str(col_totals[j])
    table_data[-1][-1] = str(total)
    for j in range(n_cols):
        cell_colors[-1][j] = COLORS["panel"]
        cell_text_colors[-1][j] = COLORS["muted"]
    cell_text_colors[-1][0] = COLORS["ink"]

    tbl = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.22, 0.9, 0.58],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(cell_colors[row][col])
        cell.set_edgecolor("#334155")
        cell.set_linewidth(1.2)
        text = cell.get_text()
        text.set_color(cell_text_colors[row][col])
        if row > 0 and row < n_rows - 1 and col > 0 and col < n_cols - 1:
            text.set_fontweight("bold")
            text.set_fontsize(14)
        if row == 0 or col == 0:
            text.set_fontweight("bold")

    note = (
        "201 gold-EASY predicted NESTED — largest off-diagonal block; "
        "mis-routes prompts before SQL generation."
    )
    ax.text(
        0.5,
        0.08,
        note,
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["muted"],
        wrap=True,
    )

    legend_items = [
        mpatches.Patch(facecolor=COLORS["accent"], edgecolor="#334155", label="Diagonal (agree)"),
        mpatches.Patch(facecolor=COLORS["high_bg"], edgecolor="#334155", label="High off-diagonal"),
        mpatches.Patch(facecolor=COLORS["mid_bg"], edgecolor="#334155", label="Mid off-diagonal"),
        mpatches.Patch(facecolor=COLORS["low_bg"], edgecolor="#334155", label="Low off-diagonal"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=COLORS["muted"],
        bbox_to_anchor=(0.5, 0.01),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor=COLORS["bg"], bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot classifier confusion matrix PNG")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_JSON,
        help="confusion_matrices.json path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PNG,
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Missing {args.input}. Run: python scripts/compute_confusion_matrices.py", file=sys.stderr)
        return 1

    classes, grid, meta = load_matrix(args.input)
    out = plot_matrix(classes, grid, meta, args.output, dpi=args.dpi)
    print(f"Wrote {out} ({args.dpi} DPI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
