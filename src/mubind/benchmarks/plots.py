"""Property-control figures from the current evaluation's candidates."""

from .common import PROPERTIES


def make_figures(out, summary, *, records):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = ("length_only", "charge_only", "hydrophobicity_only")
    labels = ("Length (residues)", "Net charge", "Mean hydrophobicity")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), layout="constrained")
    for col, prop in enumerate(PROPERTIES):
        for row in range(2):
            setting = settings[col] if row == 0 else "all_properties"
            rows = [
                r
                for r in records
                if r["property_conditioning"] == setting and r["sequence_valid"]
            ]
            x = [r[f"requested_{prop}"] for r in rows]
            y = [r[f"observed_{prop}"] for r in rows]
            ax = axes[row, col]
            ax.scatter(
                x,
                y,
                s=10,
                alpha=0.35,
                color="#2168a6" if row == 0 else "#188478",
                edgecolors="none",
                rasterized=True,
            )
            low, high = min(x + y), max(x + y)
            padding = max((high - low) * 0.06, 0.01)
            bounds = (low - padding, high + padding)
            ax.plot(bounds, bounds, "--", color="#333333", linewidth=1)
            ax.set(
                xlim=bounds,
                ylim=bounds,
                xlabel="Requested " + labels[col].lower(),
                ylabel="Generated " + labels[col].lower(),
                title=labels[col] + " only" if row == 0 else "All three properties",
            )
            metric = summary["properties"]["settings"][setting]["metrics"][prop]
            ax.text(
                0.04,
                0.96,
                f"MAE {metric['mae']:.3f}\nStandardized {metric['standardized_mae']:.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"property_steering.{suffix}", dpi=180)
    plt.close(fig)
