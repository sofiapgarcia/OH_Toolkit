"""
Functions to generate kde plots per metric.

Available Functions
-------------------
[Public]
kde_plots(...): Plot KDE distributions for a given metric.

------------------
[Private]
_prepare_dataframe(...): Preprocess dataframe for KDE plotting.
_plot_kde_by_week(...): Plot a simple KDE for weekly data.
_plot_kde_by_weekday(...): Plot KDE with facets by weekday.
------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
# internal imports
from constants import METRIC_READABLE_MAP


# ------------------------------------------------------------------------------------------------------------------- #
# file specific constants
# ------------------------------------------------------------------------------------------------------------------- #

# Map weekdays to Portuguese
weekday_pt_map = {
    "Monday": "Segunda",
    "Tuesday": "Terça",
    "Wednesday": "Quarta",
    "Thursday": "Quinta",
    "Friday": "Sexta"
}

# Enforce weekday order
weekday_order_pt = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]

# Assign session order labels
session_labels = ["I", "II", "III", "IV"]

# Color palette
palette = {"BO": "#1f77b4", "FO": "#d62728"}

# ------------------------------------------------------------------------------------------------------------------- #
# public functions
# ------------------------------------------------------------------------------------------------------------------- #

def kde_plots(df, metric, save_path=None):
    """
    Plot KDE distributions for a given metric.

    If the dataframe contains a 'side' column, separate plots are
    automatically generated for each side (e.g., left and right).
    If the dataframe lacks date/session info, a single simple KDE is generated.

    :param df: pandas DataFrame containing metric data. Must include:
               - 'subject_id'
               - 'work_type'
               - 'date' (format '%d-%m-%Y') optional
               - 'session' (format '%H-%M-%S') optional
               - the metric column to plot
               - optional: 'side'
    :param metric: str, the column name of the metric to plot
    :param save_path: str or None, directory where figures should be saved
    :returns: None. Displays one or multiple Seaborn KDE plots.
    """
    # If 'side' column exists, create one plot per side (for EMG data)
    if "side" in df.columns:
        for side_value in df["side"].dropna().unique():
            df_side = df[df["side"] == side_value]
            # Decide whether to facet by weekday or week
            if "date" in df_side.columns:
                _plot_kde_by_weekday(df_side, metric, save_path, side_label=side_value)
            else:
                _plot_kde_by_week(df_side, metric, save_path, side_label=side_value)
    else:
        # No side column, decide plot type
        if "date" in df.columns:
            _plot_kde_by_weekday(df, metric, save_path)
        else:
            _plot_kde_by_week(df, metric, save_path)

# ------------------------------------------------------------------------------------------------------------------- #
# private functions
# ------------------------------------------------------------------------------------------------------------------- #

def _prepare_dataframe(df):
    """
    Preprocess dataframe for KDE plotting.

    Converts date and session columns to datetime/time,
    maps weekdays to Portuguese, and assigns session order labels.

    :param df: pandas DataFrame
    :returns: pandas DataFrame with additional columns:
              - 'weekday' (Portuguese weekday)
              - 'session_order' (I, II, III, IV)
    """
    data = df.copy()

    # Convert date and map weekdays
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y")
        data["weekday_en"] = data["date"].dt.day_name()
        data["weekday"] = data["weekday_en"].map(weekday_pt_map)
        data["weekday"] = pd.Categorical(
            data["weekday"], categories=weekday_order_pt, ordered=True
        )

    # Convert session to time
    if "session" in data.columns:
        data["session_time"] = pd.to_datetime(
            data["session"], format="%H-%M-%S", errors="coerce"
        ).dt.time

    # Assign session order labels per subject per day
    if "date" in data.columns and "session" in data.columns:
        data = data.sort_values(["subject_id", "date", "session_time"])
        data["session_order"] = (
            data.groupby(["subject_id", "date"])
                .cumcount()
                .map(lambda x: session_labels[x] if x < 4 else None)
        )
        # Drop rows without a valid session order
        data = data.dropna(subset=["session_order"])

    return data


def _plot_kde_by_week(df, metric, save_path=None, side_label=None):
    """
    Plot a simple KDE.

    :param df: pandas DataFrame containing metric data.
               May optionally contain 'work_type' for hue separation.
    :param metric: str, column name of the metric to plot
    :param save_path: str or None, directory to save figure
    :param side_label: str or None, side name for title and filename
    :returns: None. Displays the plot.
    """

    plt.figure(figsize=(8, 5))

    ax = sns.kdeplot(
        data=df,
        x=metric,
        hue="work_type",
        fill=True,
        common_norm=False,
        palette=palette
    )

    # Build title
    metric_title = METRIC_READABLE_MAP.get(metric, metric)
    title = metric_title

    if side_label:
        side_pt = "esquerdo" if side_label == "left" else "direito"
        title += f" - {side_pt}"

    plt.title(f"Distribuição da métrica: {title}", fontsize=11)

    # Remove x-axis label
    plt.xlabel("")

    # Y-axis label
    plt.ylabel("Densidade de probabilidade")

    # Only adjust legend title if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Tipo de trabalho")

    # Save figure if needed
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"kde_{metric}"
        if side_label:
            filename += f"_{side_label}"
        filename += ".png"
        plt.savefig(os.path.join(save_path, filename))

    plt.tight_layout()
    plt.show()



def _plot_kde_by_weekday(df, metric, save_path=None, side_label=None):
    """
    Plot KDE with facets by weekday.

    :param df: pandas DataFrame with date, session, metric, and work_type columns
    :param metric: str, column name of the metric to plot
    :param save_path: str or None, directory to save figure
    :param side_label: str or None, side name for title and filename
    :returns: None. Displays the Seaborn displot.
    """
    # Prepare dataframe
    data = _prepare_dataframe(df)

    # Count unique sessions to determine row facets
    unique_sessions = data["session_order"].nunique() if "session_order" in data.columns else 0
    row_var = "session_order" if unique_sessions > 1 else None

    # Create KDE displot
    g = sns.displot(
        data=data,
        x=metric,
        hue="work_type",
        col="weekday" if "weekday" in data.columns else None,
        row=row_var,
        kind="kde",
        fill=True,
        common_norm=False,
        palette=palette,
        height=2.3,
        aspect=1.2
    )

    # Remove facet titles
    g.set_titles("")

    # Remove internal y-labels
    for ax_row in g.axes:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_ylabel("")
        else:
            ax_row.set_ylabel("")

    # Set global x/y labels
    g.fig.text(0.5, 0.03, "Dia da semana", ha='center', fontsize=14)
    ylabel = "Número da sessão" if unique_sessions > 1 else "Densidade de probabilidade"
    g.fig.text(0.03, 0.5, ylabel, va='center', rotation='vertical', fontsize=14)

    # Label bottom row axes with weekday names
    if "weekday" in data.columns:
        bottom_axes = g.axes[-1, :] if unique_sessions > 1 else g.axes.ravel()
        for ax, day in zip(bottom_axes, weekday_order_pt):
            ax.set_xlabel(day, fontsize=11)

    # Set session labels on y-axis if multiple sessions
    if unique_sessions > 1:
        for ax, label in zip(g.axes[:, 0], session_labels):
            ax.set_ylabel(label, rotation=0, labelpad=30, fontsize=11, va="center")

    # Adjust figure size and title
    g.fig.set_size_inches(18, 10)
    metric_title = METRIC_READABLE_MAP.get(metric, metric)
    title = metric_title
    if side_label:
        side_pt = "esquerdo" if side_label == "left" else "direito"
        title += f" ({side_pt})"
    g.fig.suptitle(f"Distribuição da métrica: {title}", fontsize=18, y=0.98)

    # Adjust legend
    if g._legend:
        g._legend.set_bbox_to_anchor((0.95, 0.95))
        g._legend.set_frame_on(True)
        g._legend.set_title("Tipo de trabalho")

    # Apply tight layout
    g.fig.subplots_adjust(left=0.10, right=0.88, bottom=0.12, top=0.92)

    # Save figure if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"kde_{metric}"
        if side_label:
            filename += f"_{side_label}"
        filename += ".png"
        g.fig.savefig(os.path.join(save_path, filename))

    # Show plot
    plt.show()