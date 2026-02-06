"""
Functions to generate kde plots per metric.

Available Functions
-------------------
[Public]
kde_plots(...): Plot KDE distributions for a given metric.

------------------
[Private]

------------------
"""
# ------------------------------------------------------------------------------------------------------------------- #
# imports
# ------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def kde_plots(df, metric):
    """
    Plot KDE distributions for a given metric, grouped by weekday and session order per subject in case of per session data.

    If multiple sessions exist, the y-axis shows the session order with labels (I, II, III, IV).
    If only one session exists, the y-axis is labeled as 'Densidade de probabilidade' and session labels are hidden.

    :param df: pandas DataFrame containing session-level data. Must include columns:
               - 'subject_id'
               - 'work_type'
               - 'date' (format '%d-%m-%Y')
               - 'session' (format '%H-%M-%S')
               - the metric column to plot
    :param metric: str, the column name of the metric to plot.
    :returns: None. Displays a Seaborn displot with KDE distributions.
    """

    data = df.copy()

    # Convert 'date' column to datetime
    data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y")

    # Convert 'session' column to time
    data["session_time"] = pd.to_datetime(
        data["session"], format="%H-%M-%S", errors="coerce"
    ).dt.time

    # Extract weekday name
    data["weekday_en"] = data["date"].dt.day_name()

    # Map weekdays to Portuguese
    data["weekday"] = data["weekday_en"].map(weekday_pt_map)

    # Ensure weekday order is consistent
    data["weekday"] = pd.Categorical(data["weekday"], categories=weekday_order_pt, ordered=True)

    # Sort data by subject, date, and session time
    data = data.sort_values(["subject_id", "date", "session_time"])

    # Assign session order labels (I, II, III, IV) per subject per day
    data["session_order"] = (
        data.groupby(["subject_id", "date"])
            .cumcount()
            .map(lambda x: session_labels[x] if x < 4 else None)
    )

    # Drop rows without session order
    data = data.dropna(subset=["session_order"])

    # Get readable metric name for plot title
    metric_title = METRIC_READABLE_MAP.get(metric, metric)

    # Count unique sessions to adjust y-axis behavior
    unique_sessions = data["session_order"].nunique()

    # Generate KDE plot
    g = sns.displot(
        data=data,
        x=metric,
        hue="work_type",
        col="weekday",
        row="session_order" if unique_sessions > 1 else None,  # facet by session only if more than one
        kind="kde",
        fill=True,
        common_norm=False,
        palette=palette,
        height=2.3,
        aspect=1.2
    )

    # Remove facet titles
    g.set_titles("")

    # Set global x-axis label
    g.fig.text(0.5, 0.03, "Dia da semana", ha='center', fontsize=14)

    # Set y-axis label depending on number of sessions
    if unique_sessions > 1:
        g.fig.text(0.03, 0.5, "Número da sessão", va='center', rotation='vertical', fontsize=14)
    else:
        g.fig.text(0.03, 0.5, "Densidade de probabilidade", va='center', rotation='vertical', fontsize=14)

    # Identify bottom row axes for labeling
    if unique_sessions > 1:
        bottom_axes = g.axes[-1, :]
    else:
        # Flatten axes for a single session
        if isinstance(g.axes, np.ndarray):
            bottom_axes = g.axes.ravel()
        else:
            bottom_axes = [g.axes]

    # Set weekday names on x-axis
    for ax, day in zip(bottom_axes, weekday_order_pt):
        ax.set_xlabel(day, fontsize=11)

    if unique_sessions == 1:
        for ax in bottom_axes:
            ax.set_ylabel("")

    # Set session labels on y-axis only if multiple sessions
    if unique_sessions > 1:
        for ax, label in zip(g.axes[:, 0], session_labels):
            ax.set_ylabel(label, rotation=0, labelpad=30, fontsize=11, va="center")

    # Adjust figure size
    g.fig.set_size_inches(18, 10)

    # Set global plot title
    g.fig.suptitle(f"Distribuição da métrica: {metric_title}", fontsize=18, y=0.98)

    # Adjust legend inside figure
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((0.95, 0.95))
        g._legend.set_frame_on(True)
        g._legend.set_title("Tipo de trabalho")

    # Apply tight layout
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.94])

    # Display the plot
    plt.show()


