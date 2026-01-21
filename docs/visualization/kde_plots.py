import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Map technical metric names to readable names
metric_readable_map = {
    "HR_BPM_stats.mean": "Média do batimento cardíaco",
    "HR_BPM_stats.std": "Desvio padrão do batimento cardíaco",
    "HR_ratio_stats.mean": "Média da razão cardíaca",
    "HR_ratio_stats.std": "Desvio padrão da razão cardíaca",
    "WRIST_significant_rotation_percentage": "Percentagem de rotação significativa",
    "HAR_distributions.Sentado": "Percentagem do tempo Sentado",
}

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



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def kde_by_weekday_and_session_order(df, metric):
    """
    Plot KDE distributions grouped by weekday and ordered session within each subject,
    with weekdays in Portuguese and readable metric names in the title.
    If only one session exists, replace 'Número da sessão' with 'Densidade de probabilidade' and hide y-axis labels.
    """

    data = df.copy()

    # Convert date and session time
    data["date"] = pd.to_datetime(data["date"], format="%d-%m-%Y")
    data["session_time"] = pd.to_datetime(
        data["session"], format="%H-%M-%S", errors="coerce"
    ).dt.time

    # Extract weekday in English first
    data["weekday_en"] = data["date"].dt.day_name()
    data["weekday"] = data["weekday_en"].map(weekday_pt_map)
    data["weekday"] = pd.Categorical(data["weekday"], categories=weekday_order_pt, ordered=True)

    # Order sessions chronologically
    data = data.sort_values(["subject_id", "date", "session_time"])

    # Assign session order
    data["session_order"] = (
        data.groupby(["subject_id", "date"])
            .cumcount()
            .map(lambda x: session_labels[x] if x < 4 else None)
    )
    data = data.dropna(subset=["session_order"])

    metric_title = metric_readable_map.get(metric, metric)

    # Check number of unique session orders
    unique_sessions = data["session_order"].nunique()

    # KDE plot
    g = sns.displot(
        data=data,
        x=metric,
        hue="work_type",
        col="weekday",
        row="session_order" if unique_sessions > 1 else None,  # only facet by session if more than 1
        kind="kde",
        fill=True,
        common_norm=False,
        palette=palette,
        height=2.3,
        aspect=1.2
    )

    # Remove facet titles
    g.set_titles("")

    # Global axis labels
    g.fig.text(0.5, 0.03, "Dia da semana", ha='center', fontsize=14)  # X-axis
    if unique_sessions > 1:
        g.fig.text(0.03, 0.5, "Número da sessão", va='center', rotation='vertical', fontsize=14)  # Y-axis
    else:
        g.fig.text(0.03, 0.5, "Densidade de probabilidade", va='center', rotation='vertical', fontsize=14)

    # Set weekday names only on bottom row
    if unique_sessions > 1:
        bottom_axes = g.axes[-1, :]
    else:
        if isinstance(g.axes, np.ndarray):
            bottom_axes = g.axes.ravel()
        else:
            bottom_axes = [g.axes]

    for ax, day in zip(bottom_axes, weekday_order_pt):
        ax.set_xlabel(day, fontsize=11)

    # Remove automatic 'Density' ylabel if only one session
    if unique_sessions == 1:
        for ax in bottom_axes:
            ax.set_ylabel("")  # remove Seaborn's default 'Density'

    # Set session labels on y-axis only if multiple sessions
    if unique_sessions > 1:
        for ax, label in zip(g.axes[:, 0], session_labels):
            ax.set_ylabel(label, rotation=0, labelpad=30, fontsize=11, va="center")

    # Increase figure size
    g.fig.set_size_inches(18, 10)

    # Global title with readable metric name
    g.fig.suptitle(f"Distribuição da métrica: {metric_title}", fontsize=18, y=0.98)

    # Legend inside figure, compact
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((0.95, 0.95))
        g._legend.set_frame_on(True)
        g._legend.set_title("Tipo de trabalho")

    # Layout
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.94])
    plt.show()


