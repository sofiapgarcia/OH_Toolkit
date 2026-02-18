# imports
from oh_parser import (
    load_profiles,
    extract_nested, extract)
import pandas as pd

# internal imports
from utils import autofill_nan_groups, add_side_column
from kde_plots import kde_plots
from constants import EMG_WEEKLY_PATHS

# Path for OH profiles
OH_PROFILES_PATH = r"D:\Documents\PrevOccupAI\OH_profiles\OH_profiles"
# Path for distribution plots
SAVE_PATH=r"D:\Documents\PrevOccupAI\plots\distributions"
# Set of profiles
profiles = load_profiles(OH_PROFILES_PATH)

# DATA

# smartwatch data

# df with heart rate features
df_heart = extract_nested(
    profiles,
    base_path="sensor_metrics.heart_rate",
    level_names=["date", "session"],
    value_paths=[
        "HR_BPM_stats.*",
        "HR_ratio_stats.*",
        "HR_distributions.*",
    ],
    exclude_patterns=["HR_timeline"]
)

# df with wrist features
df_wrist = extract_nested(
    profiles,
    base_path="sensor_metrics.wrist_activities",
    level_names=["date", "session"],
    value_paths=[
        "WRIST_significant_rotation_percentage",
        "WRIST_significant_acceleration_percentage",

    ],
    exclude_patterns=[]
)

# df with the features from smartwatch data
df_smartwatch = pd.merge(
    df_heart,
    df_wrist,
    on=["subject_id", "work_type", "date", "session"],
    how="outer"
)

df_smartwatch = autofill_nan_groups(df_smartwatch)

# smartphone data

# df with the noise features
df_noise = extract_nested(
    profiles,
    base_path="sensor_metrics.noise",
    level_names=["date", "session"],
    value_paths=[
        "Noise_statistics.*",
        "Noise_distributions.*",
        "Noise_durations.*",
    ],
    exclude_patterns=[
        "Noise_timeline*"
    ]
)

# df with human activity features
df_human = extract_nested(
    profiles,
    base_path="sensor_metrics.human_activities",
    level_names=["date", "session"],
    value_paths=[
        "HAR_distributions.*",
        "HAR_durations.*",
        "HAR_steps.*",
    ],
    exclude_patterns=[
        "HAR_timeline*"
    ]
)

# df with posture features
df_posture = extract_nested(
    profiles,
    base_path="sensor_metrics.posture",
    level_names=["date", "session"]
)

# df with smartphone features without posture data
df_smartphone = pd.merge(
    df_human,
    df_noise,
    on=["subject_id", "work_type", "date", "session"],
    how="outer"
)
# df with smartphone features complete
df_smartphone = pd.merge(
    df_posture,
    df_smartphone,
    on=["subject_id", "work_type", "date", "session"],
    how="outer"
)

df_smartphone = autofill_nan_groups(df_smartphone)

# df with emg data
df_emg = extract_nested(
    profiles,
    base_path="sensor_metrics.emg",
    level_names=["date", "session","side"],
    value_paths=[
        "EMG_intensity.*",
        "EMG_apdf.*",
        "EMG_rest_recovery.*",
        "EMG_relative_bins.*"
    ],
    exclude_patterns=["EMG_daily_metrics", "EMG_weekly_metrics"],
)

# extract daily emg data
df_emg_daily = extract_nested(
    profiles,
    base_path="sensor_metrics.emg",
    level_names=["date", "session","side"],
    value_paths=[
            "EMG_intensity.*",
            "EMG_apdf.*",
            "EMG_rest_recovery.*",
            "EMG_relative_bins.*"
        ],
    exclude_patterns=["EMG_weekly_metrics"],
)

df_emg_daily = df_emg_daily.loc[df_emg_daily["session"] == "EMG_daily_metrics"]

# extract weekly emg data
df_emg_weekly = extract(profiles, paths=EMG_WEEKLY_PATHS)
df_emg_weekly = add_side_column(df_emg_weekly)

# PLOTS
# choose the dataframe to explore
df = df_smartwatch

# columns that do not correspond to metrics
exclude_cols = ["subject_id", "work_type", "date", "session", "side"]

# list of metrics
metrics = [col for col in df.columns if col not in exclude_cols]

# Generate KDE plots for each metric
for metric in metrics:
    kde_plots(df, metric, save_path=None)

