# imports
from oh_parser import (
    load_profiles,
    list_subjects,
    extract_nested)
import pandas as pd

# internal imports
from utils import autofill_nan_groups
from kde_plots import kde_plots

# Path for OH profiles
OH_PROFILES_PATH = r"C:\Users\Sofia\Desktop\oh_profile"
# Set of profiles
profiles = load_profiles(OH_PROFILES_PATH)

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

# df with human activity faetures
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

# df with smartphone features
df_smartphone = pd.merge(
    df_human,
    df_noise,
    on=["subject_id", "work_type", "date", "session"],
    how="outer"
)

df_smartphone = autofill_nan_groups(df_smartphone)

# PLOTS

kde_plots(df_smartphone, "HAR_distributions.Sentado")
#kde_by_weekday_and_session_order(df_smartwatch, "HR_BPM_stats.std")