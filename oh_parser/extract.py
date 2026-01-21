"""
OH Parser Extraction Functions.

Functions to extract data from OH profiles into pandas DataFrames.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .filters import (
    apply_subject_filters,
    exclude_keys,
    filter_date_keys,
)
from .path_resolver import (
    expand_wildcards,
    list_keys_at_path,
    path_exists,
    resolve_path,
)
from .utils import flatten_dict, is_date_key, is_time_key, print_tree, get_nested_keys


def extract(
    profiles: Dict[str, dict],
    paths: Dict[str, str],
    filters: Optional[Dict[str, Any]] = None,
    include_subject_id: bool = True,
) -> pd.DataFrame:
    """
    Extract specific paths from profiles into a wide-format DataFrame.
    
    One row per subject, with columns for each extracted path.
    
    :param profiles: Dictionary mapping subject_id -> profile dict.
    :param paths: Mapping of output column names to dot-notation paths.
    :param filters: Optional extraction filters.
    :param include_subject_id: Whether to include subject_id column.
    :returns: DataFrame with one row per subject.
    
    Example:
        >>> extract(profiles, {
        ...     "emg_p50_left": "sensor_metrics.emg.EMG_weekly_metrics.left.EMG_apdf.active.p50",
        ...     "age": "meta_data.age",
        ... })
    """
    filtered_profiles = apply_subject_filters(profiles, filters)
    
    rows = []
    for subject_id, profile in filtered_profiles.items():
        row = {}
        if include_subject_id:
            row["subject_id"] = subject_id
        
        for col_name, path in paths.items():
            row[col_name] = resolve_path(profile, path)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def extract_nested(
    profiles: Dict[str, dict],
    base_path: str,
    level_names: List[str],
    value_paths: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    exclude_patterns: Optional[List[str]] = None,
    flatten_values: bool = True,
    include_subject_id: bool = True,
    include_work_type: bool = True,
) -> pd.DataFrame:
    """
    Extract nested structures into a long-format DataFrame.

    Iterates through nested levels (dates, sessions, sides, etc.) and extracts
    values from each leaf, producing one row per combination.

    :param profiles: Dictionary mapping subject_id -> profile dict.
    :param base_path: Starting path (e.g., "sensor_metrics.emg").
    :param level_names: Names for each nesting level (e.g., ["date", "session", "side"]).
    :param value_paths: Paths to extract relative to the leaf level (None = extract all).
                        Supports wildcards: "EMG_intensity.*" extracts all keys under EMG_intensity.
    :param filters: Optional extraction filters.
    :param exclude_patterns: Patterns to exclude at each level (e.g., ["EMG_*_metrics"]).
    :param flatten_values: If True, flatten nested value dicts into columns.
    :param include_subject_id: Whether to include subject_id column.
    :returns: Long-format DataFrame with one row per leaf node.

    Example:
        >>> extract_nested(
        ...     profiles,
        ...     base_path="sensor_metrics.emg",
        ...     level_names=["date", "session", "side"],
        ...     value_paths=["EMG_intensity.mean_percent_mvc", "EMG_rest_recovery.rest_percent"],
        ...     exclude_patterns=["EMG_daily_metrics", "EMG_weekly_metrics"],
        ... )
    """
    filtered_profiles = apply_subject_filters(profiles, filters)
    exclude_patterns = exclude_patterns or []
    rows = []

    for subject_id, profile in filtered_profiles.items():
        base_data = resolve_path(profile, base_path)
        if base_data is None or not isinstance(base_data, dict):
            continue

        context = {}

        if include_subject_id:
            context["subject_id"] = subject_id

        if include_work_type:
            context["work_type"] = resolve_path(profile, "meta_data.work_type")

        _extract_levels(
            data=base_data,
            level_names=level_names,
            level_idx=0,
            context=context,
            value_paths=value_paths,
            exclude_patterns=exclude_patterns,
            flatten_values=flatten_values,
            rows=rows,
            filters=filters,
        )

    if not rows:
        columns = []
        if include_subject_id:
            columns.append("subject_id")
        if include_work_type:
            columns.append("work_type")
        columns += level_names
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows)


def _extract_levels(
    data: dict,
    level_names: List[str],
    level_idx: int,
    context: Dict[str, Any],
    value_paths: Optional[List[str]],
    exclude_patterns: List[str],
    flatten_values: bool,
    rows: List[dict],
    filters: Optional[Dict[str, Any]],
) -> None:
    """
    Recursive helper to iterate through nested levels.

    Internal function - do not call directly.
    """
    if level_idx >= len(level_names):
        # At leaf level - extract values
        row = context.copy()

        if value_paths is None:
            # Extract all values from current dict
            if flatten_values and isinstance(data, dict):
                flat = flatten_dict(data, sep=".")
                row.update(flat)
            else:
                row["_data"] = data
        else:
            # Extract specific paths
            for vpath in value_paths:
                if vpath.endswith(".*"):
                    # Wildcard: extract all keys under this path
                    parent_path = vpath[:-2]
                    parent_data = resolve_path(data, parent_path) if parent_path else data
                    if isinstance(parent_data, dict):
                        for key, value in parent_data.items():
                            col_name = f"{parent_path}.{key}" if parent_path else key
                            if isinstance(value, dict) and flatten_values:
                                flat = flatten_dict(value, parent_key=col_name, sep=".")
                                row.update(flat)
                            else:
                                row[col_name] = value
                else:
                    # Exact path
                    value = resolve_path(data, vpath)
                    if isinstance(value, dict) and flatten_values:
                        flat = flatten_dict(value, parent_key=vpath, sep=".")
                        row.update(flat)
                    else:
                        row[vpath] = value

        rows.append(row)
        return

    # Not at leaf - iterate through keys at this level
    current_level_name = level_names[level_idx]

    if not isinstance(data, dict):
        return

    # Get keys, apply exclusions
    keys = list(data.keys())
    keys = exclude_keys(keys, exclude_patterns)

    # Apply date filtering if this looks like a date level
    if filters and filters.get("date_range"):
        keys = filter_date_keys(keys, filters["date_range"])

    for key in keys:
        child_data = data[key]
        if not isinstance(child_data, dict):
            continue

        new_context = context.copy()
        new_context[current_level_name] = key

        _extract_levels(
            data=child_data,
            level_names=level_names,
            level_idx=level_idx + 1,
            context=new_context,
            value_paths=value_paths,
            exclude_patterns=exclude_patterns,
            flatten_values=flatten_values,
            rows=rows,
            filters=filters,
        )


def extract_flat(
    profiles: Dict[str, dict],
    base_path: str,
    filters: Optional[Dict[str, Any]] = None,
    max_depth: int = 10,
    include_subject_id: bool = True,
) -> pd.DataFrame:
    """
    Extract all values under a base path into a fully flattened DataFrame.

    Creates columns using dot-notation for all nested keys.
    One row per subject.

    :param profiles: Dictionary mapping subject_id -> profile dict.
    :param base_path: Starting path (e.g., "sensor_metrics.emg.EMG_weekly_metrics").
    :param filters: Optional extraction filters.
    :param max_depth: Maximum nesting depth to flatten.
    :param include_subject_id: Whether to include subject_id column.
    :returns: Wide-format DataFrame with flattened columns.
    """
    filtered_profiles = apply_subject_filters(profiles, filters)

    rows = []
    for subject_id, profile in filtered_profiles.items():
        data = resolve_path(profile, base_path)
        if data is None:
            continue

        row = {}
        if include_subject_id:
            row["subject_id"] = subject_id

        if isinstance(data, dict):
            flat = flatten_dict(data, sep=".", max_depth=max_depth)
            row.update(flat)
        else:
            row[base_path] = data

        rows.append(row)

    return pd.DataFrame(rows)


def get_available_paths(
    profile: dict,
    base_path: str = "",
    max_depth: int = 6,
) -> List[str]:
    """
    Get all available paths from a single profile.

    :param profile: Single OH profile dictionary.
    :param base_path: Starting path (empty for root).
    :param max_depth: Maximum depth to traverse.
    :returns: List of all dot-notation paths.
    """

    data = resolve_path(profile, base_path) if base_path else profile
    if not isinstance(data, dict):
        return [base_path] if base_path else []

    paths = get_nested_keys(data, max_depth=max_depth)

    if base_path:
        paths = [f"{base_path}.{p}" for p in paths]

    return paths


def inspect_profile(
    profile: dict,
    base_path: str = "",
    max_depth: int = 4,
    show_values: bool = False,
) -> None:
    """
    Pretty-print the structure of a profile.

    :param profile: Single OH profile dictionary.
    :param base_path: Starting path (empty for root).
    :param max_depth: Maximum depth to display.
    :param show_values: Whether to show leaf values.
    """

    data = resolve_path(profile, base_path) if base_path else profile

    if base_path:
        print(f"Structure at '{base_path}':")
    else:
        print("Profile structure:")

    print_tree(data, max_depth=max_depth, show_values=show_values)


def summarize_profiles(
    profiles: Dict[str, dict],
    check_paths: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Generate a summary of all profiles showing data availability.

    :param profiles: Dictionary mapping subject_id -> profile dict.
    :param check_paths: Specific paths to check for (None = use defaults).
    :returns: DataFrame with one row per subject showing which data is available.
    """
    if check_paths is None:
        # Default paths to check
        check_paths = [
            "meta_data",
            "single_instance_questionnaires",
            "daily_questionnaires",
            "sensor_metrics.emg",
            "sensor_metrics.emg.EMG_weekly_metrics",
            "sensor_timeline",
            "human_activities",
        ]

    rows = []
    for subject_id, profile in profiles.items():
        row: Dict[str, Any] = {"subject_id": subject_id}
        for path in check_paths:
            # Use a cleaner column name
            col_name = path.split(".")[-1] if "." in path else path
            row[f"has_{col_name}"] = path_exists(profile, path)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("subject_id", ignore_index=True)