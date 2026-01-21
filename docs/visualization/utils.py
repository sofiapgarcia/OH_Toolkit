def autofill_nan_groups(df, min_group_size=2):
    """
    Fill NaN values within groups of related columns based on a shared prefix.

    For each group of columns that share the same prefix (everything before the
    last '.'), the behavior is:

    - If ALL columns in the group are NaN for a given row → keep NaN
    - If ANY column in the group has a value for a given row → replace NaN
      values in the group with 0 for that row

    This is useful when a set of related metrics should be treated as a block:
    if one metric is present, missing metrics are assumed to be 0; otherwise, if
    the whole group is missing, the entire group stays NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing flattened metric columns (e.g., "Noise_statistics.mean").
    min_group_size : int, default 2
        Minimum number of columns required to form a group. Groups with fewer
        columns are ignored.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with NaNs filled according to the rule above.
    """
    import pandas as pd

    # Select only columns that contain a dot (flattened metric columns)
    metric_cols = [c for c in df.columns if "." in c]

    # Group columns by prefix (everything before the last '.')
    groups = {}
    for col in metric_cols:
        prefix = col.rsplit(".", 1)[0]
        groups.setdefault(prefix, []).append(col)

    for prefix, cols in groups.items():
        if len(cols) < min_group_size:
            continue  # Ignore groups that are too small

        # Identify rows where at least one column in the group is not NaN
        mask_any = df[cols].notna().any(axis=1)

        # Fill NaNs with 0 only for rows where some value exists in the group
        df.loc[mask_any, cols] = df.loc[mask_any, cols].fillna(0)

    return df