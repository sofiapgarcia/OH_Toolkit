"""
OH Parser Utilities.

Helper functions for dictionary manipulation and DataFrame operations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def safe_get(data: dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely navigate nested dictionary using a list of keys.
    
    :param data: Nested dictionary to navigate.
    :param keys: List of keys to traverse.
    :param default: Value to return if path doesn't exist.
    :returns: Value at path or default.
    
    Example:
        >>> d = {"a": {"b": {"c": 1}}}
        >>> safe_get(d, ["a", "b", "c"])
        1
        >>> safe_get(d, ["a", "x"], default=0)
        0
    """
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def flatten_dict(
    data: dict,
    parent_key: str = "",
    sep: str = ".",
    max_depth: Optional[int] = None,
    _current_depth: int = 0,
) -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dict with dot-notation keys.
    
    :param data: Nested dictionary to flatten.
    :param parent_key: Prefix for keys (used in recursion).
    :param sep: Separator between key levels.
    :param max_depth: Maximum depth to flatten (None = unlimited).
    :param _current_depth: Internal counter for recursion depth.
    :returns: Flattened dictionary.
    
    Example:
        >>> d = {"a": {"b": 1, "c": {"d": 2}}}
        >>> flatten_dict(d)
        {"a.b": 1, "a.c.d": 2}
    """
    items: Dict[str, Any] = {}
    
    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict) and (max_depth is None or _current_depth < max_depth):
            items.update(flatten_dict(
                value,
                parent_key=new_key,
                sep=sep,
                max_depth=max_depth,
                _current_depth=_current_depth + 1,
            ))
        else:
            items[new_key] = value
    
    return items


def unflatten_dict(data: Dict[str, Any], sep: str = ".") -> dict:
    """
    Unflatten a dot-notation dictionary back to nested structure.
    
    :param data: Flat dictionary with dot-notation keys.
    :param sep: Separator used in keys.
    :returns: Nested dictionary.
    
    Example:
        >>> d = {"a.b": 1, "a.c.d": 2}
        >>> unflatten_dict(d)
        {"a": {"b": 1, "c": {"d": 2}}}
    """
    result: dict = {}
    
    for key, value in data.items():
        parts = key.split(sep)
        current = result
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return result


def get_nested_keys(
    data: dict,
    max_depth: Optional[int] = None,
    _current_path: str = "",
    _current_depth: int = 0,
) -> List[str]:
    """
    Get all leaf key paths from a nested dictionary.
    
    :param data: Nested dictionary.
    :param max_depth: Maximum depth to traverse.
    :param _current_path: Internal path accumulator.
    :param _current_depth: Internal depth counter.
    :returns: List of dot-notation paths to all leaf values.
    """
    paths: List[str] = []
    
    for key, value in data.items():
        new_path = f"{_current_path}.{key}" if _current_path else key
        
        if isinstance(value, dict) and (max_depth is None or _current_depth < max_depth):
            paths.extend(get_nested_keys(
                value,
                max_depth=max_depth,
                _current_path=new_path,
                _current_depth=_current_depth + 1,
            ))
        else:
            paths.append(new_path)
    
    return paths


def print_tree(
    data: dict,
    indent: int = 0,
    max_depth: Optional[int] = 4,
    _current_depth: int = 0,
    show_values: bool = False,
) -> None:
    """
    Pretty-print a nested dictionary as a tree structure.
    
    :param data: Nested dictionary to print.
    :param indent: Current indentation level.
    :param max_depth: Maximum depth to display.
    :param _current_depth: Internal depth counter.
    :param show_values: Whether to show leaf values.
    """
    if max_depth is not None and _current_depth >= max_depth:
        print("  " * indent + "...")
        return
    
    for key, value in data.items():
        if isinstance(value, dict):
            print("  " * indent + f"├── {key}/")
            print_tree(
                value,
                indent=indent + 1,
                max_depth=max_depth,
                _current_depth=_current_depth + 1,
                show_values=show_values,
            )
        else:
            if show_values:
                val_repr = repr(value) if not isinstance(value, (int, float)) else value
                print("  " * indent + f"├── {key}: {val_repr}")
            else:
                type_name = type(value).__name__
                print("  " * indent + f"├── {key} ({type_name})")


def is_date_key(key: str) -> bool:
    """
    Check if a key looks like a date (YYYY-MM-DD or DD-MM-YYYY format).
    
    :param key: Key string to check.
    :returns: True if key matches date pattern.
    """
    if len(key) != 10:
        return False
    parts = key.split("-")
    if len(parts) != 3:
        return False
    try:
        p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Check YYYY-MM-DD format
        if 1900 <= p0 <= 2100 and 1 <= p1 <= 12 and 1 <= p2 <= 31:
            return True
        
        # Check DD-MM-YYYY format
        if 1 <= p0 <= 31 and 1 <= p1 <= 12 and 1900 <= p2 <= 2100:
            return True
        
        return False
    except ValueError:
        return False


def is_time_key(key: str) -> bool:
    """
    Check if a key looks like a time (HH-MM-SS format).
    
    :param key: Key string to check.
    :returns: True if key matches time pattern.
    """
    if len(key) != 8:
        return False
    parts = key.split("-")
    if len(parts) != 3:
        return False
    try:
        hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
        return 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59
    except ValueError:
        return False
