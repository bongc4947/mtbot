import yaml
import os
import re


def _expand_vars(obj):
    """
    Recursively expand %VAR% and $VAR env-var placeholders in config strings.
    Converts forward slashes so Windows paths work cross-platform.
    """
    if isinstance(obj, str):
        # Windows %VAR% style
        expanded = re.sub(r"%([^%]+)%", lambda m: os.environ.get(m.group(1), m.group(0)), obj)
        # Unix $VAR style
        expanded = os.path.expandvars(expanded)
        return expanded.replace("\\", "/")
    if isinstance(obj, dict):
        return {k: _expand_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_vars(i) for i in obj]
    return obj


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_vars(raw)


def load_symbols(path: str = "config/symbols.yaml") -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    symbols = []
    for group in data.get("symbols", {}).values():
        symbols.extend(group)
    return list(dict.fromkeys(symbols))  # deduplicate, preserve order


def get_nested(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node
