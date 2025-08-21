from loguru import logger
import sys

def to_dict(var):
    if not isinstance(var, dict):
        if isinstance(var, list):
            var = {step: None for step in var}
        elif isinstance(var, str):
            var = var.split()
            var = {step: None for step in var}
        else:
            logger.error(f"Error: variable must be either str list or dict!")
            sys.exit(1)
    if isinstance(var, dict):
        return var


import re

def resolve(obj, context, max_passes=10):
    """
    Recursively resolve all placeholders using .format(**context), including nested ones.
    """
    def has_placeholders(s):
        return isinstance(s, str) and re.search(r"{\w+}", s)

    def resolve_once(o):
        if isinstance(o, dict):
            return {k: resolve_once(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [resolve_once(i) for i in o]
        elif isinstance(o, str):
            try:
                return o.format(**context)
            except KeyError:
                return o  # leave unresolved
        return o

    # Apply resolve_once repeatedly, max N times, or until no changes
    for _ in range(max_passes):
        prev = obj
        obj = resolve_once(obj)

        # Update context from newly resolved keys
        if isinstance(obj, dict):
            context.update(flatten_dict(obj))

        if obj == prev or not contains_placeholders(obj):
            break

    return obj


def contains_placeholders(obj):
    if isinstance(obj, dict):
        return any(contains_placeholders(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(contains_placeholders(i) for i in obj)
    elif isinstance(obj, str):
        return re.search(r"{\w+}", obj)
    return False


def flatten_dict(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary for .format(**...) use. Keeps only top-level keys in practice here.
    """
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten_dict(v, parent_key + k + sep, sep=sep))
        else:
            items[k] = v
    return items
