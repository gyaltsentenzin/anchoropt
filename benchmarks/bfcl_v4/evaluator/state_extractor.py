"""
Universal state extractor for AnchorOpt.

Turns `involved_instances` (dict[class_name -> live BFCL state object]) into
clean, JSON-safe snapshots, and diffs two snapshots into a delta.

No BFCL imports — class types are detected by class name string so this module
is usable in isolation without the full BFCL package on sys.path.
"""

import datetime

# Classes that carry no meaningful state between turns.
STATELESS_CLASSES = {"MathAPI"}

# Instance attributes that are bookkeeping, not state — always skip.
_SKIP_ATTRS = {"_api_description", "long_context"}

# The one underscore-prefixed attribute that IS real state.
_KEEP_UNDERSCORE = {"_current_dir"}


# ── Public API ────────────────────────────────────────────────────────────────

def snapshot(involved_instances: dict) -> dict:
    """
    Serialize all stateful instances to a JSON-safe dict.

    Returns: {class_name: serialized_state_dict}
    """
    result = {}
    for class_name, instance in involved_instances.items():
        if class_name in STATELESS_CLASSES:
            continue
        result[class_name] = _serialize_instance(class_name, instance)
    return result


def delta(prev: dict, curr: dict) -> dict:
    """
    Diff two snapshots returned by `snapshot()`.

    Returns a flat dict of changed paths:
      {"GorillaFileSystem.cwd": {"old": "/home", "new": "/home/docs"}, ...}

    Empty dict means nothing changed — the signal for silent/delta_if_fail modes.
    """
    changes = {}
    all_classes = set(prev) | set(curr)
    for cls in all_classes:
        p = prev.get(cls, {})
        c = curr.get(cls, {})
        _diff_dicts(p, c, prefix=cls, out=changes)
    return changes


# ── Serialization ─────────────────────────────────────────────────────────────

def _serialize_instance(class_name: str, obj) -> dict:
    """
    Serialize one instance to a plain dict.
    GorillaFileSystem gets special treatment: emit cwd + tree instead of raw attrs.
    """
    if class_name == "GorillaFileSystem":
        return _serialize_filesystem(obj)

    result = {}
    for key, value in vars(obj).items():
        if key in _SKIP_ATTRS:
            continue
        if key.startswith("_") and key not in _KEEP_UNDERSCORE:
            continue
        serialized = _to_jsonable(value)
        if serialized is not _SKIP_SENTINEL:
            result[key] = serialized
    return result


def _serialize_filesystem(obj) -> dict:
    """
    Special serializer for GorillaFileSystem.

    Emits:
      {"cwd": "/home/user/docs", "tree": <recursive dir structure>}

    cwd is computed by walking _current_dir.parent up to root.
    """
    cwd = _compute_cwd(obj._current_dir) if hasattr(obj, "_current_dir") else "/"
    tree = _to_jsonable(obj.root) if hasattr(obj, "root") else {}
    return {"cwd": cwd, "tree": tree}


def _compute_cwd(directory) -> str:
    """Walk Directory.parent chain to build absolute path string."""
    parts = []
    node = directory
    while node is not None:
        parts.append(node.name)
        node = getattr(node, "parent", None)
    parts.reverse()
    # Root Directory.name is typically "/" or the root name; join carefully.
    path = "/".join(p for p in parts if p and p != "/")
    return "/" + path if not path.startswith("/") else path


def _to_jsonable(value, _visited=None):
    """
    Recursively convert a value to a JSON-safe type.
    Returns _SKIP_SENTINEL for types that should be dropped entirely.
    _visited guards against circular references in object graphs (e.g. Directory.parent back-refs).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if _visited is None:
        _visited = set()
    oid = id(value)
    if oid in _visited:
        return "<cycle>"
    _visited.add(oid)

    cls_name = type(value).__name__

    if cls_name == "Directory":
        contents = {}
        for k, v in value.contents.items():
            serialized = _to_jsonable(v, _visited)
            if serialized is not _SKIP_SENTINEL:
                contents[k] = serialized
        return {"name": value.name, "contents": contents}

    if cls_name == "File":
        return {"name": value.name, "content": value.content}

    if isinstance(value, set):
        return sorted(str(x) for x in value)

    if isinstance(value, datetime.datetime):
        return value.isoformat()

    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            serialized = _to_jsonable(v, _visited)
            if serialized is not _SKIP_SENTINEL:
                result[k] = serialized
        return result

    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            serialized = _to_jsonable(item, _visited)
            if serialized is not _SKIP_SENTINEL:
                out.append(serialized)
        return out

    # random.Random, unpicklable objects, etc. — drop silently.
    return _SKIP_SENTINEL


# Sentinel object to signal "drop this field".
class _SkipSentinel:
    def __repr__(self):
        return "<SKIP>"

_SKIP_SENTINEL = _SkipSentinel()


# ── Delta computation ──────────────────────────────────────────────────────────

def _diff_dicts(prev, curr, prefix: str, out: dict):
    """Recursively diff two JSON-safe dicts, writing changes to `out`."""
    if not isinstance(prev, dict) or not isinstance(curr, dict):
        # Leaf comparison.
        if prev != curr:
            out[prefix] = {"old": prev, "new": curr}
        return

    all_keys = set(prev) | set(curr)
    for key in all_keys:
        path = f"{prefix}.{key}"
        if key not in prev:
            out[path] = {"added": curr[key]}
        elif key not in curr:
            out[path] = {"removed": prev[key]}
        else:
            p_val, c_val = prev[key], curr[key]
            if isinstance(p_val, dict) and isinstance(c_val, dict):
                _diff_dicts(p_val, c_val, path, out)
            elif isinstance(p_val, list) and isinstance(c_val, list):
                _diff_lists(p_val, c_val, path, out)
            elif p_val != c_val:
                out[path] = {"old": p_val, "new": c_val}


def _diff_lists(prev: list, curr: list, prefix: str, out: dict):
    """Report list changes as added/removed items (order-insensitive for readability)."""
    prev_set = set(_hashable(x) for x in prev)
    curr_set = set(_hashable(x) for x in curr)
    added = [x for x in curr if _hashable(x) not in prev_set]
    removed = [x for x in prev if _hashable(x) not in curr_set]
    if added or removed:
        change = {}
        if added:
            change["added"] = added
        if removed:
            change["removed"] = removed
        out[prefix] = change


def _hashable(value):
    """Make a value hashable for set membership checks."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, list):
        return tuple(_hashable(x) for x in value)
    return value
