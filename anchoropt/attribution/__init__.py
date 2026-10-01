

# ------------------------------------------------------------------------------------------------
# The active benchmark adapter -- SUPPLIED to the core, never located by it
# ------------------------------------------------------------------------------------------------
#
# The dependency runs one way:
#
#     benchmarks/<name>/run.py  ->  its adapter  ->  register_adapter()  ->  core consumes it
#
# NOT the other way. The core knows there is *an* adapter with a known set of predicates; it does not
# know that a benchmark called "bfcl_v4" exists, nor where its file lives. An earlier version of this
# function defaulted to `name="bfcl_v4"` and built the path `benchmarks/<name>/adapter.py`, which
# meant the core was discovering a specific benchmark -- the coupling this refactor exists to remove,
# reintroduced one level up.
#
# Deliberately not a plugin framework: one module-level slot, one register call, one accessor.

_ACTIVE_ADAPTER = None


class NoAdapterRegistered(RuntimeError):
    """Raised when core code needs benchmark vocabulary and none has been supplied.

    The message names the fix rather than the symptom, because the failure surfaces deep in a
    trajectory scan where "None has no attribute chain_of" would be useless.
    """


def register_adapter(adapter) -> None:
    """Install the active benchmark adapter. Called by a benchmark's entry point, not by the core.

    `adapter` is any object exposing the predicates the core uses -- `chain_of`, `backend_of`,
    `is_read_all`, `is_list_keys`, `target_id`. A module satisfies that, which is why
    `benchmarks/bfcl_v4/adapter.py` can be passed directly with no wrapper class.

    Duck-typed on purpose: pinning an ABC now would fix the interface from one example, and the point
    of stopping short of a canonical event model is to let the second benchmark inform that shape.
    """
    global _ACTIVE_ADAPTER
    _ACTIVE_ADAPTER = adapter


def active_adapter():
    """The registered adapter, or raise. The core's ONLY route to benchmark vocabulary."""
    if _ACTIVE_ADAPTER is None:
        raise NoAdapterRegistered(
            "no benchmark adapter registered. A benchmark entry point must call "
            "anchoropt.attribution.register_adapter(<its adapter module>) before the core reads "
            "trajectories or case ids. The core does not locate adapters by name or path."
        )
    return _ACTIVE_ADAPTER


def adapter_or_none():
    """The registered adapter, or None -- for call sites with a legitimate fallback.

    `redundant_write._state_attrs` is the one such site: it has a historical default and must stay
    importable standalone.
    """
    return _ACTIVE_ADAPTER
