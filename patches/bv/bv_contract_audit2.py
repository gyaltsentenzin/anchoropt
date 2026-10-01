"""Per-cell contract audit, scoped to each firing site rather than grepping globally."""
import re, pathlib
src = pathlib.Path("anchoropt/memory_evaluator.py").read_text()
lines = src.splitlines()

def window(anchor, before=10, after=90):
    i = src.index(anchor)
    ln = src[:i].count("\n")
    return "\n".join(lines[max(0, ln - before): ln + after]), ln + 1

CELLS = [
    ("post_generation_pre_exec/suppress", "_up_withheld, _up_keep = [], []",
     {"consumes": ("retry_budget",), "requires": ()}),
    ("post_generation_pre_exec/reprompt", 'remedy_enabled(templates, "enable_zero_call_reprompt")',
     {"consumes": ("instruction", "retry_budget"), "requires": ()}),
    ("post_execution/reroute (primitive hook)", 'if _prim != "additional_read_and_merge"',
     {"consumes": (), "requires": ("primitive", "retry_semantics")}),
]

ETA_RE = re.compile(r'(?:_eta|_uc\.eta|getattr\(_uc, ["\']eta["\']|eta)(?:\.get\(|\[)\s*["\'](\w+)["\']')

for name, anchor, decl in CELLS:
    if anchor not in src:
        print(f"{name:42s} SITE ABSENT -> EXECUTOR_UNAVAILABLE"); continue
    w, ln = window(anchor)
    read = sorted(set(ETA_RE.findall(w)))
    missing_consumes = [k for k in decl["consumes"] if k not in read]
    print(f"\n{name}   (line ~{ln})")
    print(f"   declares consumes : {list(decl['consumes'])}")
    print(f"   declares requires : {list(decl['requires'])}")
    print(f"   eta keys READ here: {read}")
    if missing_consumes:
        print(f"   *** DECLARED-BUT-UNREAD: {missing_consumes}  -> inert eta, arms differing only in "
              f"these are byte-identical")
    else:
        print(f"   ok: every declared-consumed key is read at the site")
