"""Reusable contract tests a benchmark adapter runs against ITSELF.

    from anchoropt.testing import AdapterContract, check_adapter_contract

A TauBench or AppWorld porter imports this, points it at their adapter, and gets the same checks the
toy host and the BFCL adapter are held to. Nothing here is specific to any benchmark.
"""

from anchoropt.testing.adapter_contract import (
    AdapterContract, ContractReport, ContractViolation, MANDATORY_HOOKS, OPTIONAL_HOOKS,
    check_adapter_contract,
)

__all__ = ["AdapterContract", "ContractReport", "ContractViolation", "MANDATORY_HOOKS",
           "OPTIONAL_HOOKS", "check_adapter_contract"]
