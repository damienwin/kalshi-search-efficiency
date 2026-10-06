"""Feature search: the shared grammar every arm draws from.

Importing the package loads grammar.py, which fills the operator registry, so
a FeatureSpec can never be validated against an empty catalog.
"""

from ksearch.search.spec import OPS, FeatureSpec, Op, SpecError, col, node
from ksearch.search.grammar import (
    CROSS_EXCLUDED_REASON, DEFAULT_COLUMNS, DEFAULT_MAX_DEPTH, FORBIDDEN_COLUMNS, build_spec,
    catalog, catalog_description, check_columns, degenerate_reason, evaluate, flat_space, parse,
    random_spec, space_ops, space_size, spec_to_flat, validate,
)

__all__ = [
    "OPS", "FeatureSpec", "Op", "SpecError", "col", "node",
    "CROSS_EXCLUDED_REASON", "DEFAULT_COLUMNS", "DEFAULT_MAX_DEPTH", "FORBIDDEN_COLUMNS",
    "build_spec", "catalog", "catalog_description", "check_columns", "degenerate_reason",
    "evaluate", "flat_space", "parse", "random_spec", "space_ops", "space_size", "spec_to_flat",
    "validate",
]
