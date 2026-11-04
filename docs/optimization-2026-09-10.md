# Schema enumeration construction

`FieldSpec` previously validated each declared choice against the entire choice
list. Each candidate is already a member, so this repeated scan added quadratic
work, especially for composite JSON values. Construction now checks each copied
candidate's type, bounds, and nested constraints before installing the outer
enumeration. Nested fields still check their own choices. Empty enumerations,
defensive copying, and runtime membership validation retain their behavior.

The outer candidate validation is now linear in candidate count for fixed-size
values and fixed nested constraints. Runtime membership lookup still scans the
enumeration; this change improves schema construction, not request lookup.

Local measurements against `d16a6d7`, median of three runs:

| Candidate type | Count | Before | After |
| --- | ---: | ---: | ---: |
| Integer | 1,000 | 1.81 ms | 0.30 ms |
| Integer | 3,000 | 12.94 ms | 0.86 ms |
| Object containing an array | 1,000 | 371.70 ms | 1.34 ms |
| Object containing an array | 3,000 | 3,246.39 ms | 3.52 ms |

Absolute timings depend on the machine. Reproduce from the repository root:

```bash
python -m examples.benchmark_schema_choices --reference d16a6d7
python -m pytest -q
ruff check orchestrion integrations tests examples
```

Regression tests check that later candidates cannot bypass nested enumerations,
input mutation cannot alter stored choices, empty choices reject every value,
and construction avoids quadratic candidate comparisons without timing limits.
