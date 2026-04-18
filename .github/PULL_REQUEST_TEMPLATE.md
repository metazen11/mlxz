## Summary
<!-- what changes, why -->

## Benchmark impact
<!-- paste `mlxz bench --regression` output, or N/A with justification -->

## Correctness
<!-- which tests cover this; link to new tests -->

## Rollback
<!-- how we undo this if it regresses silently in 2 weeks -->

## Checklist
- [ ] Conventional Commit title
- [ ] Docstrings + invariants on new public APIs
- [ ] Tests added (unit / integration / correctness as appropriate)
- [ ] No new dependency, or justification present
- [ ] `mlxz doctor` still passes on dev machine
- [ ] No new `mx.eval` call site outside the engine thread
