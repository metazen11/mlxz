# Contributing to mlxz

Thank you for considering contributing to mlxz. This document covers the conventions and expectations for all contributions.

## Commit conventions

**Conventional Commits** are required and enforced by a `commitlint` pre-push hook.

```
feat(engine): add chunked prefill support
fix(api): handle empty stop sequences in chat completions
perf(cache): reduce prefix-cache lookup from O(n) to O(1)
docs(whitepaper): update KV-cache memory model diagram
```

## Pull request guidelines

**One concern per PR.** A feature and an unrelated fix are two PRs, not one.

**Every PR description must answer three questions:**

1. **What benchmark moved?** Paste `mlxz bench --regression` output, or explain why N/A.
2. **What test proves correctness?** Link to the new or existing tests that cover the change.
3. **What is the rollback plan?** How do we undo this if it regresses silently in two weeks?

## Performance PR gate

If a PR changes the engine, scheduler, cache policy, or benchmark harness, it is a performance PR.

Performance PRs must include:

- A baseline run on `Llama-3.1-8B-Instruct-4bit`.
- The same change measured on at least two additional models, preferably one smaller and one larger.
- A short note explaining whether the change is intended to improve `TTFT`, decode throughput, concurrency, or memory fit.
- A rejected-ideas note if the optimization was attempted and discarded before landing.

If the optimization only helps one model size, say that explicitly. Do not generalize it into a broad engine claim.

This gate is enforced by the project contract in [CONTRACT.md](CONTRACT.md).

## Dependencies

**No new dependency without justification** in the PR body. Address:

- Package size and install footprint
- Maintainer health and release cadence
- License compatibility (Apache-2.0)
- Why stdlib or an existing dependency cannot do it

## Public API

**Every new public API ships with:**

- A docstring that includes an invariant (what the caller can always rely on).
- At least one `>>> doctest` example.

## mx.eval discipline

**Never call `mx.eval` outside the engine thread.** This is enforced by a runtime assertion. Adding a new `mx.eval` call site triggers mandatory review from `@CODEOWNERS`.

All cross-thread communication uses `janus.Queue`, never `asyncio.Queue`.

## Development setup

```bash
uv sync --all-extras
uv run pre-commit install
uv run mlxz doctor
uv run pytest tests/unit
```

## Code style

- **Formatter:** `ruff format` (line length 100)
- **Linter:** `ruff check` with the rule set defined in `pyproject.toml`
- **Type checker:** `pyright` in strict mode on `src/`

All three run in CI and as pre-commit hooks.

## Testing

- Unit tests go in `tests/unit/`.
- Integration tests go in `tests/integration/`.
- Correctness tests (PPL, logit drift) go in `tests/correctness/`.
- Use `pytest.mark.metal` for tests that require Apple Metal GPU.
- Use `pytest.mark.slow` for tests over 30 seconds.
- All tests must pass with `--timeout=30` (correctness tests get `--timeout=3600`).
