# MLXZ Thesis

mlxz exists to answer a narrow question: which serving ideas from vLLM can be made to work well on MLX on Apple Silicon, and which of them actually move measured performance?

The project is engine-first. The API, observability, and benchmark harness matter because they let us prove whether an optimization is real. They are not the product.

## What We Are Trying To Port

- Continuous batching for concurrent requests
- Chunked prefill so long prompts do not monopolize the engine
- Prefix caching so repeated system prompts avoid redundant prefill
- Speculative decoding when the draft/target pair is compatible
- Paged or block-managed KV state for longer contexts
- Shape-bucketed compilation where MLX benefits from stable execution shapes
- Token-delivery batching so transport overhead does not erase compute wins

## What Success Looks Like

- Better TTFT on repeated-prefix and agent-style workloads
- Better aggregate throughput under concurrency
- No correctness regressions in sampling, stop handling, or determinism contracts
- Reproducible benchmark results across multiple models, not just one headline model
- Clear evidence when an idea does not help on MLX, so we stop claiming it

## What We Should Not Drift Into

- Feature shipping without a benchmark
- Marketing a runtime feature that is not actually wired end to end
- Treating infra work as the goal instead of the measurement system
- Keeping dead experiments around without documenting why they failed

## Operational Rule

Every meaningful engine change should answer one of these:

1. Does it improve TTFT?
2. Does it improve decode throughput?
3. Does it improve concurrency or memory fit?
4. Does it improve correctness or benchmark credibility?

If the answer is no, it is probably a doc change, not an engine change.
