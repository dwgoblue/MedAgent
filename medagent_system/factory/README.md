# Factory

Superpowers-style engineering plane for safe and reproducible runtime evolution.

## Mandatory workflow

1. Brainstorm/design doc.
2. Isolated branch/worktree implementation.
3. Short executable plan with file paths and checks.
4. TDD (RED -> GREEN -> REFACTOR).
5. Severity-gated review.
6. CI and merge.

## CI gates target

- Schema validation.
- Deterministic replay of sample runs.
- Verification harness tests (mock MCP).
- Causal perturbation invariants.
