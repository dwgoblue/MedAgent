---
name: medagent-runtime-implementation
description: Use this skill when implementing or extending the actual MedGemma x SynthLab x BioMCP runtime in medagent, including agent contracts, CPB/CCG/claim schemas, evidence verification, and causal perturbation robustness.
---

# MedAgent Runtime Implementation

## Use this skill when

- Building the runtime agent graph in `medagent/runtime`.
- Implementing evidence contracts and BioMCP verification loops.
- Adding causal verification and sensitivity mapping.
- Extending factory workflows, tests, or CI in `medagent/factory`.

## Implementation order

1. Define or evolve schemas in `runtime/schemas`.
2. Implement agent contracts and typed tool interfaces.
3. Add verification loops (BioMCP and causal perturbation).
4. Implement synthesis and stop conditions.
5. Add tests/golden cases in `factory/tests`.
6. Add CI checks in `factory/ci`.

## Non-negotiables

- Runtime/Factory separation is mandatory.
- SynthLab remains data/eval authority.
- Outputs remain non-prescriptive decision support.
- Must-verify claims cannot pass without evidence or explicit caveat.
