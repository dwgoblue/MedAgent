# MedAgent Design Upgrade Plan

Date: 2026-02-16
Inputs:
- `/home/daweilin/Downloads/aiagents/biomni.pdf` (bioRxiv preprint DOI: 10.1101/2025.05.30.656746, posted 2025-06-02)
- User-provided Superpowers workflow summary
- Current MedAgent implementation in `medagent/runtime/`

## 1) Biomni workflow digest

Biomni is split into:
- `Biomni-E1`: unified biomedical action environment
  - 150 specialized tools
  - 105 software packages
  - 59 databases
  - action discovery pipeline mines literature and curates executable actions
- `Biomni-A1`: generalist agent architecture
  - retrieval-guided tool/database selection
  - code as the universal action interface (Python/R/Bash)
  - adaptive planning loop (plan -> execute -> revise)
  - dynamic, non-template workflows

Design principles observed:
- Environment-first design: robust action space before orchestration.
- Code-centric execution, not only static function-calling.
- Retrieval + planning + execution loop with trajectory traceability.
- Human verification and benchmark-driven iteration.

## 2) Superpowers workflow digest

Superpowers emphasizes strict process control:
1. Brainstorming before coding.
2. Git worktree isolation after design approval.
3. Writing fine-grained implementation plans.
4. Subagent-driven execution or batch execution with checkpoints.
5. Mandatory TDD (RED-GREEN-REFACTOR).
6. Mandatory code review gates.
7. Structured branch finalization workflow.

Design principles observed:
- Process reliability over ad-hoc coding.
- Tight quality gates between every stage.
- Explicit handoffs and artifacts at each phase.

## 3) Gap analysis: current MedAgent vs Biomni + Superpowers

Current strengths:
- Clear modular stages (ingest, planning, evidence, safety, verifier).
- Shared schema and provenance support.
- Initial role split into SynthLab/MedGemma/Verifier agents.

Current gaps:
- Limited action space:
  - BioMCP retrieval is still stubbed.
  - No dynamic tool registry / capability retrieval.
- Planning loop is shallow:
  - No iterative replanning based on execution outcomes.
  - Disease ranking is heuristic, not model-calibrated.
- Weak process gates:
  - No mandatory brainstorming -> plan -> execution workflow in code.
  - No strict TDD enforcement gate in orchestration.
  - No branch/worktree automation.
- Verification depth:
  - Evidence gating exists, but no per-claim traceability matrix.
  - No hard blocker policy for unsupported high-risk recommendations.

## 4) Target architecture for MedAgent v2

### A. Action Environment Layer (Biomni-E1-style)
- Build `ActionRegistry` abstraction:
  - tool metadata, input schema, output schema, failure modes
  - source tags: `synthlab`, `biomcp`, `local_analytics`, `medgemma`
- Add retrieval index over capabilities:
  - select tools from task intent + patient context

### B. Planner-Executor Loop (Biomni-A1-style)
- Replace single-pass planning with loop:
  1) initial plan
  2) execute selected actions
  3) inspect outputs/errors
  4) revise plan
  5) finalize
- Persist step trajectory (inputs, tool calls, outputs, exceptions).

### C. Process Guardrail Layer (Superpowers-style)
- Add mandatory workflow state machine:
  - `brainstormed -> designed -> planned -> executing -> reviewed -> finalized`
- Hard gates:
  - cannot execute if no approved plan artifact
  - cannot finalize if tests/review fail
- Optional branch/worktree automation hooks.

### D. Clinical Reliability Layer
- Per-recommendation evidence matrix:
  - recommendation -> supporting evidence IDs -> confidence -> verification status
- Policy gates:
  - genotype-only claim => downgrade unless orthogonal evidence exists
  - high-risk recommendation without evidence => blocked

## 5) Concrete implementation plan

Phase 1: Reliability foundation (short)
- Implement real BioMCP evidence adapter.
- Add `evidence_trace` structure in `PlanOutput.provenance`.
- Add hard evidence policy checks for high-risk actions.

Phase 2: Process workflow (short)
- Add planning artifacts:
  - `agentic/artifacts/design.md`
  - `agentic/artifacts/plan.md`
- Add orchestration checks that require these artifacts before run modes.

Phase 3: Adaptive agent loop (medium)
- Add `PlanStep` objects with execution status.
- Implement iterative planner-executor-verifier loop.
- Capture full trajectory for audit/debug.

Phase 4: Quality gates (medium)
- Add tests for:
  - negation handling
  - evidence gating
  - contraindication blocking
  - provenance completeness
- Add CI gate that fails when required policy tests fail.

## 6) Immediate coding priorities (next sprint)

1. Replace local BioMCP retrieval stubs with actual retrieval calls in runtime retrieval adapters.
2. Add `EvidencePolicyAgent` to block unsafe unsupported steps.
3. Add structured `disease_ranking` and `evidence_trace` fields to schema.
4. Add workflow mode config enforcing draft plan artifact before execution.
5. Add TDD tests for policy constraints before further feature work.

## 7) Success metrics

- Evidence coverage ratio for recommendations.
- Percent of genotype-influenced suggestions with orthogonal evidence.
- Contraindication block precision/recall.
- Trajectory completeness rate (all tool calls traceable).
- Human review acceptance rate.
