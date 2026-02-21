# Threat Model and Safety Notes

## Primary risks

- Unsupported genotype-driven conclusions.
- Hallucinated citations or stale evidence links.
- Modality leakage where removed concepts still appear in rationales.
- Instability under minor concept perturbation.

## Mitigations

- Claim-level evidence contract with pass/weak/fail outcomes.
- Mandatory caveats for weak evidence.
- Causal perturbation checks with critical-flip detection.
- Provenance logging for inputs, tools, and model outputs.

## Output policy

- Decision-support framing only.
- No prescribing/dosing instructions.
- Include uncertainty and escalation guidance in every final artifact.
