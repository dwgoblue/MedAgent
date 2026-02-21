from __future__ import annotations

import re

from medagent_system.runtime.core.models_v2 import (
    BlackboardState,
    ClaimEvidenceItem,
    ClaimLedgerEntry,
)


def _claim_id(idx: int) -> str:
    return f"CLM-{idx:04d}"


def _split_plan(plan_text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[;\n]+", plan_text) if p.strip()]
    return parts[:8]


def build_claims_ledger_v2(state: BlackboardState) -> list[ClaimLedgerEntry]:
    if state.draft_soap is None:
        state.claims_ledger = []
        return []

    draft = state.draft_soap
    entries: list[ClaimLedgerEntry] = []
    claim_texts_in_draft: set[str] = set()

    for item in draft.differential_ranked:
        claim_text = f"Differential includes: {item}"
        claim_texts_in_draft.add(claim_text)
        evidence = []
        if state.medgemma_report and state.medgemma_report.supporting_observations:
            evidence.append(
                ClaimEvidenceItem(
                    source_type="ehr_or_imaging",
                    source_id="medgemma_report",
                    quote_or_summary=state.medgemma_report.supporting_observations[0],
                    method="qual",
                    strength_level="moderate",
                )
            )
        entries.append(
            ClaimLedgerEntry(
                claim_id=_claim_id(len(entries) + 1),
                claim_text=claim_text,
                claim_type="Inferred",
                confidence="Medium",
                evidence_items=evidence,
                reasoning="Differential item derived from integrated EHR+image narrative",
                uncertainty="Requires verifier consistency check",
                depends_on_claim_ids=[],
            )
        )

    observed_claim = f"Objective summary: {draft.objective}"
    claim_texts_in_draft.add(observed_claim)
    entries.append(
        ClaimLedgerEntry(
            claim_id=_claim_id(len(entries) + 1),
            claim_text=observed_claim,
            claim_type="Observed",
            confidence="Medium",
            evidence_items=[
                ClaimEvidenceItem(
                    source_type="patient_summary",
                    source_id="patient_summary",
                    quote_or_summary="Derived from timeline/labs/vitals summary",
                    method="qual",
                    strength_level="moderate",
                )
            ],
            reasoning="Objective section maps directly to extracted observations",
            uncertainty="May omit missing negatives",
            depends_on_claim_ids=[],
        )
    )

    for plan_item in _split_plan(draft.plan):
        claim_text = f"Recommended action: {plan_item}"
        claim_texts_in_draft.add(claim_text)
        evidence_items = []
        for citation in state.citations[:2]:
            evidence_items.append(
                ClaimEvidenceItem(
                    source_type="literature",
                    source_id=citation.citation_id,
                    quote_or_summary=citation.summary,
                    method="qual",
                    strength_level="moderate",
                )
            )
        entries.append(
            ClaimLedgerEntry(
                claim_id=_claim_id(len(entries) + 1),
                claim_text=claim_text,
                claim_type="Recommended",
                confidence="Low" if not evidence_items else "Medium",
                evidence_items=evidence_items,
                reasoning="Plan recommendation from supervisor synthesis",
                uncertainty="Recommendation remains provisional without strong supporting evidence",
                depends_on_claim_ids=["CLM-0001"] if entries else [],
            )
        )

    # No-new-claims guard: all claims must be generated from draft sections only.
    for entry in entries:
        if not (
            entry.claim_text.startswith("Differential includes:")
            or entry.claim_text.startswith("Objective summary:")
            or entry.claim_text.startswith("Recommended action:")
        ):
            raise ValueError("Claims ledger introduced non-draft claim pattern")

    state.claims_ledger = entries
    state.add_audit("a4_evidence_assembler", "write_claims_ledger", claims=len(entries))
    return entries
