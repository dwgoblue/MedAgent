from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from medagent_system.runtime.core.models import CPB


CLAIM_TYPES = {"Observed", "Inferred", "Recommended"}
CONFIDENCE_LEVELS = {"High", "Medium", "Low"}


@dataclass
class PatientSummary:
    timeline: list[dict[str, Any]] = field(default_factory=list)
    key_vitals: list[dict[str, Any]] = field(default_factory=list)
    key_labs: list[dict[str, Any]] = field(default_factory=list)
    phenotype_keywords: list[str] = field(default_factory=list)


@dataclass
class MedGemmaReport:
    findings: list[str] = field(default_factory=list)
    candidate_assessments: list[str] = field(default_factory=list)
    supporting_observations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class GenotypeEvidence:
    source_type: str
    source_id: str
    quote_or_summary: str
    method: str = "qual"
    strength_level: str = "moderate"


@dataclass
class GenotypeHypothesis:
    hypothesis: str
    phenotype_links: list[str] = field(default_factory=list)
    evidence_strength: str = "Low"
    caveats: list[str] = field(default_factory=list)
    evidence_items: list[GenotypeEvidence] = field(default_factory=list)


@dataclass
class GenotypeReport:
    interpretations: list[str] = field(default_factory=list)
    hypotheses: list[GenotypeHypothesis] = field(default_factory=list)
    actionability_caveats: list[str] = field(default_factory=list)
    biomcp_query_count: int = 0
    biomcp_empty_count: int = 0


@dataclass
class DraftSOAP:
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    differential_ranked: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    confidence_score: float = 0.5


@dataclass
class ClaimEvidenceItem:
    source_type: str
    source_id: str
    quote_or_summary: str
    method: str
    strength_level: str


@dataclass
class ClaimLedgerEntry:
    claim_id: str
    claim_text: str
    claim_type: str
    confidence: str
    evidence_items: list[ClaimEvidenceItem] = field(default_factory=list)
    reasoning: str = ""
    uncertainty: str = ""
    depends_on_claim_ids: list[str] = field(default_factory=list)


@dataclass
class VerifierPatch:
    claim_id: str
    issue_type: str
    message: str
    suggested_fix: str


@dataclass
class VerifierReport:
    status: str = "FAIL"
    patch_list: list[VerifierPatch] = field(default_factory=list)
    check_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class CitationRecord:
    citation_id: str
    intent: str
    query: str
    source: str
    summary: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    timestamp_utc: str
    agent: str
    action: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlackboardState:
    patient_id: str
    revision_id: int = 0
    patient_summary: PatientSummary = field(default_factory=PatientSummary)
    medgemma_report: MedGemmaReport | None = None
    genotype_report: GenotypeReport | None = None
    draft_soap: DraftSOAP | None = None
    claims_ledger: list[ClaimLedgerEntry] = field(default_factory=list)
    verifier_report: VerifierReport = field(default_factory=VerifierReport)
    citations: list[CitationRecord] = field(default_factory=list)
    audit_log: list[AuditEvent] = field(default_factory=list)
    escalate_to_doctor: bool = False
    critic_report: dict[str, Any] = field(default_factory=dict)

    def add_audit(self, agent: str, action: str, **details: Any) -> None:
        self.audit_log.append(
            AuditEvent(
                timestamp_utc=datetime.now(UTC).isoformat(),
                agent=agent,
                action=action,
                details=details,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_patient_summary(cpb: CPB) -> PatientSummary:
    timeline: list[dict[str, Any]] = []
    keywords: set[str] = set()
    key_vitals: list[dict[str, Any]] = []
    key_labs: list[dict[str, Any]] = []

    for event in cpb.timeline:
        timeline.append(
            {
                "t": event.t,
                "encounter_type": event.encounter_type,
                "ehr_excerpt": event.ehr_text[:400],
                "imaging_count": len(event.imaging),
                "variant_count": len(event.genomics.variants),
            }
        )

        for vital in event.structured.get("vitals", [])[:5]:
            key_vitals.append(vital)
        for lab in event.structured.get("labs", [])[:8]:
            key_labs.append(lab)

        text = event.ehr_text.lower()
        for term in [
            "chest pain",
            "shortness of breath",
            "dyspnea",
            "fever",
            "cough",
            "abdominal pain",
            "headache",
            "weakness",
            "syncope",
        ]:
            if term in text:
                keywords.add(term)

    return PatientSummary(
        timeline=timeline,
        key_vitals=key_vitals,
        key_labs=key_labs,
        phenotype_keywords=sorted(keywords),
    )


def init_blackboard_from_cpb(cpb: CPB) -> BlackboardState:
    state = BlackboardState(
        patient_id=cpb.patient_id,
        patient_summary=build_patient_summary(cpb),
    )
    state.add_audit("orchestrator_v2", "init_blackboard", timeline_events=len(cpb.timeline))
    return state
