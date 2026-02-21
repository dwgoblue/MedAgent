"""Shared schema for SynthLab + medgemma_cup agentic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvidenceItem:
    source: str
    snippet: str
    url: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class DiagnosticStep:
    name: str
    rationale: str
    urgency: str
    info_gain: str
    contraindications: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)


@dataclass
class EhRSummary:
    chief_complaint: str
    key_findings: List[str]
    red_flags: List[str]
    constraints: List[str]


@dataclass
class SoapNote:
    subjective: str
    objective: str
    assessment: str
    plan: str


@dataclass
class PatientContext:
    patient_id: str
    notes: str
    genotype_card: str = ""
    imaging_summary: str = ""
    fhir_bundle_path: Optional[str] = None
    timeline_path: Optional[str] = None


@dataclass
class PlanDraft:
    ehr_summary: EhRSummary
    diagnostic_steps: List[DiagnosticStep]
    soap_note: Optional[SoapNote] = None
    missing_info_questions: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PlanOutput:
    ehr_summary: EhRSummary
    diagnostic_steps: List[DiagnosticStep]
    soap_note: Optional[SoapNote] = None
    missing_info_questions: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
