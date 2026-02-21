from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class Variant:
    gene: str | None = None
    hgvs: str | None = None
    zygosity: str | None = None
    consequence: str | None = None
    quality: float | str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Variant":
        return cls(
            gene=data.get("gene"),
            hgvs=data.get("hgvs"),
            zygosity=data.get("zygosity"),
            consequence=data.get("consequence"),
            quality=data.get("quality"),
        )


@dataclass
class Genomics:
    vcf_ref: str | None = None
    variants: list[Variant] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Genomics":
        if not data:
            return cls()
        variants = [Variant.from_dict(v) for v in data.get("variants", [])]
        return cls(vcf_ref=data.get("vcf_ref"), variants=variants)


@dataclass
class ImagingRecord:
    modality: str | None = None
    body_part: str | None = None
    report_text: str | None = None
    refs: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImagingRecord":
        return cls(
            modality=data.get("modality"),
            body_part=data.get("body_part"),
            report_text=data.get("report_text"),
            refs=list(data.get("refs", [])),
        )


@dataclass
class TimelineEvent:
    t: str
    encounter_type: str
    ehr_text: str = ""
    structured: dict[str, Any] = field(default_factory=dict)
    imaging: list[ImagingRecord] = field(default_factory=list)
    genomics: Genomics = field(default_factory=Genomics)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimelineEvent":
        _ = datetime.fromisoformat(data["t"].replace("Z", "+00:00"))
        imaging = [ImagingRecord.from_dict(i) for i in data.get("imaging", [])]
        genomics = Genomics.from_dict(data.get("genomics"))
        return cls(
            t=data["t"],
            encounter_type=data["encounter_type"],
            ehr_text=data.get("ehr_text", ""),
            structured=data.get("structured", {}),
            imaging=imaging,
            genomics=genomics,
        )


@dataclass
class CPB:
    patient_id: str
    timeline: list[TimelineEvent]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CPB":
        if not data.get("patient_id"):
            raise ValueError("CPB missing patient_id")
        if not isinstance(data.get("timeline"), list) or not data["timeline"]:
            raise ValueError("CPB requires non-empty timeline")
        events = [TimelineEvent.from_dict(e) for e in data["timeline"]]
        return cls(patient_id=data["patient_id"], timeline=events)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimEvidence:
    source: str
    id: str
    url: str | None = None
    title: str | None = None


@dataclass
class ClaimObject:
    claim_text: str
    category: str
    must_verify: bool
    evidence_requirements: list[str] = field(default_factory=list)
    evidence: list[ClaimEvidence] = field(default_factory=list)
    status: str = "weak"
    resolution: str = ""


@dataclass
class DraftOutput:
    soap_draft: str
    differential: list[str]
    rationale_snippets: list[str]
    uncertainty_flags: list[str]


@dataclass
class SensitivityFinding:
    problem: str
    sensitivity_class: str
    fragile_on: list[str] = field(default_factory=list)


@dataclass
class FinalOutput:
    soap_final: str
    problem_list_ranked: list[str]
    plan_options_ranked_non_prescriptive: list[str]
    evidence_table: list[dict[str, Any]]
    sensitivity_map: list[dict[str, Any]]
    uncertainty_and_escalation_guidance: str
    provenance: dict[str, Any]
