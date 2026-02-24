from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any

# Allow direct script execution from repo root without package install.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medagent.runtime.agents.orchestrator.factory import build_orchestrator
from medagent.runtime.core.models import CPB, FinalOutput, TimelineEvent, ImagingRecord, Genomics, Variant


def _import_synthlab() -> Any:
    synthlab_repo = ROOT / "synthlab"
    if str(synthlab_repo) not in sys.path:
        sys.path.insert(0, str(synthlab_repo))

    import importlib

    try:
        sl = importlib.import_module("synthlab")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"SynthLab dependency missing: {exc}. "
            "Install synthlab dependencies in the active environment."
        ) from exc
    if not hasattr(sl, "load_multimodal_dataset"):
        raise RuntimeError(
            "Loaded synthlab module does not expose load_multimodal_dataset. "
            "Ensure synthlab package is installed or /synthlab repo path is available."
        )
    return sl


def _extract_outcome_from_fhir(patient: Any) -> str:
    fhir = getattr(patient, "fhir", None)
    if not fhir:
        return "unknown"

    for entry in fhir.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            if resource.get("deceasedBoolean") is True or resource.get("deceasedDateTime"):
                return "deceased"
            return "alive_or_unknown"
    return "unknown"


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _variants_from_genomics_df(df: Any, max_rows: int = 25) -> list[Variant]:
    variants: list[Variant] = []
    if df is None:
        return variants

    rows: list[dict[str, Any]]
    if hasattr(df, "to_dicts"):
        rows = df.head(max_rows).to_dicts()
    else:
        rows = []

    for row in rows:
        lower = {k.lower(): v for k, v in row.items()}
        gene = lower.get("gene") or lower.get("symbol")
        hgvs = lower.get("hgvs") or lower.get("variant") or lower.get("rsid")
        zygosity = lower.get("zygosity") or lower.get("genotype")
        consequence = lower.get("consequence") or lower.get("effect")
        quality = lower.get("qual") or lower.get("quality")
        variants.append(
            Variant(
                gene=_safe_str(gene) or None,
                hgvs=_safe_str(hgvs) or None,
                zygosity=_safe_str(zygosity) or None,
                consequence=_safe_str(consequence) or None,
                quality=quality,
            )
        )
    return variants


def _build_ehr_text(patient: Any) -> str:
    chunks: list[str] = []

    demographics = patient.get_demographics() or {}
    conds = patient.get_conditions() or []
    meds = patient.get_medications() or []

    if demographics.get("gender") or demographics.get("birth_date"):
        chunks.append(
            f"Patient demographics: gender={demographics.get('gender')}, birth_date={demographics.get('birth_date')}"
        )

    if conds:
        condition_labels = [c.get("display") for c in conds[:10] if c.get("display")]
        if condition_labels:
            chunks.append("Known conditions: " + ", ".join(condition_labels))

    if meds:
        med_labels = [m.get("display") for m in meds[:10] if m.get("display")]
        if med_labels:
            chunks.append("Medications: " + ", ".join(med_labels))

    notes_paths = getattr(patient, "notes_paths", []) or []
    for p in notes_paths[:2]:
        try:
            txt = Path(p).read_text(encoding="utf-8")[:1200]
        except Exception:
            continue
        if txt.strip():
            chunks.append("Clinical note excerpt: " + txt.replace("\n", " "))

    if not chunks:
        return "No structured FHIR/notes text available for this synthetic patient."
    return "\n".join(chunks)


def _patient_to_cpb(patient: Any) -> tuple[CPB, dict[str, Any]]:
    encounters = patient.get_encounters() if hasattr(patient, "get_encounters") else []
    encounter = encounters[0] if encounters else {}

    timestamp = encounter.get("start") or datetime.now(UTC).isoformat()
    encounter_type = encounter.get("class") or encounter.get("type") or "outpatient"

    genomics_df = getattr(patient, "genomics", None)
    variants = _variants_from_genomics_df(genomics_df)

    dicom_paths = getattr(patient, "dicom_paths", []) or []
    imaging = [
        ImagingRecord(
            modality="DICOM",
            body_part=None,
            report_text=None,
            refs=[str(p)],
        )
        for p in dicom_paths[:16]
    ]

    conditions = patient.get_conditions() if hasattr(patient, "get_conditions") else []
    meds = patient.get_medications() if hasattr(patient, "get_medications") else []

    event = TimelineEvent(
        t=timestamp,
        encounter_type=_safe_str(encounter_type),
        ehr_text=_build_ehr_text(patient),
        structured={
            "problems": conditions,
            "meds": meds,
            "vitals": [],
            "labs": [],
            "allergies": [],
            "procedures": [],
        },
        imaging=imaging,
        genomics=Genomics(
            vcf_ref=str(getattr(patient, "genomics_path", "") or ""),
            variants=variants,
        ),
    )

    cpb = CPB(patient_id=_safe_str(patient.patient_id), timeline=[event])
    meta = {
        "modalities": getattr(patient, "modalities", []),
        "n_dicom_files": len(dicom_paths),
        "n_variants_loaded": len(variants),
        "outcome": _extract_outcome_from_fhir(patient),
    }
    return cpb, meta


def _safe_node_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {" ", "_"} else " " for ch in (text or ""))
    cleaned = "_".join(cleaned.split())
    return cleaned[:80] or "node"


def _build_dashboard_kg_artifact(patient_id: str, output: FinalOutput) -> dict[str, Any]:
    problems = output.problem_list_ranked or []
    evidence = output.evidence_table or []
    plans = output.plan_options_ranked_non_prescriptive or []
    sensitivity = output.sensitivity_map or []

    nodes: list[dict[str, str]] = [{"id": "patient", "label": patient_id, "type": "patient"}]
    edges: list[dict[str, str]] = []

    for idx, p in enumerate(problems[:10], 1):
        nid = f"p{idx}"
        nodes.append({"id": nid, "label": str(p), "type": "problem"})
        edges.append({"source": "patient", "target": nid, "relation": "has_problem"})

    for idx, e in enumerate(evidence[:30], 1):
        if not isinstance(e, dict):
            continue
        nid = f"e{idx}"
        label = str(e.get("claim_text", e.get("claim", "evidence")))
        nodes.append({"id": nid, "label": label, "type": "evidence"})
        edges.append({"source": nid, "target": "patient", "relation": "supports"})

    for idx, plan in enumerate(plans[:8], 1):
        nid = f"pl{idx}"
        nodes.append({"id": nid, "label": str(plan), "type": "plan"})
        edges.append({"source": nid, "target": "patient", "relation": "targets"})

    for idx, s in enumerate(sensitivity[:12], 1):
        nid = f"s{idx}"
        label = str(s) if not isinstance(s, dict) else f"{s.get('problem', 'problem')}: {s.get('sensitivity_class', 'unknown')}"
        nodes.append({"id": nid, "label": label, "type": "sensitivity"})
        edges.append({"source": nid, "target": "patient", "relation": "robustness"})

    return {"backend": "dashboard", "nodes": nodes, "edges": edges}


def _build_synthlab_notebook_kg_artifact(patient_id: str, output: FinalOutput) -> dict[str, Any]:
    sl = _import_synthlab()
    problems = output.problem_list_ranked or []
    evidence = output.evidence_table or []
    lines: list[str] = []
    patient_node = _safe_node_name(patient_id)

    for p in problems[:10]:
        p_node = _safe_node_name(str(p))
        lines.append(f"{patient_node}[patient] ++> {p_node}[condition]")

    for e in evidence[:30]:
        if not isinstance(e, dict):
            continue
        c = _safe_node_name(str(e.get('claim_text', e.get('claim', 'evidence'))))
        target = _safe_node_name(str(problems[0])) if problems else patient_node
        lines.append(f"{c}[finding] +> {target}[condition]")

    raw_text = "\n".join(lines)
    graph = sl.parse_causal_graph(raw_text)
    return {
        "backend": "synthlab_notebook",
        "raw_text": raw_text,
        "summary": graph.summary(),
        "graph": graph.to_dict(),
    }


def build_kg_artifact(*, patient_id: str, output: FinalOutput, kg_backend: str) -> dict[str, Any]:
    if kg_backend == "synthlab_notebook":
        try:
            return _build_synthlab_notebook_kg_artifact(patient_id, output)
        except Exception as exc:
            fallback = _build_dashboard_kg_artifact(patient_id, output)
            fallback["error"] = f"synthlab_notebook_backend_failed:{type(exc).__name__}"
            return fallback
    return _build_dashboard_kg_artifact(patient_id, output)


def _evaluate_run(output: FinalOutput) -> dict[str, Any]:
    evidence_table = output.evidence_table
    must_verify: list[dict[str, Any]] = []
    for row in evidence_table:
        category = str(row.get("category", "")).strip().lower()
        must = row.get("must_verify")
        if isinstance(must, bool):
            if must:
                must_verify.append(row)
            continue
        if category in {
            "guideline",
            "variant",
            "gene-disease",
            "imaging",
            "lab",
            "inferred",
            "recommended",
        }:
            must_verify.append(row)
    if must_verify:
        pass_rate = sum(1 for e in must_verify if e.get("status") == "pass") / len(must_verify)
        weak_rate = sum(1 for e in must_verify if e.get("status") == "weak") / len(must_verify)
        fail_rate = sum(1 for e in must_verify if e.get("status") == "fail") / len(must_verify)
    else:
        pass_rate = weak_rate = fail_rate = 0.0

    return {
        "must_verify_count": len(must_verify),
        "must_verify_pass_rate": round(pass_rate, 4),
        "must_verify_weak_rate": round(weak_rate, 4),
        "must_verify_fail_rate": round(fail_rate, 4),
        "problem_count": len(output.problem_list_ranked),
        "sensitivity_count": len(output.sensitivity_map),
    }


def run_patient_case(patient: Any, *, engine: Any, kg_backend: str = "dashboard") -> dict[str, Any]:
    cpb, patient_meta = _patient_to_cpb(patient)
    final = engine.run(cpb)
    run_eval = _evaluate_run(final)
    kg = build_kg_artifact(patient_id=cpb.patient_id, output=final, kg_backend=kg_backend)
    return {
        "patient_id": cpb.patient_id,
        "patient_meta": patient_meta,
        "evaluation": run_eval,
        "result": asdict(final),
        "kg": kg,
    }


def run_from_synthlab(
    max_patients: int = 1,
    modalities: list[str] | None = None,
    output_path: Path | None = None,
    download_if_missing: bool = False,
    kg_backend: str = "dashboard",
) -> dict[str, Any]:
    sl = _import_synthlab()

    if modalities is None:
        modalities = ["fhir", "genomics", "notes", "dicom"]

    if download_if_missing:
        # SynthLab uses "notes" in dataset loading, but downloader expects "csv".
        download_components: list[str] = []
        for modality in modalities:
            component = "csv" if modality == "notes" else modality
            if component not in download_components:
                download_components.append(component)
        try:
            sl.download_coherent_dataset(
                components=download_components,
                max_patients=max_patients,
                verbose=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download coherent dataset: {exc}") from exc

    dataset = sl.load_multimodal_dataset(max_patients=max_patients, modalities=modalities)
    engine = build_orchestrator()

    runs: list[dict[str, Any]] = []
    for patient in dataset:
        runs.append(run_patient_case(patient, engine=engine, kg_backend=kg_backend))

    summary = {
        "n_runs": len(runs),
        "avg_pass_rate": round(
            sum(r["evaluation"]["must_verify_pass_rate"] for r in runs) / max(1, len(runs)),
            4,
        ),
        "avg_fail_rate": round(
            sum(r["evaluation"]["must_verify_fail_rate"] for r in runs) / max(1, len(runs)),
            4,
        ),
    }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "modalities": modalities,
        "kg_backend": kg_backend,
        "summary": summary,
        "runs": runs,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedAgent on SynthLab multimodal patients")
    parser.add_argument("--max-patients", type=int, default=1)
    parser.add_argument("--modalities", type=str, default="fhir,genomics,notes,dicom")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--kg-backend", type=str, choices=["dashboard", "synthlab_notebook"], default="dashboard")
    parser.add_argument(
        "--output",
        type=str,
        default="medagent/runtime/examples/synthlab_run_output.json",
    )
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    result = run_from_synthlab(
        max_patients=args.max_patients,
        modalities=modalities,
        output_path=Path(args.output),
        download_if_missing=args.download_if_missing,
        kg_backend=args.kg_backend,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved full output to: {args.output}")


if __name__ == "__main__":
    main()
