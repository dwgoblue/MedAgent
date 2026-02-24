from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any


def _parse_date(s: str | None) -> datetime | None:
    """Parse FHIR-style date string to datetime (UTC). Handles YYYY, YYYY-MM, YYYY-MM-DD and full ISO."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        if len(s) == 4 and s.isdigit():
            return datetime(int(s), 7, 1, tzinfo=UTC)
        if len(s) >= 7 and s[4] == "-":
            year, month = int(s[:4]), int(s[5:7])
            day = int(s[8:10]) if len(s) >= 10 and s[7] == "-" else 1
            return datetime(year, month, day, tzinfo=UTC)
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _normalize_code_system(system: str | None) -> str:
    """Map coding system URI to short display name (e.g. ICD10, SNOMED)."""
    if not system:
        return "unknown"
    s = (system or "").strip().lower()
    if "icd-10" in s or "icd10" in s:
        return "ICD10"
    if "snomed" in s or "snomed.info" in s:
        return "SNOMED"
    if "loinc" in s:
        return "LOINC"
    if "rxnorm" in s:
        return "RxNorm"
    if "cpt" in s or "ama" in s:
        return "CPT"
    if "hcpcs" in s:
        return "HCPCS"
    if "ndc" in s:
        return "NDC"
    if "ucum" in s:
        return "UCUM"
    # Fallback: use last path segment or up to 20 chars
    m = re.search(r"([a-z0-9_-]+)$", s)
    return (m.group(1).upper() if m else s[:20]) or "other"


def _ids_from_items(items: list[dict]) -> tuple[int, int, dict[str, dict[str, int]]]:
    """From list of dicts with code/system, return (total, unique_count, ids_by_system)."""
    total = len(items)
    seen: set[tuple[str, str]] = set()
    by_system: dict[str, set[str]] = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        code = str(it.get("code") or "").strip()
        system = _normalize_code_system(it.get("system"))
        if code or system != "unknown":
            key = (code or "(no code)", system)
            seen.add(key)
            by_system.setdefault(system, set()).add(code or "(no code)")
    ids_by_system = {
        sys_name: {"total": len(codes), "unique": len(codes)}
        for sys_name, codes in sorted(by_system.items())
    }
    return total, len(seen), ids_by_system


def _compute_overview_stats(patient: Any) -> dict[str, Any]:
    """
    Compute overview stats from a SynthLab patient for dashboard.
    Returns dict with n_visits, max_age, age_span, mean_visit_frequency_per_year,
    conditions_total, conditions_unique, medications_total, medications_unique,
    procedures_total, procedures_unique, ids_by_system (per-system total/unique).
    Only includes keys for which data is available.
    """
    out: dict[str, Any] = {}
    encounters = patient.get_encounters() if hasattr(patient, "get_encounters") else []
    n_visits = len(encounters) if encounters else 0
    if n_visits > 0:
        out["n_visits"] = n_visits

    birth_dt = None
    if hasattr(patient, "get_demographics"):
        demo = patient.get_demographics()
        if isinstance(demo, dict) and demo.get("birth_date"):
            birth_dt = _parse_date(demo["birth_date"])

    visit_dates: list[datetime] = []
    for enc in encounters or []:
        if isinstance(enc, dict):
            start = enc.get("start") or enc.get("end")
            dt = _parse_date(start)
            if dt:
                visit_dates.append(dt)
    if birth_dt and visit_dates:
        ages = [(d - birth_dt).days / 365.25 for d in visit_dates]
        max_age = max(ages)
        min_age = min(ages)
        out["max_age_years"] = round(max_age, 2)
        out["age_span_years"] = round(max_age - min_age, 2)
        span_years = (max(visit_dates) - min(visit_dates)).days / 365.25 if len(visit_dates) > 1 else 1.0
        if span_years > 0:
            out["mean_visit_frequency_per_year"] = round(n_visits / span_years, 2)

    conditions = patient.get_conditions() if hasattr(patient, "get_conditions") else []
    meds = patient.get_medications() if hasattr(patient, "get_medications") else []
    procs = patient.get_procedures() if hasattr(patient, "get_procedures") else []

    c_tot, c_uni, c_by_sys = _ids_from_items(conditions)
    if c_tot > 0:
        out["conditions_total"] = c_tot
        out["conditions_unique"] = c_uni
    m_tot, m_uni, m_by_sys = _ids_from_items(meds)
    if m_tot > 0:
        out["medications_total"] = m_tot
        out["medications_unique"] = m_uni
    p_tot, p_uni, p_by_sys = _ids_from_items(procs)
    if p_tot > 0:
        out["procedures_total"] = p_tot
        out["procedures_unique"] = p_uni

    all_systems: dict[str, dict[str, int]] = {}
    for d in (c_by_sys, m_by_sys, p_by_sys):
        for sys_name, counts in d.items():
            if sys_name not in all_systems:
                all_systems[sys_name] = {"total": 0, "unique": 0}
            all_systems[sys_name]["total"] += counts["total"]
            all_systems[sys_name]["unique"] += counts["unique"]
    if all_systems:
        out["ids_by_system"] = dict(sorted(all_systems.items()))
    return out


def _aggregate_overview_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate overview_stats from multiple runs for 'All patients' summary."""
    if not runs:
        return {}
    stats_list = [r.get("overview_stats") for r in runs if isinstance(r.get("overview_stats"), dict)]
    if not stats_list:
        return {}
    agg: dict[str, Any] = {}
    if any("n_visits" in s for s in stats_list):
        agg["n_visits"] = sum(s.get("n_visits", 0) for s in stats_list)
    if any("max_age_years" in s for s in stats_list):
        agg["max_age_years"] = max(s.get("max_age_years", 0) for s in stats_list)
    if any("age_span_years" in s for s in stats_list):
        agg["age_span_years"] = max(s.get("age_span_years", 0) for s in stats_list)
    if any("mean_visit_frequency_per_year" in s for s in stats_list):
        vals = [s["mean_visit_frequency_per_year"] for s in stats_list if "mean_visit_frequency_per_year" in s]
        agg["mean_visit_frequency_per_year"] = round(sum(vals) / len(vals), 2) if vals else None
    if any("conditions_total" in s for s in stats_list):
        agg["conditions_total"] = sum(s.get("conditions_total", 0) for s in stats_list)
        agg["conditions_unique"] = sum(s.get("conditions_unique", 0) for s in stats_list)
    if any("medications_total" in s for s in stats_list):
        agg["medications_total"] = sum(s.get("medications_total", 0) for s in stats_list)
        agg["medications_unique"] = sum(s.get("medications_unique", 0) for s in stats_list)
    if any("procedures_total" in s for s in stats_list):
        agg["procedures_total"] = sum(s.get("procedures_total", 0) for s in stats_list)
        agg["procedures_unique"] = sum(s.get("procedures_unique", 0) for s in stats_list)
    all_ids: dict[str, dict[str, int]] = {}
    for s in stats_list:
        for sys_name, counts in (s.get("ids_by_system") or {}).items():
            all_ids.setdefault(sys_name, {"total": 0, "unique": 0})
            all_ids[sys_name]["total"] += counts.get("total", 0)
            all_ids[sys_name]["unique"] += counts.get("unique", 0)
    if all_ids:
        agg["ids_by_system"] = dict(sorted(all_ids.items()))
    return agg


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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


def _check_synthlab_soap_deps() -> str | None:
    """Verify transformers and torch are importable in this process. Returns None if OK, else error message."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        return f"torch not available in this process: {e}. Install with: pip install torch"
    try:
        from transformers import AutoProcessor  # noqa: F401
    except ImportError as e:
        err = str(e)
        if "Could not import module" in err or "requirements defined correctly" in err:
            cause_str = ""
            c = e.__cause__
            while c:
                cause_str += " " + str(c)
                c = getattr(c, "__cause__", None)
            if "torchvision::nms" in cause_str or "operator torchvision" in cause_str:
                return (
                    "AutoProcessor failed: torch and torchvision are out of sync (e.g. operator torchvision::nms does not exist). "
                    "Reinstall a matching pair in this env: pip install --upgrade torch torchvision (or install both from the same index, e.g. PyTorch CUDA page). "
                    "Script Python: " + sys.executable + "."
                )
            return (
                "transformers is installed but AutoProcessor could not be loaded in this process. "
                "This script is using: " + sys.executable + ". "
                "If pip shows transformers[vision] already satisfied, you are likely using a different Python when running the script. "
                "Run the script with the same interpreter (e.g. conda activate medagent then run), or verify with: "
                + sys.executable + " -c \"from transformers import AutoProcessor\" to see the full error. "
                "Otherwise try: pip install 'transformers[vision]' pillow (and torch) in the env of " + sys.executable + "."
            )
        return (
            f"transformers not available in this process: {e}. "
            "Install with: pip install 'transformers>=4.50.0'. "
            "Run this script with the same Python that has these packages (e.g. conda activate medagent)."
        )
    return None


def _synthlab_soap_text(patient: Any) -> tuple[str | None, str | None, dict[str, Any], str | None]:
    """
    Generate a full, structured SOAP note using SynthLab's SOAPNoteGenerator (same as the
    original SynthLab agentic pipeline). Returns (soap_text, causal_graph, prompts_used, None) on success,
    (None, None, {}, error_message) on failure. Causal graph is omitted from soap_text and
    should be shown in the dashboard Knowledge Graph section. prompts_used is the note's prompts dict for the dashboard.
    """
    dep_err = _check_synthlab_soap_deps()
    if dep_err:
        return None, None, {}, dep_err
    _import_synthlab()
    try:
        from synthlab.soap import generate_soap_note
    except ImportError as e:
        return None, None, {}, f"cannot import synthlab.soap: {e}"
    try:
        # separate_causal_graph=True so causal graph is generated in a separate step, not in the Patient Story
        note = generate_soap_note(
            patient,
            include_imaging=os.getenv("MEDAGENT_SYNTHLAB_SOAP_IMAGING", "1") == "1",
            use_biomcp=os.getenv("MEDAGENT_SYNTHLAB_SOAP_BIOMCP", "0") == "1",
            verbose=False,
            separate_causal_graph=True,
            only_abnormal_labs=True,
        )
    except Exception as e:
        err_str = str(e)
        if "or_mask_function" in err_str or "and_mask_function" in err_str or "require torch>=2.6" in err_str:
            err_str += " → Upgrade PyTorch: pip install 'torch>=2.6'"
        if "Missing required dependencies" in err_str and "Details:" not in err_str:
            err_str += " (pre-check passed in this process; see SynthLab logs or install deps in the same env and re-run)"
        return None, None, {}, f"generate_soap_note failed: {err_str}"
    if note is None:
        return None, None, {}, "generate_soap_note returned None"
    lines: list[str] = []
    if getattr(note, "summary", "").strip():
        lines.append("## Summary\n" + (note.summary or "").strip() + "\n")
    if getattr(note, "patient_story", "").strip():
        lines.append("## Patient Story\n" + (note.patient_story or "").strip() + "\n")
    lines.append("## Subjective\n" + (getattr(note, "subjective", "") or "(No data)").strip() + "\n")
    lines.append("## Objective\n" + (getattr(note, "objective", "") or "(No data)").strip() + "\n")
    lines.append("## Assessment\n" + (getattr(note, "assessment", "") or "(No data)").strip() + "\n")
    lines.append("## Plan\n" + (getattr(note, "plan", "") or "(No data)").strip() + "\n")
    # Causal graph is not included in SOAP note text; it is shown in the dashboard Knowledge Graph section.
    if getattr(note, "future_considerations", "").strip():
        lines.append("## Future Considerations\n" + (note.future_considerations or "").strip() + "\n")
    if getattr(note, "genetic_summary", "").strip():
        lines.append("## Genetic Summary\n" + (note.genetic_summary or "").strip() + "\n")
    text = "\n".join(lines) if lines else None
    causal_graph = (getattr(note, "causal_graph", "") or "").strip() or None
    prompts_used = getattr(note, "prompts_used", None) or {}
    if not isinstance(prompts_used, dict):
        prompts_used = {}
    return (text, causal_graph, prompts_used, None) if text else (None, None, {}, "SOAP note had no content")


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
    procedures = patient.get_procedures() if hasattr(patient, "get_procedures") else []

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
            "procedures": procedures,
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


def run_patient_case(
    patient: Any,
    *,
    engine: Any,
    kg_backend: str = "dashboard",
    use_synthlab_soap: bool | None = None,
) -> dict[str, Any]:
    if use_synthlab_soap is None:
        use_synthlab_soap = os.getenv("MEDAGENT_USE_SYNTHLAB_SOAP", "0") == "1"
    cpb, patient_meta = _patient_to_cpb(patient)
    final = engine.run(cpb)
    result = asdict(final)
    causal_graph: str | None = None
    if use_synthlab_soap:
        soap_str, causal_graph, synthlab_prompts, soap_err = _synthlab_soap_text(patient)
        if soap_str:
            result["soap_final"] = soap_str
            prov = result.get("provenance")
            if isinstance(prov, dict):
                prov["soap_source"] = "synthlab_generator"
            if synthlab_prompts and isinstance(prov, dict):
                bb = prov.get("blackboard")
                if isinstance(bb, dict):
                    pu = bb.get("prompts_used")
                    if not isinstance(pu, dict):
                        pu = {}
                    for k, v in synthlab_prompts.items():
                        # Ensure every prompt has both system and user keys for dashboard
                        if isinstance(v, dict):
                            pu["synthlab_" + str(k)] = {
                                "system": v.get("system", "") or "",
                                "user": v.get("user", v.get("prompt", "")) or "",
                            }
                        else:
                            pu["synthlab_" + str(k)] = {"system": "", "user": str(v)}
                    bb["prompts_used"] = pu
                    prov["blackboard"] = bb
                    result["provenance"] = prov
        else:
            import sys
            reason = f" ({soap_err})" if soap_err else ""
            print(
                f"[MedAgent] --use-synthlab-soap was set but SynthLab SOAP returned nothing{reason}; using pipeline SOAP.",
                file=sys.stderr,
            )
    run_eval = _evaluate_run(final)
    kg = build_kg_artifact(patient_id=cpb.patient_id, output=final, kg_backend=kg_backend)
    # Persist SynthLab knowledge graph in JSON output: raw causal graph text and optional structured form
    if causal_graph:
        kg["causal_graph"] = causal_graph
        try:
            sl = _import_synthlab()
            parsed = sl.parse_causal_graph(causal_graph)
            kg["synthlab_causal_graph"] = parsed.to_dict()
        except Exception:
            pass  # keep raw text; structured form is optional
    overview_stats = _compute_overview_stats(patient)
    return {
        "patient_id": cpb.patient_id,
        "patient_meta": patient_meta,
        "overview_stats": overview_stats if overview_stats else None,
        "evaluation": run_eval,
        "result": result,
        "kg": kg,
    }


def run_from_synthlab(
    max_patients: int = 1,
    modalities: list[str] | None = None,
    output_path: Path | None = None,
    download_if_missing: bool = False,
    kg_backend: str = "dashboard",
    use_synthlab_soap: bool | None = None,
) -> dict[str, Any]:
    if use_synthlab_soap is None:
        use_synthlab_soap = os.getenv("MEDAGENT_USE_SYNTHLAB_SOAP", "0") == "1"
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
    patients_iter = tqdm(dataset, desc="Patients", unit="patient") if tqdm else dataset
    for patient in patients_iter:
        runs.append(
            run_patient_case(
                patient,
                engine=engine,
                kg_backend=kg_backend,
                use_synthlab_soap=use_synthlab_soap,
            )
        )

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

    runs_with_stats = [r for r in runs if isinstance(r.get("overview_stats"), dict) and r["overview_stats"]]
    overview_stats_aggregated = _aggregate_overview_stats(runs_with_stats) if runs_with_stats else None

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "modalities": modalities,
        "kg_backend": kg_backend,
        "summary": summary,
        "runs": runs,
    }
    if overview_stats_aggregated:
        payload["overview_stats_aggregated"] = overview_stats_aggregated

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Full payload is written: each run includes result, evaluation, and kg (SynthLab KG in kg["causal_graph"] and optionally kg["synthlab_causal_graph"])
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedAgent on SynthLab multimodal patients")
    parser.add_argument("--max-patients", type=int, default=1)
    parser.add_argument("--modalities", type=str, default="fhir,genomics,notes,dicom")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--kg-backend", type=str, choices=["dashboard", "synthlab_notebook"], default="dashboard")
    def _parse_use_synthlab_soap(s: str | None) -> bool:
        if s is None or s == "":
            return True
        return str(s).strip().lower() not in ("0", "false", "no")

    parser.add_argument(
        "--use-synthlab-soap",
        nargs="?",
        const=True,
        default=False,
        type=lambda x: _parse_use_synthlab_soap(x) if x is not None else True,
        metavar="0|1",
        help="Use SynthLab SOAPNoteGenerator for full structured SOAP (same as original SynthLab pipeline). Pass --use-synthlab-soap or --use-synthlab-soap 1.",
    )
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
        use_synthlab_soap=args.use_synthlab_soap,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved full output to: {args.output}")


if __name__ == "__main__":
    main()
