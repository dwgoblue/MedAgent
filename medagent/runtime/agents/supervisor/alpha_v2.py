from __future__ import annotations

from dataclasses import asdict
import json
import os
from typing import Any

from medagent.runtime.core.models_v2 import BlackboardState, DraftSOAP


def _safe_parse(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def _build_draft_with_openai(state: BlackboardState, verifier_feedback: list[str]) -> DraftSOAP:
    from openai import OpenAI

    med = state.medgemma_report
    geno = state.genotype_report
    model = os.getenv("MEDAGENT_OPENAI_MODEL", "gpt-5.2").strip() or "gpt-5.2"
    payload = {
        "medgemma_findings": med.findings if med else [],
        "medgemma_candidate_assessments": med.candidate_assessments if med else [],
        "medgemma_observations": med.supporting_observations if med else [],
        "genotype_interpretations": geno.interpretations if geno else [],
        "genotype_caveats": geno.actionability_caveats if geno else [],
        "phenotype_keywords": state.patient_summary.phenotype_keywords,
        "verifier_feedback": verifier_feedback,
    }

    system = (
        "You are a medical supervisor agent. Return strict JSON only with keys: "
        "subjective, objective, assessment, plan, differential_ranked, open_questions, confidence_score. "
        "confidence_score must be a number in [0,1]."
    )
    user = (
        "Produce a concise non-prescriptive SOAP summary grounded only in the given inputs. "
        "No dosing/prescriptions.\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
    )
    client = OpenAI(max_retries=0)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    parsed = _safe_parse(getattr(resp, "output_text", "") or "")
    if not parsed:
        raise RuntimeError("OpenAI supervisor returned non-JSON output")

    differential = [str(x).strip() for x in parsed.get("differential_ranked", []) if str(x).strip()][:5]
    questions = [str(x).strip() for x in parsed.get("open_questions", []) if str(x).strip()][:6]
    confidence = parsed.get("confidence_score", 0.5)
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))

    return DraftSOAP(
        subjective=str(parsed.get("subjective", "")).strip(),
        objective=str(parsed.get("objective", "")).strip(),
        assessment=str(parsed.get("assessment", "")).strip(),
        plan=str(parsed.get("plan", "")).strip(),
        differential_ranked=differential or ["Undifferentiated syndrome"],
        open_questions=questions,
        confidence_score=score,
    )


def _default_confidence_script() -> str:
    return (
        "def compute_confidence(medgemma_report, genotype_report):\n"
        "    med = medgemma_report or {}\n"
        "    geno = genotype_report or {}\n"
        "    obs = med.get('supporting_observations') or []\n"
        "    assessments = med.get('candidate_assessments') or []\n"
        "    hypotheses = geno.get('hypotheses') or []\n"
        "    bq = int(geno.get('biomcp_query_count') or 0)\n"
        "    be = int(geno.get('biomcp_empty_count') or 0)\n"
        "    score = 0.25\n"
        "    score += min(len(obs), 5) * 0.08\n"
        "    score += min(len(assessments), 5) * 0.06\n"
        "    score += min(len(hypotheses), 5) * 0.05\n"
        "    if bq > 0 and be >= bq:\n"
        "        score = 0.0\n"
        "    elif be > 0:\n"
        "        score -= 0.35\n"
        "    elif bq > 0:\n"
        "        score += 0.10\n"
        "    if score < 0.0:\n"
        "        score = 0.0\n"
        "    if score > 1.0:\n"
        "        score = 1.0\n"
        "    return float(score)\n"
    )


def _compute_confidence_via_python(state: BlackboardState) -> tuple[float, str, str]:
    script = _default_confidence_script()
    blocked = ["__import__", "open(", "exec(", "eval(", "subprocess", "os.", "sys."]
    if any(tok in script for tok in blocked):
        return 0.0, script, "confidence_script_blocked"

    safe_globals = {
        "__builtins__": {
            "len": len,
            "min": min,
            "max": max,
            "float": float,
            "int": int,
            "bool": bool,
            "str": str,
        }
    }
    safe_locals: dict[str, Any] = {}
    try:
        exec(script, safe_globals, safe_locals)
        fn = safe_locals.get("compute_confidence")
        if not callable(fn):
            return 0.0, script, "confidence_function_missing"
        med = asdict(state.medgemma_report) if state.medgemma_report else {}
        geno = asdict(state.genotype_report) if state.genotype_report else {}
        score = float(fn(med, geno))
    except Exception as exc:
        return 0.0, script, f"confidence_script_error:{type(exc).__name__}"
    return max(0.0, min(1.0, score)), script, ""


def run_supervisor_integrator_v2(state: BlackboardState, verifier_feedback: list[str] | None = None) -> DraftSOAP:
    verifier_feedback = verifier_feedback or []
    med = state.medgemma_report
    geno = state.genotype_report
    use_openai = os.getenv("MEDAGENT_USE_OPENAI", "0").strip() == "1"

    if use_openai:
        draft = _build_draft_with_openai(state, verifier_feedback)
    else:
        subjective = med.findings[0] if med and med.findings else "No subjective narrative available"

        objective_parts: list[str] = []
        if med and med.supporting_observations:
            objective_parts.extend(med.supporting_observations[:5])
        if state.patient_summary.key_labs:
            objective_parts.append(f"Labs available: {len(state.patient_summary.key_labs)}")
        if state.patient_summary.key_vitals:
            objective_parts.append(f"Vitals available: {len(state.patient_summary.key_vitals)}")
        objective = "; ".join(objective_parts) if objective_parts else "Objective evidence limited"

        differential = med.candidate_assessments[:5] if med else ["Undifferentiated syndrome"]

        assessment_lines = list(differential)
        if geno and geno.interpretations:
            assessment_lines.append("Genotype interpretation included as supporting evidence only")
        if verifier_feedback:
            assessment_lines.append("Verifier-requested updates addressed")
        assessment = "; ".join(assessment_lines)

        plan_items = [
            "Correlate symptoms with objective findings and timeline",
            "Prioritize evidence-backed differential items",
            "Collect missing objective data for unresolved uncertainty",
        ]
        if verifier_feedback:
            plan_items.append("Apply verifier patch list before release")
        if geno and geno.actionability_caveats:
            plan_items.append("Keep genotype-dependent recommendations caveated")

        draft = DraftSOAP(
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan="; ".join(plan_items),
            differential_ranked=differential,
            open_questions=[
                "What key negatives are still missing?",
                "Any temporal mismatch between symptoms and imaging findings?",
                "Any high-risk recommendation lacking strong evidence?",
            ],
            confidence_score=0.6,
        )

    if (
        os.getenv("MEDAGENT_USE_BIOMCP_SDK", "0").strip() == "1"
        and geno is not None
        and geno.biomcp_empty_count > 0
    ):
        draft.confidence_score = 0.0
        if "BioMCP returned empty evidence; confidence forced to 0.0 for critic review." not in draft.open_questions:
            draft.open_questions.append("BioMCP returned empty evidence; confidence forced to 0.0 for critic review.")

    confidence_script = ""
    if os.getenv("MEDAGENT_V2_SUPERVISOR_PYTHON_CONFIDENCE", "0").strip() == "1":
        score, confidence_script, err = _compute_confidence_via_python(state)
        draft.confidence_score = score
        if err:
            draft.confidence_score = 0.0
            draft.open_questions.append("Confidence script failed; forced 0.0 for critic review.")

    state.draft_soap = draft
    state.add_audit(
        "a3_supervisor",
        "write_draft_soap",
        differential=len(draft.differential_ranked),
        confidence_score=draft.confidence_score,
        used_openai=use_openai,
        confidence_script=confidence_script[:1200] if confidence_script else "",
    )
    return draft


def build_targeted_revision_requests(state: BlackboardState) -> tuple[list[str], list[str]]:
    if state.verifier_report.status == "PASS":
        return [], []

    a1_focus: list[str] = []
    a2_focus: list[str] = []
    for patch in state.verifier_report.patch_list:
        if patch.issue_type in {"cross_modal_conflict", "timeline_mismatch", "missing_negative"}:
            a1_focus.append(patch.message)
        if patch.issue_type in {"unsupported_claim", "high_risk_weak_evidence"}:
            a2_focus.append(patch.message)

    evolve_mode = os.getenv("MEDAGENT_V2_SUPERVISOR_EVOLVE_MODE", "0").strip() == "1"
    use_openai = os.getenv("MEDAGENT_USE_OPENAI", "0").strip() == "1"
    if evolve_mode and use_openai and state.verifier_report.patch_list:
        try:
            from openai import OpenAI

            model = os.getenv("MEDAGENT_OPENAI_MODEL", "gpt-5.2").strip() or "gpt-5.2"
            payload = {
                "verifier_patches": [
                    {
                        "issue_type": p.issue_type,
                        "message": p.message,
                        "suggested_fix": p.suggested_fix,
                    }
                    for p in state.verifier_report.patch_list[:8]
                ],
                "base_a1_focus": a1_focus[:4],
                "base_a2_focus": a2_focus[:4],
            }
            resp = OpenAI(max_retries=0).responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "Rewrite role-specific revision prompts for upstream agents. "
                            "Return strict JSON keys: a1_focus, a2_focus. "
                            "Each value must be a list of concise instruction strings."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
                ],
            )
            parsed = _safe_parse(getattr(resp, "output_text", "") or "")
            a1_new = [str(x).strip() for x in parsed.get("a1_focus", []) if str(x).strip()]
            a2_new = [str(x).strip() for x in parsed.get("a2_focus", []) if str(x).strip()]
            if a1_new:
                a1_focus = a1_new
            if a2_new:
                a2_focus = a2_new
        except Exception:
            pass

    return a1_focus[:4], a2_focus[:4]
