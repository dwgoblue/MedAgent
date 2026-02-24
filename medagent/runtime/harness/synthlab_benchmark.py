from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from time import perf_counter
from typing import Any

from medagent.runtime.agents.orchestrator.factory import build_orchestrator
from medagent.runtime.harness.biomcp_gate import grade_prediction_with_biomcp
from medagent.runtime.harness.eval_labels import extract_primary_encounter_dx, normalize_label
from medagent.runtime.harness.synthlab_runner import _import_synthlab, run_patient_case


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        pid = str(row.get("patient_id", "")).strip()
        if pid:
            done.add(pid)
    return done


def _safe_top1(problems: list[Any]) -> str:
    if not problems:
        return ""
    return str(problems[0] or "").strip()


def _safe_top3(problems: list[Any]) -> list[str]:
    return [str(x).strip() for x in problems[:3] if str(x).strip()]


def _calc_classification_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    labeled = [r for r in rows if r.get("gold_label_norm")]
    with_pred = [r for r in labeled if r.get("pred_top1_norm")]
    if not labeled:
        return {"top1_accuracy": 0.0, "top3_recall": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}

    top1 = sum(1 for r in labeled if r.get("is_top1_correct")) / len(labeled)
    top3 = sum(1 for r in labeled if r.get("is_top3_hit")) / len(labeled)

    labels = set()
    for r in with_pred:
        labels.add(r["gold_label_norm"])
        labels.add(r["pred_top1_norm"])

    if not with_pred or not labels:
        return {
            "top1_accuracy": round(top1, 4),
            "top3_recall": round(top3, 4),
            "macro_f1": 0.0,
            "micro_f1": 0.0,
        }

    tp_total = fp_total = fn_total = 0
    f1s: list[float] = []
    for label in labels:
        tp = sum(1 for r in with_pred if r["gold_label_norm"] == label and r["pred_top1_norm"] == label)
        fp = sum(1 for r in with_pred if r["gold_label_norm"] != label and r["pred_top1_norm"] == label)
        fn = sum(1 for r in with_pred if r["gold_label_norm"] == label and r["pred_top1_norm"] != label)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        denom = (2 * tp + fp + fn)
        f1s.append((2 * tp / denom) if denom else 0.0)

    micro_denom = (2 * tp_total + fp_total + fn_total)
    micro_f1 = (2 * tp_total / micro_denom) if micro_denom else 0.0

    return {
        "top1_accuracy": round(top1, 4),
        "top3_recall": round(top3, 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "micro_f1": round(micro_f1, 4),
    }


def run_benchmark(
    *,
    max_patients: int,
    modalities: list[str],
    output_dir: Path,
    download_if_missing: bool,
    use_biomcp_sdk: bool,
    resume_from: Path | None = None,
    kg_backend: str = "dashboard",
) -> dict[str, Any]:
    sl = _import_synthlab()
    if download_if_missing:
        components: list[str] = []
        for m in modalities:
            c = "csv" if m == "notes" else m
            if c not in components:
                components.append(c)
        sl.download_coherent_dataset(components=components, max_patients=max_patients, verbose=True)

    dataset = sl.load_multimodal_dataset(max_patients=max_patients, modalities=modalities)
    engine = build_orchestrator()

    output_dir.mkdir(parents=True, exist_ok=True)
    per_patient_path = output_dir / "benchmark_per_patient.jsonl"
    if resume_from is None:
        resume_from = per_patient_path
    done_ids = _load_done_ids(resume_from)

    rows: list[dict[str, Any]] = []
    modality_counter: Counter[str] = Counter()
    gate_counter: Counter[str] = Counter()

    for patient in dataset:
        patient_id = str(getattr(patient, "patient_id", "") or "").strip()
        if patient_id and patient_id in done_ids:
            continue

        t0 = perf_counter()
        run = run_patient_case(patient, engine=engine, kg_backend=kg_backend)
        final = run["result"]
        patient_meta = run["patient_meta"]
        patient_id_out = run["patient_id"]

        gold_raw = extract_primary_encounter_dx(patient) or ""
        gold_norm = normalize_label(gold_raw)
        pred_top1_raw = _safe_top1(final["problem_list_ranked"])
        pred_top1_norm = normalize_label(pred_top1_raw)
        pred_top3_norm = [normalize_label(x) for x in _safe_top3(final["problem_list_ranked"])]

        gate = grade_prediction_with_biomcp(
            predicted_label=pred_top1_raw or "unknown diagnosis",
            context=str(final.get("soap_final", "")),
            use_biomcp_sdk=use_biomcp_sdk,
        )

        is_top1_correct = bool(gold_norm and pred_top1_norm and gold_norm == pred_top1_norm)
        is_top3_hit = bool(gold_norm and gold_norm in pred_top3_norm)
        runtime_ms = round((perf_counter() - t0) * 1000, 2)

        row = {
            "patient_id": patient_id_out,
            "modalities_present": patient_meta.get("modalities", []),
            "gold_label_raw": gold_raw,
            "gold_label_norm": gold_norm,
            "pred_top1_raw": pred_top1_raw,
            "pred_top1_norm": pred_top1_norm,
            "pred_top3_norm": pred_top3_norm,
            "is_top1_correct": is_top1_correct,
            "is_top3_hit": is_top3_hit,
            "biomcp_grade": gate.grade,
            "biomcp_result_count": gate.result_count,
            "biomcp_citations": gate.citations,
            "biomcp_debug": {
                "attempted_calls": gate.attempted_calls,
                "errors": gate.errors,
            },
            "runtime_ms": runtime_ms,
            "problems_predicted": final.get("problem_list_ranked", []),
            "kg": run.get("kg", {}),
        }
        rows.append(row)
        _append_jsonl(per_patient_path, row)
        gate_counter[gate.grade] += 1

        for m in row["modalities_present"]:
            modality_counter[str(m)] += 1

    if not rows and per_patient_path.exists():
        rows = [json.loads(x) for x in per_patient_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        gate_counter = Counter(str(r.get("biomcp_grade", "unknown")) for r in rows)
        for r in rows:
            for m in r.get("modalities_present", []):
                modality_counter[str(m)] += 1

    cls_metrics = _calc_classification_metrics(rows)
    n = len(rows)
    verified_top1 = 0.0
    if n:
        verified_top1 = sum(
            1 for r in rows if r.get("is_top1_correct") and r.get("biomcp_grade") in {"pass", "weak"}
        ) / n

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_patients": n,
        "modalities_requested": modalities,
        "kg_backend": kg_backend,
        "modality_presence_counts": dict(modality_counter),
        "label_coverage_rate": round(sum(1 for r in rows if r.get("gold_label_norm")) / n, 4) if n else 0.0,
        "prediction_coverage_rate": round(sum(1 for r in rows if r.get("pred_top1_norm")) / n, 4) if n else 0.0,
        **cls_metrics,
        "verified_top1_accuracy": round(verified_top1, 4),
        "biomcp_pass_rate": round(gate_counter.get("pass", 0) / n, 4) if n else 0.0,
        "biomcp_weak_rate": round(gate_counter.get("weak", 0) / n, 4) if n else 0.0,
        "biomcp_fail_rate": round(gate_counter.get("fail", 0) / n, 4) if n else 0.0,
        "biomcp_not_run_rate": round(gate_counter.get("not_run", 0) / n, 4) if n else 0.0,
        "paths": {
            "per_patient": str(per_patient_path),
            "summary": str(output_dir / "benchmark_summary.json"),
        },
    }

    summary_path = output_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1k-scale SynthLab outcome benchmark")
    parser.add_argument("--max-patients", type=int, default=1000)
    parser.add_argument("--modalities", type=str, default="fhir,genomics,notes,dicom")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--use-biomcp-sdk", type=int, choices=[0, 1], default=1)
    parser.add_argument("--kg-backend", type=str, choices=["dashboard", "synthlab_notebook"], default="dashboard")
    parser.add_argument("--output-dir", type=str, default="medagent/runtime/examples/benchmark_runs/latest")
    parser.add_argument("--resume-from", type=str, default="")
    args = parser.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    summary = run_benchmark(
        max_patients=args.max_patients,
        modalities=modalities,
        output_dir=Path(args.output_dir),
        download_if_missing=args.download_if_missing,
        use_biomcp_sdk=bool(args.use_biomcp_sdk),
        resume_from=Path(args.resume_from) if args.resume_from.strip() else None,
        kg_backend=args.kg_backend,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
