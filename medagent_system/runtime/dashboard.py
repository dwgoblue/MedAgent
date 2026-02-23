from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _normalize_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append({str(k): _normalize_cell(v) for k, v in row.items()})
    return normalized


def _try_latest_run_json(repo_root: Path) -> Path | None:
    runs_root = repo_root / "medagent_system" / "runtime" / "examples" / "cluster_runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        list(runs_root.rglob("synthlab_output.json")) + list(runs_root.rglob("final_output.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _try_latest_benchmark_json(repo_root: Path) -> Path | None:
    runs_root = repo_root / "medagent_system" / "runtime" / "examples" / "cluster_runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        list(runs_root.rglob("benchmark_summary.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _find_logs_for_run(output_path: Path) -> tuple[Path | None, Path | None]:
    run_dir = output_path.parent
    logs_dir = run_dir / "logs"
    outputs = logs_dir / "agent_outputs.jsonl"
    comms = logs_dir / "agent_comms.jsonl"
    return (outputs if outputs.exists() else None, comms if comms.exists() else None)


def _find_benchmark_paths(output_path: Path) -> tuple[Path | None, Path | None]:
    if output_path.name == "benchmark_summary.json":
        summary = output_path
        per_patient = output_path.parent / "benchmark_per_patient.jsonl"
        return summary, (per_patient if per_patient.exists() else None)
    run_dir = output_path.parent
    summary = run_dir / "benchmark" / "benchmark_summary.json"
    per_patient = run_dir / "benchmark" / "benchmark_per_patient.jsonl"
    return (
        summary if summary.exists() else None,
        per_patient if per_patient.exists() else None,
    )


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    runs = payload.get("runs")
    if isinstance(runs, list):
        return [r for r in runs if isinstance(r, dict)]
    if "soap_final" in payload:
        return [{"patient_id": "single_case", "patient_meta": {}, "evaluation": {}, "result": payload}]
    return []


def _short(text: str, limit: int = 52) -> str:
    s = " ".join(text.split())
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _make_kg_dot(row: dict[str, Any]) -> str:
    patient_id = str(row.get("patient_id", "unknown_patient"))
    result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
    problems = result.get("problem_list_ranked", []) or []
    evidence = result.get("evidence_table", []) or []
    plans = result.get("plan_options_ranked_non_prescriptive", []) or []
    sensitivity = result.get("sensitivity_map", []) or []

    lines = [
        "digraph MedAgentKG {",
        '  rankdir="LR";',
        '  graph [bgcolor="white"];',
        '  node [style="filled,rounded", fontname="Helvetica", fontsize=10];',
        '  patient [label="Patient\\n%s", shape=box, fillcolor="#d8ecff"];' % _short(patient_id, 30).replace('"', "'"),
    ]

    problem_nodes: list[str] = []
    for idx, p in enumerate(problems[:10], 1):
        node = f"p{idx}"
        problem_nodes.append(node)
        label = _short(str(p), 42).replace('"', "'")
        lines.append(f'  {node} [label="Problem {idx}\\n{label}", fillcolor="#ffe8c2"];')
        lines.append(f'  patient -> {node} [label="has_problem"];')

    for idx, e in enumerate(evidence[:30], 1):
        if not isinstance(e, dict):
            continue
        node = f"e{idx}"
        claim = _short(str(e.get("claim", e.get("claim_text", "evidence"))), 40).replace('"', "'")
        status = str(e.get("status", "unknown"))
        category = str(e.get("category", "evidence"))
        lines.append(f'  {node} [label="{category}\\n{claim}\\n[{status}]", fillcolor="#e8f7d4"];')
        target = "patient"
        claim_l = claim.lower()
        for p_idx, p in enumerate(problems[:10], 1):
            p_txt = str(p).lower()
            if p_txt and (p_txt in claim_l or any(tok and tok in claim_l for tok in p_txt.split()[:3])):
                target = f"p{p_idx}"
                break
        lines.append(f'  {node} -> {target} [label="supports"];')

    for idx, plan in enumerate(plans[:8], 1):
        node = f"pl{idx}"
        text = _short(str(plan), 42).replace('"', "'")
        lines.append(f'  {node} [label="Plan {idx}\\n{text}", fillcolor="#f3ddff"];')
        target = problem_nodes[0] if problem_nodes else "patient"
        plan_l = str(plan).lower()
        for p_idx, p in enumerate(problems[:10], 1):
            p_txt = str(p).lower()
            if p_txt and any(tok and tok in plan_l for tok in p_txt.split()[:3]):
                target = f"p{p_idx}"
                break
        lines.append(f'  {node} -> {target} [label="targets"];')

    for idx, s in enumerate(sensitivity[:12], 1):
        node = f"s{idx}"
        if isinstance(s, dict):
            label = _short(f"{s.get('problem', 'problem')}: {s.get('sensitivity_class', 'unknown')}", 44)
        else:
            label = _short(str(s), 44)
        lines.append(f'  {node} [label="Sensitivity\\n{label}", fillcolor="#ffdede"];')
        lines.append(f'  {node} -> patient [label="robustness"];')

    lines.append("}")
    return "\n".join(lines)


def _render_reports(row: dict[str, Any]) -> None:
    result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
    eval_data = row.get("evaluation", {}) if isinstance(row.get("evaluation"), dict) else {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Problems", len(result.get("problem_list_ranked", []) or []))
    c2.metric("Evidence Rows", len(result.get("evidence_table", []) or []))
    c3.metric("Must-Verify", int(eval_data.get("must_verify_count", 0)))
    c4.metric("Pass Rate", float(eval_data.get("must_verify_pass_rate", 0.0)))

    st.subheader("SOAP Note")
    st.text_area(
        "soap_final",
        value=str(result.get("soap_final", "")),
        height=260,
        label_visibility="collapsed",
    )

    st.subheader("Ranked Problems")
    st.table(result.get("problem_list_ranked", []))

    st.subheader("Plan Options (Non-Prescriptive)")
    st.table(result.get("plan_options_ranked_non_prescriptive", []))

    st.subheader("Evidence Table")
    evidence = result.get("evidence_table", [])
    if isinstance(evidence, list) and evidence and isinstance(evidence[0], dict):
        st.dataframe(_normalize_rows(evidence), width="stretch")
    else:
        st.write("No evidence rows.")

    st.subheader("Sensitivity Map")
    sensitivity = result.get("sensitivity_map", [])
    if isinstance(sensitivity, list) and sensitivity and isinstance(sensitivity[0], dict):
        st.dataframe(_normalize_rows(sensitivity), width="stretch")
    else:
        st.write("No sensitivity rows.")

    st.subheader("Escalation Guidance")
    st.code(str(result.get("uncertainty_and_escalation_guidance", "")))

    with st.expander("Provenance JSON"):
        st.json(result.get("provenance", {}))

    with st.expander("Patient Metadata"):
        st.json(row.get("patient_meta", {}))


def _render_benchmark(summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    st.subheader("Benchmark Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", int(summary.get("n_patients", 0)))
    c2.metric("Top1 Accuracy", float(summary.get("top1_accuracy", 0.0)))
    c3.metric("Top3 Recall", float(summary.get("top3_recall", 0.0)))
    c4.metric("Verified Top1", float(summary.get("verified_top1_accuracy", 0.0)))

    d1, d2, d3 = st.columns(3)
    d1.metric("BioMCP Pass", float(summary.get("biomcp_pass_rate", 0.0)))
    d2.metric("BioMCP Weak", float(summary.get("biomcp_weak_rate", 0.0)))
    d3.metric("BioMCP Fail", float(summary.get("biomcp_fail_rate", 0.0)))

    with st.expander("Summary JSON"):
        st.json(summary)

    st.subheader("Per-Patient Rows")
    if rows:
        st.dataframe(_normalize_rows(rows), width="stretch")
    else:
        st.write("No benchmark per-patient rows found.")


def main() -> None:
    st.set_page_config(page_title="MedAgent Dashboard", layout="wide")
    st.title("MedAgent Run Dashboard")

    repo_root = Path(__file__).resolve().parents[2]
    latest = _try_latest_run_json(repo_root)
    latest_benchmark = _try_latest_benchmark_json(repo_root)
    default_path = str(latest_benchmark or latest) if (latest_benchmark or latest) else ""

    st.sidebar.header("Data Source")
    output_path_text = st.sidebar.text_input("Run JSON path", value=default_path)
    output_path = Path(output_path_text).expanduser() if output_path_text else None

    if not output_path or not output_path.exists():
        st.warning("Provide a valid path to synthlab_output.json or final_output.json.")
        st.stop()

    payload = _read_json(output_path)
    rows = _rows_from_payload(payload)

    outputs_log, comms_log = _find_logs_for_run(output_path)
    outputs_rows = _read_jsonl(outputs_log) if outputs_log else []
    comms_rows = _read_jsonl(comms_log) if comms_log else []
    outputs_rows_norm = _normalize_rows(outputs_rows)
    comms_rows_norm = _normalize_rows(comms_rows)
    bench_summary_path, bench_rows_path = _find_benchmark_paths(output_path)
    bench_summary = _read_json(bench_summary_path) if bench_summary_path else {}
    bench_rows = _read_jsonl(bench_rows_path) if bench_rows_path else []

    st.sidebar.caption(f"Loaded: `{output_path}`")
    st.sidebar.caption(f"Patients: {len(rows)}")
    if outputs_log:
        st.sidebar.caption(f"agent_outputs: `{outputs_log}` ({len(outputs_rows)} rows)")
    if comms_log:
        st.sidebar.caption(f"agent_comms: `{comms_log}` ({len(comms_rows)} rows)")
    if bench_summary_path:
        st.sidebar.caption(f"benchmark: `{bench_summary_path}`")
    if bench_rows_path:
        st.sidebar.caption(f"benchmark_rows: `{bench_rows_path}` ({len(bench_rows)} rows)")

    row: dict[str, Any] = {}
    if rows:
        labels = [f"{r.get('patient_id', 'unknown')} (#{i + 1})" for i, r in enumerate(rows)]
        selected = st.sidebar.selectbox("Patient", options=list(range(len(rows))), format_func=lambda i: labels[i])
        row = rows[selected]

    tab_overview, tab_reports, tab_kg, tab_logs, tab_bench = st.tabs(
        ["Overview", "SOAP + Reports", "Knowledge Graph", "Agent Logs", "Benchmark"]
    )

    with tab_overview:
        st.subheader("Run Summary")
        summary = payload.get("summary", {})
        if isinstance(summary, dict):
            a, b, c = st.columns(3)
            a.metric("Total Runs", int(summary.get("n_runs", len(rows))))
            b.metric("Avg Pass Rate", float(summary.get("avg_pass_rate", 0.0)))
            c.metric("Avg Fail Rate", float(summary.get("avg_fail_rate", 0.0)))
        st.write("Modalities:", ", ".join(payload.get("modalities", []) or []))
        st.write("Patient ID:", row.get("patient_id", "N/A"))

    with tab_reports:
        if rows:
            _render_reports(row)
        else:
            st.write("No per-patient run payload found in selected JSON.")

    with tab_kg:
        if rows:
            st.subheader("Case Knowledge Graph")
            st.caption("Graph is built from ranked problems, evidence, plan options, and sensitivity map.")
            st.graphviz_chart(_make_kg_dot(row), width="stretch")
        else:
            st.write("No graph data available for selected JSON.")

        if outputs_rows:
            st.subheader("Agent Stage Graph")
            edges: set[tuple[str, str]] = set()
            for item in outputs_rows:
                agent = str(item.get("agent", "unknown_agent"))
                stage = str(item.get("stage", "unknown_stage"))
                edges.add((agent, f"{agent}:{stage}"))
            dot = [
                "digraph AgentFlow {",
                '  rankdir="LR";',
                '  node [style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#eef5ff"];',
            ]
            for src, dst in sorted(edges):
                dot.append(f'  "{src}" -> "{dst}";')
            dot.append("}")
            st.graphviz_chart("\n".join(dot), width="stretch")

    with tab_logs:
        st.subheader("agent_outputs.jsonl")
        if outputs_rows:
            st.dataframe(outputs_rows_norm, width="stretch")
        else:
            st.write("No `agent_outputs.jsonl` found.")

        st.subheader("agent_comms.jsonl")
        if comms_rows:
            st.dataframe(comms_rows_norm, width="stretch")
        else:
            st.write("No `agent_comms.jsonl` found.")

    with tab_bench:
        if bench_summary:
            _render_benchmark(bench_summary, bench_rows)
        else:
            st.write("No benchmark summary found for selected run.")


if __name__ == "__main__":
    main()
