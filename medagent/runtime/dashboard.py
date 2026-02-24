from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import streamlit as st


def _zoomable_graph_html(svg_bytes: bytes, height_px: int = 500) -> str:
    """Return HTML for a zoomable/pannable graph (wheel zoom, drag to pan)."""
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"/><style>
  body {{ margin: 0; overflow: hidden; }}
  #c {{ overflow: auto; width: 100%; height: {height_px}px; cursor: grab; }}
  #c:active {{ cursor: grabbing; }}
  #z {{ display: inline-block; transform-origin: 0 0; }}
  #z img {{ display: block; vertical-align: top; }}
</style></head>
<body>
<div id="c"><div id="z"><img id="g" src="data:image/svg+xml;base64,{b64}" alt="graph"/></div></div>
<script>
(function() {{
  var s = 1, tx = 0, ty = 0, down = false, startX, startY, startTx, startTy;
  var z = document.getElementById('z');
  var c = document.getElementById('c');
  c.addEventListener('wheel', function(e) {{
    e.preventDefault();
    s *= e.deltaY > 0 ? 0.9 : 1.1;
    s = Math.max(0.2, Math.min(8, s));
    z.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
  }}, {{ passive: false }});
  c.addEventListener('mousedown', function(e) {{ down = true; startX = e.clientX; startY = e.clientY; startTx = tx; startTy = ty; }});
  c.addEventListener('mouseup', function() {{ down = false; }});
  c.addEventListener('mouseleave', function() {{ down = false; }});
  c.addEventListener('mousemove', function(e) {{
    if (!down) return;
    tx = startTx + (e.clientX - startX);
    ty = startTy + (e.clientY - startY);
    z.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
  }});
}})();
</script>
</body>
</html>
"""


def _render_graph(dot_str: str, key: str, height_px: int = 500) -> None:
    """Render graph with zoom/pan when SVG is available, otherwise fallback to static chart."""
    if not dot_str or not dot_str.strip():
        return
    svg_bytes = _render_dot_to_bytes(dot_str, "svg")
    if svg_bytes:
        try:
            from streamlit.components.v1 import html as st_html
            html = _zoomable_graph_html(svg_bytes, height_px)
            st_html(html, height=height_px + 20, scrolling=False)
        except Exception:
            st.graphviz_chart(dot_str, width="stretch")
    else:
        st.graphviz_chart(dot_str, width="stretch")


def _render_dot_to_bytes(dot_str: str, fmt: str) -> bytes | None:
    """Render DOT source to PNG or PDF bytes. Returns None if graphviz unavailable or render fails."""
    if not dot_str or not dot_str.strip():
        return None
    try:
        import graphviz
        g = graphviz.Source(dot_str)
        return g.pipe(format=fmt)
    except Exception:
        return None


# Match directed edges: "node1" -> "node2";  or  node1 -> node2 [label="..."];
_DOT_EDGE_RE = re.compile(
    r'\s*(?:"([^"]+)"|(\w+))\s*->\s*(?:"([^"]+)"|(\w+))\s*(?:\[label="([^"]*)"\])?\s*;'
)
# Match node with label:  id [label="..."];  or  id [label="...", shape=box];
_DOT_NODE_RE = re.compile(r'\s*(\w+)\s*\[\s*label\s*=\s*"((?:[^"\\]|\\.)*)"[^\]]*\]\s*;')


def _parse_dot_label(label: str) -> tuple[str, str]:
    """Split DOT label into display label and optional type from trailing \\n[type]."""
    label = label.replace("\\n", "\n")
    if "\n" in label:
        rest, tail = label.rsplit("\n", 1)
        type_match = re.match(r"\[([^\]]*)\]$", tail.strip())
        if type_match:
            return (rest.strip(), type_match.group(1).strip())
    return (label.strip(), "")


def _dot_to_graph_attrs(dot_str: str) -> tuple[list[tuple[str, dict[str, str]]], list[tuple[str, str, dict[str, str]]]]:
    """Parse DOT into (nodes with attrs, edges with attrs). Node: (id, {label?, type?}). Edge: (src, tgt, {relation?})."""
    nodes: list[tuple[str, dict[str, str]]] = []
    edges: list[tuple[str, str, dict[str, str]]] = []
    seen_nodes: set[str] = set()
    for line in dot_str.splitlines():
        line = line.strip()
        if not line or line.startswith("}"):
            continue
        m = _DOT_NODE_RE.match(line)
        if m:
            nid, label_raw = m.group(1), m.group(2)
            if nid not in seen_nodes:
                seen_nodes.add(nid)
                label, ntype = _parse_dot_label(label_raw)
                attrs: dict[str, str] = {}
                if label:
                    attrs["label"] = label
                if ntype:
                    attrs["type"] = ntype
                nodes.append((nid, attrs))
            continue
        m = _DOT_EDGE_RE.match(line)
        if m:
            src = m.group(1) or m.group(2)
            tgt = m.group(3) or m.group(4)
            rel = (m.group(5) or "").strip().replace("\\n", " ")
            if src and tgt:
                for nid in (src, tgt):
                    if nid not in seen_nodes:
                        seen_nodes.add(nid)
                        nodes.append((nid, {"label": nid}))
                edges.append((src, tgt, {"relation": rel} if rel else {}))
    return (nodes, edges)


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _dot_to_graphml(dot_str: str) -> str:
    """Convert DOT to GraphML XML preserving node (label, type) and edge (relation) attributes. Load with nx.read_graphml()."""
    if not dot_str or not dot_str.strip():
        return ""
    nodes, edges = _dot_to_graph_attrs(dot_str)
    # Ensure we have all nodes referenced by edges
    node_ids = {n[0] for n in nodes}
    for src, tgt, _ in edges:
        for nid in (src, tgt):
            if nid not in node_ids:
                node_ids.add(nid)
                nodes.append((nid, {"label": nid}))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for nid, attrs in nodes:
        lines.append(f'    <node id="{_escape_xml(nid)}">')
        for k, v in attrs.items():
            if v:
                lines.append(f'      <data key="{k}">{_escape_xml(v)}</data>')
        lines.append("    </node>")
    for src, tgt, attrs in edges:
        lines.append(f'    <edge source="{_escape_xml(src)}" target="{_escape_xml(tgt)}">')
        for k, v in attrs.items():
            if v:
                lines.append(f'      <data key="{k}">{_escape_xml(v)}</data>')
        lines.append("    </edge>")
    lines.extend(["  </graph>", "</graphml>"])
    return "\n".join(lines)


def _filename_suffix(run_id: str | None, patient_id: str | None) -> str:
    """Return a safe suffix for download filenames: run_id and patient_id joined with underscore."""
    parts: list[str] = []
    for v in (run_id, patient_id):
        if v is None or str(v).strip() == "":
            continue
        s = re.sub(r"[^\w\-.]", "_", str(v).strip()).strip("_.")
        if s:
            parts.append(s[:64])
    return "_".join(parts) if parts else ""


def _graph_download_buttons(
    dot_str: str,
    filename_base: str,
    raw_text: tuple[str, str] | None = None,
    run_id: str | None = None,
    patient_id: str | None = None,
) -> None:
    """Add PNG, PDF, GraphML (metadata), and optionally raw text download buttons in one row."""
    if not dot_str or not dot_str.strip():
        if not raw_text or not raw_text[0]:
            return
    base = (filename_base or "graph").replace(" ", "_")
    suffix = _filename_suffix(run_id, patient_id)
    if suffix:
        base = f"{base}_{suffix}"
    has_dot = bool(dot_str and dot_str.strip())
    n_cols = (3 + 1) if (has_dot and raw_text and raw_text[0]) else (3 if has_dot else 1)
    cols = st.columns(n_cols)
    idx = 0
    if dot_str and dot_str.strip():
        with cols[idx]:
            png_bytes = _render_dot_to_bytes(dot_str, "png")
            if png_bytes:
                st.download_button(
                    "Download PNG",
                    data=png_bytes,
                    file_name=f"{base}.png",
                    mime="image/png",
                    key=f"dl_png_{base}",
                )
            else:
                st.caption("PNG (graphviz required)")
        idx += 1
        with cols[idx]:
            pdf_bytes = _render_dot_to_bytes(dot_str, "pdf")
            if pdf_bytes:
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"{base}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{base}",
                )
            else:
                st.caption("PDF (graphviz required)")
        idx += 1
        with cols[idx]:
            graphml = _dot_to_graphml(dot_str)
            st.download_button(
                "Download GraphML",
                data=graphml,
                file_name=f"{base}.graphml",
                mime="application/xml",
                key=f"dl_graphml_{base}",
            )
            st.caption("Node: label, type. Edge: relation.")
        idx += 1
    if raw_text and raw_text[0]:
        with cols[idx]:
            raw_fname = raw_text[1]
            raw_fname = f"{raw_fname.removesuffix('.txt')}_{suffix}.txt" if suffix else raw_fname
            st.download_button(
                "Download raw text (.txt)",
                data=raw_text[0],
                file_name=raw_fname,
                mime="text/plain",
                key=f"dl_raw_{base}",
            )


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
    runs_root = repo_root / "medagent" / "runtime" / "examples" / "cluster_runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        list(runs_root.rglob("synthlab_output.json")) + list(runs_root.rglob("final_output.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _try_latest_benchmark_json(repo_root: Path) -> Path | None:
    runs_root = repo_root / "medagent" / "runtime" / "examples" / "cluster_runs"
    if not runs_root.exists():
        return None
    candidates = sorted(
        list(runs_root.rglob("benchmark_summary.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _list_run_json_paths(repo_root: Path) -> list[Path]:
    """Return all discoverable run JSON paths (synthlab_output, final_output, benchmark_summary), deduplicated."""
    runs_root = repo_root / "medagent" / "runtime" / "examples" / "cluster_runs"
    if not runs_root.exists():
        return []
    candidates = (
        list(runs_root.rglob("synthlab_output.json"))
        + list(runs_root.rglob("final_output.json"))
        + list(runs_root.rglob("benchmark_summary.json"))
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in candidates:
        try:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(p)
        except OSError:
            continue
    return unique


def _sort_run_json_paths(paths: list[Path], order: str) -> list[Path]:
    """Sort paths by the given order: date_desc, date_asc, path_asc, path_desc, run_id_asc, run_id_desc."""
    if not paths:
        return paths
    if order == "date_desc":
        return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    if order == "date_asc":
        return sorted(paths, key=lambda p: p.stat().st_mtime)
    if order == "path_asc":
        return sorted(paths, key=str)
    if order == "path_desc":
        return sorted(paths, key=str, reverse=True)
    if order == "run_id_asc":
        return sorted(paths, key=lambda p: (p.parent.name, p.name))
    if order == "run_id_desc":
        return sorted(paths, key=lambda p: (p.parent.name, p.name), reverse=True)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


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


def _causal_graph_text_to_dot(causal_graph: str) -> str | None:
    """
    Convert SynthLab causal graph text (e.g. "Node[type] --(relation)--> Node[type]")
    to DOT for rendering. Returns None if empty or unparseable.

    - Risk-factor relations: "non-modifiable" and "modifiable" are reversed so the graph
      shows cause -> effect (e.g. Age -> CHF, not CHF -> Age).
    - Nodes are deduplicated by canonical name (e.g. Chronic_Congestive_Hard_Failure
      and Chronic_Congestive_Heart_Failure become one node).
    """
    text = (causal_graph or "").strip()
    if not text:
        return None
    arrow_pattern = re.compile(
        r"(.+?)\[(\w+)\]\s*--\(([^)]+)\)-->\s*(.+?)\[(\w+)\]",
        re.IGNORECASE,
    )
    arrow_simple = re.compile(
        r"(.+?)\[(\w+)\]\s*(=>|<=>|\+\+>|\+>|\?\+>|--?>|->|\?->|\?=>|~>)\s*(.+?)\[(\w+)\]",
        re.IGNORECASE,
    )
    nodes: dict[str, tuple[str, str]] = {}  # id -> (label, type)
    edges: list[tuple[str, str, str]] = []  # (src_id, tgt_id, relation)

    def _canonical_name(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
        s = re.sub(r"_+", "_", s)
        s = s.replace("hard_failure", "heart_failure")
        return s or "n"

    def _add_node(name: str, ntype: str) -> str:
        c = _canonical_name(name)
        nid = "n_" + c[:50]
        if nid not in nodes:
            label = _short(name.replace('"', "'"), 32)
            nodes[nid] = (label, ntype.strip().lower())
        else:
            existing_label = nodes[nid][0]
            if "heart" in name.lower() and "hard" in existing_label.lower():
                nodes[nid] = (_short(name.replace('"', "'"), 32), ntype.strip().lower())
        return nid

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = arrow_pattern.fullmatch(line)
        if m:
            src_name, src_type, rel, tgt_name, tgt_type = m.groups()
            src_id = _add_node(src_name, src_type)
            tgt_id = _add_node(tgt_name, tgt_type)
            rel_clean = rel.strip()[:24].replace('"', "'")
            rel_lower = rel_clean.lower()
            if rel_lower in ("non-modifiable", "modifiable"):
                edges.append((tgt_id, src_id, "risk factor (" + rel_clean + ")"))
            else:
                edges.append((src_id, tgt_id, rel_clean))
            continue
        m = arrow_simple.search(line)
        if m:
            src_name, src_type, arrow, tgt_name, tgt_type = m.groups()
            src_id = _add_node(src_name, src_type)
            tgt_id = _add_node(tgt_name, tgt_type)
            edges.append((src_id, tgt_id, arrow.replace('"', "'")))
            continue

    if not nodes and not edges:
        return None
    seen_edge: set[tuple[str, str]] = set()
    unique_edges: list[tuple[str, str, str]] = []
    for s, t, r in edges:
        if (s, t) not in seen_edge:
            seen_edge.add((s, t))
            unique_edges.append((s, t, r))
    lines = [
        "digraph SynthLabCausal {",
        '  rankdir="LR";',
        '  node [style="filled,rounded", fontname="Helvetica", fontsize=9, fillcolor="#e8f4fc"];',
    ]
    for nid, (label, ntype) in nodes.items():
        lines.append(f'  {nid} [label="{label}\\n[{ntype}]"];')
    for src, tgt, rel in unique_edges:
        lines.append(f'  {src} -> {tgt} [label="{rel}"];')
    lines.append("}")
    return "\n".join(lines)


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
    edge_list: list[tuple[str, str, str]] = []
    for idx, p in enumerate(problems[:10], 1):
        node = f"p{idx}"
        problem_nodes.append(node)
        label = _short(str(p), 42).replace('"', "'")
        lines.append(f'  {node} [label="Problem {idx}\\n{label}", fillcolor="#ffe8c2"];')
        edge_list.append(("patient", node, "has_problem"))

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
        edge_list.append((node, target, "supports"))

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
        edge_list.append((node, target, "targets"))

    for idx, s in enumerate(sensitivity[:12], 1):
        node = f"s{idx}"
        if isinstance(s, dict):
            label = _short(f"{s.get('problem', 'problem')}: {s.get('sensitivity_class', 'unknown')}", 44)
        else:
            label = _short(str(s), 44)
        lines.append(f'  {node} [label="Sensitivity\\n{label}", fillcolor="#ffdede"];')
        edge_list.append((node, "patient", "robustness"))

    seen_edge = set()
    for src, tgt, lbl in edge_list:
        if (src, tgt) not in seen_edge:
            seen_edge.add((src, tgt))
            lines.append(f'  {src} -> {tgt} [label="{lbl}"];')
    lines.append("}")
    return "\n".join(lines)


def _dot_from_kg_artifact(kg: dict[str, Any], patient_id: str) -> str:
    backend = str(kg.get("backend", "")).strip()
    if backend == "synthlab_notebook":
        graph = kg.get("graph", {})
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        edges = graph.get("edges", []) if isinstance(graph, dict) else []
        lines = [
            "digraph SynthLabKG {",
            '  rankdir="LR";',
            '  node [style="filled,rounded", fontname="Helvetica", fontsize=10, fillcolor="#eef5ff"];',
        ]
        for idx, n in enumerate(nodes):
            if not isinstance(n, dict):
                continue
            label = _short(str(n.get("name", n.get("label", f"node_{idx}"))), 48).replace('"', "'")
            ntype = str(n.get("type", "unknown"))
            lines.append(f'  n{idx} [label="{label}\\n[{ntype}]"];')
        seen_edge: set[tuple[int, int]] = set()
        for eidx, e in enumerate(edges):
            if not isinstance(e, dict):
                continue
            src_val = e.get("source")
            if not src_val and isinstance(e.get("sources"), list) and e["sources"]:
                src_val = e["sources"][0]
            src = str(src_val or "")
            tgt = str(e.get("target") or "")
            rel = str(e.get("type") or e.get("relation") or "rel")
            src_idx = next((i for i, n in enumerate(nodes) if str(n.get("name", n.get("label", ""))) == src), None)
            tgt_idx = next((i for i, n in enumerate(nodes) if str(n.get("name", n.get("label", ""))) == tgt), None)
            if src_idx is None or tgt_idx is None:
                continue
            if (src_idx, tgt_idx) in seen_edge:
                continue
            seen_edge.add((src_idx, tgt_idx))
            lines.append(f'  n{src_idx} -> n{tgt_idx} [label="{rel}"];')
        lines.append("}")
        return "\n".join(lines)

    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    if isinstance(nodes, list) and isinstance(edges, list) and nodes:
        lines = [
            "digraph MedAgentSavedKG {",
            '  rankdir="LR";',
            '  node [style="filled,rounded", fontname="Helvetica", fontsize=10];',
        ]
        id_map: dict[str, int] = {}
        for idx, n in enumerate(nodes):
            if not isinstance(n, dict):
                continue
            nid = str(n.get("id", f"n{idx}"))
            id_map[nid] = idx
            label = _short(str(n.get("label", nid)), 48).replace('"', "'")
            lines.append(f'  n{idx} [label="{label}"];')
        seen_edge_med: set[tuple[int, int]] = set()
        for e in edges:
            if not isinstance(e, dict):
                continue
            s = id_map.get(str(e.get("source", "")))
            t = id_map.get(str(e.get("target", "")))
            if s is None or t is None:
                continue
            if (s, t) in seen_edge_med:
                continue
            seen_edge_med.add((s, t))
            rel = _short(str(e.get("relation", "rel")), 24).replace('"', "'")
            lines.append(f'  n{s} -> n{t} [label="{rel}"];')
        lines.append("}")
        return "\n".join(lines)

    return _make_kg_dot({"patient_id": patient_id, "result": {}})


def _get_prompts_used(row: dict[str, Any]) -> dict[str, Any]:
    """Extract prompts_used from result provenance (blackboard)."""
    result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
    prov = result.get("provenance", {}) if isinstance(result.get("provenance"), dict) else {}
    blackboard = prov.get("blackboard", {}) if isinstance(prov.get("blackboard"), dict) else {}
    return blackboard.get("prompts_used", {}) if isinstance(blackboard.get("prompts_used"), dict) else {}


def _render_prompt_info(
    prompts_used: dict[str, Any], keys: list[str], label: str = "Prompts used", key_prefix: str = "prompt"
) -> None:
    """Render an info control that shows system and user prompt(s) in separate boxes for each key."""
    if not keys:
        return
    subset = {k: v for k, v in prompts_used.items() if k in keys}
    if not subset:
        return
    def _content():
        st.caption(label)
        for name, data in subset.items():
            st.markdown(f"**{name}**")
            system_text = ""
            user_text = ""
            if isinstance(data, dict):
                system_text = str(data.get("system", "") or "")
                user_text = str(data.get("user", data.get("prompt", "")) or "")
            else:
                user_text = str(data)
            st.markdown("**System prompt**")
            st.text_area(
                "System",
                value=system_text if system_text else "(No system prompt)",
                height=100,
                key=f"{key_prefix}_{name}_sys",
                label_visibility="collapsed",
            )
            st.markdown("**User prompt**")
            st.text_area(
                "User",
                value=user_text if user_text else "(No user prompt)",
                height=120,
                key=f"{key_prefix}_{name}_user",
                label_visibility="collapsed",
            )
            if isinstance(data, dict):
                if data.get("note"):
                    st.caption(str(data["note"]))
                if data.get("inputs"):
                    st.caption(f"Inputs: {data['inputs']}")
            st.divider()
    try:
        with st.popover("ℹ️"):
            _content()
    except Exception:
        with st.expander("ℹ️ " + label):
            st.json(subset)


def _section_header_with_prompts(
    title: str, prompts_used: dict[str, Any], keys: list[str]
) -> None:
    """Render a subheader with an info button that shows prompts for this section."""
    col1, col2 = st.columns([6, 1])
    with col1:
        st.subheader(title)
    with col2:
        prefix = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
        _render_prompt_info(prompts_used, keys, "Prompts used", key_prefix=f"prompt_{prefix}")


def _render_reports(row: dict[str, Any]) -> None:
    result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
    eval_data = row.get("evaluation", {}) if isinstance(row.get("evaluation"), dict) else {}
    prompts_used = _get_prompts_used(row)
    prov = result.get("provenance", {}) if isinstance(result.get("provenance"), dict) else {}
    soap_from_synthlab = prov.get("soap_source") == "synthlab_generator"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Problems", len(result.get("problem_list_ranked", []) or []))
    c2.metric("Evidence Rows", len(result.get("evidence_table", []) or []))
    c3.metric("Must-Verify", int(eval_data.get("must_verify_count", 0)))
    c4.metric("Pass Rate", float(eval_data.get("must_verify_pass_rate", 0.0)))

    # When SOAP was generated by SynthLab, show only SynthLab SOAP prompts (synthlab_causal_graph is shown in Knowledge Graph tab)
    if soap_from_synthlab:
        soap_prompt_keys = ["synthlab_soap_generation", "synthlab_aggregate", "synthlab_grounded_relationships"]
    else:
        soap_prompt_keys = ["medgemma_reporter", "supervisor_draft", "synthlab_soap_generation", "synthlab_aggregate", "synthlab_grounded_relationships"]
    _section_header_with_prompts("SOAP Note", prompts_used, soap_prompt_keys)
    st.text_area(
        "soap_final",
        value=str(result.get("soap_final", "")),
        height=260,
        label_visibility="collapsed",
    )

    _section_header_with_prompts("Ranked Problems", prompts_used, ["supervisor_draft"])
    st.table(result.get("problem_list_ranked", []))

    _section_header_with_prompts("Plan Options (Non-Prescriptive)", prompts_used, ["supervisor_draft"])
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
        ridx = st.selectbox("Benchmark patient row", options=list(range(len(rows))), format_func=lambda i: str(rows[i].get("patient_id", f"row_{i}")))
        r = rows[ridx]
        kg = r.get("kg", {})
        st.subheader("Benchmark Patient KG")
        bench_dot = _dot_from_kg_artifact(kg if isinstance(kg, dict) else {}, str(r.get("patient_id", "patient")))
        _render_graph(bench_dot, "graph_benchmark_kg")
        _graph_download_buttons(bench_dot, "benchmark_patient_kg", run_id=run_id, patient_id=r.get("patient_id"))
        if isinstance(kg, dict) and kg.get("backend") == "synthlab_notebook":
            with st.expander("SynthLab KG Summary"):
                st.write(kg.get("summary", ""))
    else:
        st.write("No benchmark per-patient rows found.")


def _extract_biomcp_memory_stats(comms_rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    planning_rows: list[dict[str, Any]] = []
    stats = {
        "planning_events": 0,
        "memory_hits": 0,
        "memory_misses": 0,
        "sdk_calls": 0,
        "sdk_non_empty": 0,
        "sdk_empty": 0,
        "fallback_notices": 0,
    }
    for row in comms_rows:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind", ""))
        sender = str(row.get("sender", ""))
        receiver = str(row.get("receiver", ""))
        prompt = row.get("prompt", {})

        if receiver == "biomcp_query_memory" and kind == "planning":
            stats["planning_events"] += 1
            mem_hit = False
            if isinstance(prompt, dict):
                mem_hit = bool(prompt.get("memory_hit", False))
            if mem_hit:
                stats["memory_hits"] += 1
            else:
                stats["memory_misses"] += 1
            planning_rows.append(row)

        if receiver == "biomcp_sdk" and kind == "tool_call":
            stats["sdk_calls"] += 1

        if sender == "biomcp_sdk" and kind == "tool_result":
            result_count = 0
            if isinstance(prompt, dict):
                try:
                    result_count = int(prompt.get("result_count", 0) or 0)
                except Exception:
                    result_count = 0
            if result_count > 0:
                stats["sdk_non_empty"] += 1
            else:
                stats["sdk_empty"] += 1

        if sender == "biomcp_sdk" and kind == "fallback_notice":
            stats["fallback_notices"] += 1

    return stats, planning_rows


def main() -> None:
    st.set_page_config(page_title="MedAgent Dashboard", layout="wide")
    st.title("MedAgent Run Dashboard")

    repo_root = Path(__file__).resolve().parents[2]
    latest = _try_latest_run_json(repo_root)
    latest_benchmark = _try_latest_benchmark_json(repo_root)
    default_path = str((latest or latest_benchmark).resolve()) if (latest or latest_benchmark) else ""

    st.sidebar.header("Data Source")
    run_json_paths = _list_run_json_paths(repo_root)
    sort_options = {
        "Date (newest first)": "date_desc",
        "Date (oldest first)": "date_asc",
        "Full path (A–Z)": "path_asc",
        "Full path (Z–A)": "path_desc",
        "Run ID (A–Z)": "run_id_asc",
        "Run ID (Z–A)": "run_id_desc",
    }
    sort_label = st.sidebar.selectbox(
        "Sort runs by",
        options=list(sort_options.keys()),
        index=0,
        key="dashboard_sort_runs",
        help="Order of runs in the dropdown below.",
    )
    sort_order = sort_options[sort_label]
    run_json_paths = _sort_run_json_paths(run_json_paths, sort_order)
    custom_label = "(Enter path manually)"
    path_options = [custom_label] + [str(p.resolve()) for p in run_json_paths]

    def _path_display_label(path_str: str) -> str:
        if path_str == custom_label:
            return path_str
        p = Path(path_str)
        return f"{p.parent.name} / {p.name}"

    default_index = 0
    if default_path and default_path in path_options:
        default_index = path_options.index(default_path)
    select_help = "Pick a run from discovered files, or enter a path manually below."
    tooltip_path = st.session_state.get("dashboard_last_path") or (
        path_options[default_index] if default_index < len(path_options) and path_options[default_index] != custom_label else None
    )
    select_help_with_path = f"{select_help} Full path: {tooltip_path}" if tooltip_path else select_help
    selected_from_list = st.sidebar.selectbox(
        "Select run",
        options=path_options,
        index=default_index,
        key="dashboard_select_run",
        format_func=_path_display_label,
        help=select_help_with_path,
    )
    if selected_from_list == custom_label:
        custom_default = st.session_state.get("dashboard_last_path", default_path)
        output_path_text = st.sidebar.text_input("Run JSON path", value=custom_default, key="dashboard_path_custom", help="Path to synthlab_output.json, final_output.json, or benchmark_summary.json")
        output_path = Path(output_path_text).expanduser() if output_path_text else None
    else:
        full_path = str(Path(selected_from_list).expanduser().resolve())
        output_path = Path(full_path)
        st.session_state["dashboard_last_path"] = full_path

    if not output_path or not output_path.exists():
        st.warning("Provide a valid path to synthlab_output.json or final_output.json.")
        st.stop()

    st.session_state["dashboard_last_path"] = str(output_path)

    payload = _read_json(output_path)
    rows = _rows_from_payload(payload)

    # If user selected a benchmark_summary.json, it has no "runs"; try same run dir for run JSON so tabs have data
    logs_base_path = output_path
    if not rows and output_path.name == "benchmark_summary.json":
        run_dir = output_path.parent.parent
        for run_name in ("synthlab_output.json", "final_output.json"):
            run_path = run_dir / run_name
            if run_path.exists():
                payload = _read_json(run_path)
                rows = _rows_from_payload(payload)
                logs_base_path = run_path  # resolve logs from run dir
                break

    run_id = logs_base_path.parent.name
    outputs_log, comms_log = _find_logs_for_run(logs_base_path)
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
        if isinstance(summary, dict) and summary:
            a, b, c = st.columns(3)
            a.metric("Total Runs", int(summary.get("n_runs", len(rows))))
            b.metric("Avg Pass Rate", float(summary.get("avg_pass_rate", 0.0)))
            c.metric("Avg Fail Rate", float(summary.get("avg_fail_rate", 0.0)))
        elif not rows and "n_patients" in payload:
            # Benchmark summary only (no "runs"); show benchmark metrics
            a, b, c = st.columns(3)
            a.metric("Patients", int(payload.get("n_patients", 0)))
            b.metric("Top-1 Accuracy", float(payload.get("top1_accuracy", 0.0)))
            c.metric("Top-3 Recall", float(payload.get("top3_recall", 0.0)))
            st.caption("This file is a benchmark summary only. For per-patient SOAP/reports, open the run directory's synthlab_output.json or final_output.json.")
        st.write("Modalities:", ", ".join(payload.get("modalities", payload.get("modalities_requested", [])) or []))
        st.write("Patient ID:", row.get("patient_id", "N/A"))

    with tab_reports:
        if rows:
            _render_reports(row)
        else:
            st.write("No per-patient run payload found in selected JSON.")

    with tab_kg:
        if rows:
            kg_prompts = _get_prompts_used(row)
            kg = row.get("kg", {}) if isinstance(row.get("kg"), dict) else {}
            causal_graph = (kg.get("causal_graph") or "").strip() if isinstance(kg.get("causal_graph"), str) else ""

            # 1. SynthLab Causal Graph first
            if causal_graph:
                if "synthlab_causal_graph" in kg_prompts:
                    _section_header_with_prompts("SynthLab Causal Graph", kg_prompts, ["synthlab_causal_graph"])
                else:
                    st.subheader("SynthLab Causal Graph")
                causal_dot = _causal_graph_text_to_dot(causal_graph)
                if causal_dot:
                    st.caption("Knowledge graph generated by SynthLab from the SOAP note.")
                    _render_graph(causal_dot, "graph_synthlab_causal")
                    _graph_download_buttons(
                        causal_dot,
                        "synthlab_causal_graph",
                        raw_text=(causal_graph, "synthlab_causal_graph_raw.txt"),
                        run_id=run_id,
                        patient_id=row.get("patient_id"),
                    )
                else:
                    _graph_download_buttons(
                        "",
                        "synthlab_causal_graph",
                        raw_text=(causal_graph, "synthlab_causal_graph_raw.txt"),
                        run_id=run_id,
                        patient_id=row.get("patient_id"),
                    )

            # 2. Case Knowledge Graph
            case_kg_keys = ["supervisor_draft"]
            if any(k in kg_prompts for k in case_kg_keys):
                _section_header_with_prompts("Case Knowledge Graph", kg_prompts, case_kg_keys)
            else:
                st.subheader("Case Knowledge Graph")
            if kg:
                backend = kg.get("backend", "unknown")
                st.caption(f"KG backend: {backend}")
                case_dot = _dot_from_kg_artifact(kg, str(row.get("patient_id", "patient")))
                _render_graph(case_dot, "graph_case_kg")
                _graph_download_buttons(case_dot, "case_knowledge_graph", run_id=run_id, patient_id=row.get("patient_id"))
                if backend == "synthlab_notebook":
                    with st.expander("SynthLab KG Summary"):
                        st.write(kg.get("summary", ""))
            else:
                st.caption("Graph is built from ranked problems, evidence, plan options, and sensitivity map.")
                case_dot = _make_kg_dot(row)
                _render_graph(case_dot, "graph_case_kg_alt")
                _graph_download_buttons(case_dot, "case_knowledge_graph", run_id=run_id, patient_id=row.get("patient_id"))
        else:
            st.write("No graph data available for selected JSON.")

        # 3. Agent Stage Graph
        if outputs_rows:
            st.subheader("Agent Stage Graph")
            st.caption("Built from agent_outputs.jsonl (pipeline stage flow; no prompt).")
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
            agent_dot = "\n".join(dot)
            _render_graph(agent_dot, "graph_agent_stage")
            _graph_download_buttons(agent_dot, "agent_stage_graph", run_id=run_id, patient_id=row.get("patient_id"))

    with tab_logs:
        st.subheader("agent_outputs.jsonl")
        if outputs_rows:
            st.dataframe(outputs_rows_norm, width="stretch")
        else:
            st.write("No `agent_outputs.jsonl` found.")

        st.subheader("BioMCP Query-Memory")
        if comms_rows:
            stats, planning_rows = _extract_biomcp_memory_stats(comms_rows)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Planning Events", stats["planning_events"])
            m2.metric("Memory Hits", stats["memory_hits"])
            m3.metric("Memory Misses", stats["memory_misses"])
            m4.metric("SDK Calls", stats["sdk_calls"])
            n1, n2, n3 = st.columns(3)
            n1.metric("SDK Non-Empty", stats["sdk_non_empty"])
            n2.metric("SDK Empty", stats["sdk_empty"])
            n3.metric("Fallback Notices", stats["fallback_notices"])
            if planning_rows:
                st.dataframe(_normalize_rows(planning_rows), width="stretch")
            else:
                st.write("No query-memory planning events found in `agent_comms.jsonl`.")
        else:
            st.write("No `agent_comms.jsonl` found.")

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
