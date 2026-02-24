from __future__ import annotations

import json
import os
from datetime import datetime, timezone

UTC = timezone.utc
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_obj(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_obj(v) for v in value]
    if hasattr(value, "__dict__"):
        return _safe_obj(vars(value))
    return str(value)


def append_jsonl(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": _now_iso(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


class RunLogger:
    def __init__(self, run_dir: Path, *, mode: str, patient_id: str) -> None:
        self.run_dir = run_dir
        self.mode = mode
        self.patient_id = patient_id
        self.outputs_path = run_dir / "agent_outputs.jsonl"
        self.comms_path = run_dir / "agent_comms.jsonl"
        self.meta_path = run_dir / "run_meta.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta()
        # Allow lower-level modules to append prompt communication events.
        os.environ["MEDAGENT_COMM_LOG_PATH"] = str(self.comms_path)

    @classmethod
    def from_env(cls, *, mode: str, patient_id: str) -> "RunLogger | None":
        root = os.getenv("MEDAGENT_RUN_LOG_DIR", "").strip()
        if not root:
            return None
        run_dir = Path(root)
        return cls(run_dir, mode=mode, patient_id=patient_id)

    def _write_meta(self) -> None:
        payload = {
            "created_at": _now_iso(),
            "mode": self.mode,
            "patient_id": self.patient_id,
            "files": {
                "agent_outputs": str(self.outputs_path),
                "agent_comms": str(self.comms_path),
            },
        }
        self.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def log_agent_output(self, *, agent: str, stage: str, output: Any) -> None:
        append_jsonl(
            self.outputs_path,
            {
                "agent": agent,
                "stage": stage,
                "output": _safe_obj(output),
            },
        )

    def log_comm(self, *, sender: str, receiver: str, kind: str, prompt: Any) -> None:
        append_jsonl(
            self.comms_path,
            {
                "sender": sender,
                "receiver": receiver,
                "kind": kind,
                "prompt": _safe_obj(prompt),
            },
        )


def maybe_log_prompt_event(sender: str, receiver: str, kind: str, prompt: Any) -> None:
    path = os.getenv("MEDAGENT_COMM_LOG_PATH", "").strip()
    if not path:
        return
    append_jsonl(
        Path(path),
        {
            "sender": sender,
            "receiver": receiver,
            "kind": kind,
            "prompt": _safe_obj(prompt),
        },
    )
