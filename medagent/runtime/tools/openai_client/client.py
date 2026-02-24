from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from medagent.runtime.core.run_logger import maybe_log_prompt_event


@dataclass(frozen=True)
class VerificationDecision:
    status: str
    resolution: str
    evidence_ids: list[str]


class OpenAIReasoner:
    def __init__(self, model: str = "gpt-5.2") -> None:
        self.model = model
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is not available") from exc
        self._client = OpenAI(max_retries=0)

    def verify_claim(
        self,
        claim_text: str,
        category: str,
        evidence_requirements: list[str],
        snippets: list[dict[str, str]],
    ) -> VerificationDecision:
        system_msg = (
            "You are a medical evidence verifier. "
            "Return strict JSON with keys: status, resolution, evidence_ids. "
            "status must be one of: pass, weak, fail. "
            "Do not provide prescribing or dosing instructions."
        )

        payload = {
            "claim_text": claim_text,
            "category": category,
            "evidence_requirements": evidence_requirements,
            "snippets": snippets,
        }
        maybe_log_prompt_event(
            sender="biomcp_verifier_v1",
            receiver=f"openai:{self.model}",
            kind="claim_verification_prompt",
            prompt=payload,
        )

        resp = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Assess support for this claim using provided snippets only. "
                        "If support is incomplete, use weak or fail.\n"
                        f"Input JSON:\n{json.dumps(payload, ensure_ascii=True)}"
                    ),
                },
            ],
        )

        raw = getattr(resp, "output_text", "") or ""
        parsed = self._safe_parse(raw)
        status = parsed.get("status", "weak")
        if status not in {"pass", "weak", "fail"}:
            status = "weak"

        evidence_ids = parsed.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []

        resolution = str(parsed.get("resolution", "Model-based verification result"))
        return VerificationDecision(status=status, resolution=resolution, evidence_ids=evidence_ids)

    @staticmethod
    def _safe_parse(raw: str) -> dict[str, Any]:
        raw = raw.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    return {}
        return {}
