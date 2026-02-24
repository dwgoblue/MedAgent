from __future__ import annotations

import re
from typing import Any


_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
_WS_RE = re.compile(r"\s+")

_SYNONYM_MAP = {
    "mi": "myocardial infarction",
    "cad": "coronary artery disease",
    "htn": "hypertension",
    "copd": "chronic obstructive pulmonary disease",
}


def normalize_label(text: str | None) -> str:
    if not text:
        return ""
    s = text.lower().strip()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return _SYNONYM_MAP.get(s, s)


def _encounter_key(enc: dict[str, Any]) -> str:
    return str(enc.get("id") or enc.get("encounter_id") or "").strip()


def _condition_display(cond: dict[str, Any]) -> str:
    return str(cond.get("display") or cond.get("text") or cond.get("code") or "").strip()


def _condition_encounter_ref(cond: dict[str, Any]) -> str:
    ref = cond.get("encounter")
    if isinstance(ref, dict):
        return str(ref.get("reference") or ref.get("id") or "").strip()
    if isinstance(ref, str):
        return ref.strip()
    return ""


def extract_primary_encounter_dx(patient: Any) -> str | None:
    encounters = patient.get_encounters() if hasattr(patient, "get_encounters") else []
    conditions = patient.get_conditions() if hasattr(patient, "get_conditions") else []
    if not conditions:
        return None

    primary = encounters[0] if encounters else {}
    encounter_id = _encounter_key(primary)
    if encounter_id:
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            display = _condition_display(cond)
            ref = _condition_encounter_ref(cond).lower()
            if display and encounter_id.lower() in ref:
                return display

    for cond in conditions:
        if not isinstance(cond, dict):
            continue
        display = _condition_display(cond)
        if display:
            return display
    return None
