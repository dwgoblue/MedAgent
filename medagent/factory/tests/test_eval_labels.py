from __future__ import annotations

from medagent.runtime.harness.eval_labels import extract_primary_encounter_dx, normalize_label


class _FakePatient:
    def __init__(self, encounters, conditions) -> None:
        self._encounters = encounters
        self._conditions = conditions

    def get_encounters(self):
        return self._encounters

    def get_conditions(self):
        return self._conditions


def test_normalize_label_synonym() -> None:
    assert normalize_label("  HTN  ") == "hypertension"
    assert normalize_label("Coronary-Artery Disease!") == "coronary artery disease"


def test_extract_primary_encounter_dx_with_ref() -> None:
    p = _FakePatient(
        encounters=[{"id": "enc-1"}],
        conditions=[
            {"display": "Migraine"},
            {"display": "Pulmonary embolism", "encounter": {"reference": "Encounter/enc-1"}},
        ],
    )
    assert extract_primary_encounter_dx(p) == "Pulmonary embolism"


def test_extract_primary_encounter_dx_fallback_first_condition() -> None:
    p = _FakePatient(
        encounters=[{"id": "enc-2"}],
        conditions=[{"display": "Hypertension"}],
    )
    assert extract_primary_encounter_dx(p) == "Hypertension"
