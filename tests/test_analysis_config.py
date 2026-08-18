from __future__ import annotations

import pytest

import analysis_config


def _identity(
    reference_data_id: str = "reference-a",
) -> str:
    return analysis_config.analysis_config_id(
        reference_data_id=reference_data_id,
    )


def test_analysis_configuration_identity_is_deterministic() -> None:
    assert _identity() == _identity()
    assert _identity().startswith("alchemy-analysis-config-")


def test_execution_choices_are_not_part_of_the_payload() -> None:
    payload = analysis_config.analysis_config_payload(reference_data_id="reference-a")
    assert "bonds_enabled" not in payload
    assert "density_map_scope" not in payload


def test_reference_data_changes_the_identity() -> None:
    baseline = _identity()
    assert _identity(reference_data_id="reference-b") != baseline


def test_policy_changes_change_the_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _identity()
    monkeypatch.setattr(analysis_config, "MODEL_POLICY", "all")
    assert _identity() != baseline


@pytest.mark.parametrize(
    ("recorded", "expected"),
    [
        ([], True),
        (["current"], True),
        (["current", "current"], True),
        (["old"], False),
        (["current", "old"], False),
        ([""], False),
    ],
)
def test_analysis_configuration_compatibility(
    recorded: list[str], expected: bool
) -> None:
    assert (
        analysis_config.analysis_configs_are_compatible(recorded, "current") is expected
    )
