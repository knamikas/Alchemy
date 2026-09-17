"""Stage-ordering guards on ``coordination.contact_record.Candidate``.

Every guard raises ``RuntimeError``, which ``worker.stages`` downgrades to a
partial entry rather than a crash, so a guard that stopped firing would turn a
programming error into silently missing bond rows. These tests pin each guard
and the ordering pre-checks that make the stage sequence explicit.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from helpers import StructureBuilder

from codes import (
    CandidateSource,
    ContactScope,
    EligibilityReason,
    EligibilityStatus,
    InferredDonorRule,
    MultiDonorStatus,
    ReferenceKind,
)
from coordination.contact_record import (
    Candidate,
    DeclaredConnectionRecord,
    DonorPolicy,
    EligibilityResult,
    GeometryResult,
    MultiDonorResult,
)
from structure_analysis import AtomSite, ContactImage, load_structure


def _neighbor(tmp_path: Path) -> AtomSite:
    """One deposited water oxygen to hang the stage results on."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (2.09, 0.0, 0.0), chain="B")
    context = load_structure("test", builder.write_pdb(tmp_path / "stages.pdb"))
    water = next(residue for residue in context.residues if residue.is_water)
    return water.contact_atoms[0]


def _candidate(neighbor: AtomSite) -> Candidate:
    """An unannotated candidate carrying inert discovery values."""
    image = ContactImage(
        distance=2.09,
        position=(2.09, 0.0, 0.0),
        crystallographic_contact=False,
        strict_ncs_contact=False,
        strict_ncs_operation_id="",
        scope=ContactScope.EXPLICIT,
        image_index=0,
        symmetry_operation="1_555",
        translation=(0, 0, 0),
    )
    return Candidate(
        neighbor=neighbor,
        image=image,
        candidate_sources={CandidateSource.PROXIMITY_4A},
    )


def _declaration(connection_id: str = "metalc1") -> DeclaredConnectionRecord:
    """One inert declaration record to hang candidate provenance on."""
    return DeclaredConnectionRecord(
        source=CandidateSource.STRUCT_CONN,
        connection_id=connection_id,
        connection_type="metalc",
        connection_link_id="",
        connection_asu="same",
        connection_reported_distance=2.09,
    )


def _donor_policy() -> DonorPolicy:
    return DonorPolicy(
        inferred_allowed=True,
        rule=InferredDonorRule.WATER_OXYGEN,
        override="",
    )


def _eligibility() -> EligibilityResult:
    return EligibilityResult(
        status=EligibilityStatus.FIRST_SPHERE_ELIGIBLE,
        reason=EligibilityReason.DISTANCE_WITHIN_TOLERANCE,
        first_sphere_eligible=True,
        inferred_contact_eligible=True,
        assignment_target=2.09,
        assignment_tolerance=0.75,
        first_sphere_cutoff=2.84,
        assignment_reference_kind=ReferenceKind.EXACT,
        assignment_reference="ZN-HOH-O",
    )


def _geometry() -> GeometryResult:
    return GeometryResult(
        distance=2.09,
        literature_distance=2.09,
        literature_stdev=0.05,
        zscore=0.0,
        reference_covered=True,
        outlier=False,
        score_eligible=True,
        score_exclusion_reason="",
    )


def _multi_donor() -> MultiDonorResult:
    return MultiDonorResult(
        detected=False,
        contact_count=1,
        geometry_status=MultiDonorStatus.SINGLE_DONOR,
        contains_suspect_bond=False,
    )


def _annotated(neighbor: AtomSite) -> Candidate:
    """A candidate carrying every stage result in pipeline order."""
    candidate = _candidate(neighbor)
    candidate.set_donor_policy(_donor_policy())
    candidate.set_eligibility(_eligibility())
    candidate.set_geometry(_geometry())
    candidate.set_multi_donor(_multi_donor())
    return candidate


def test_donor_policy_cannot_be_set_twice(tmp_path: Path) -> None:
    """A second donor-policy verdict is a programming error, not an update."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())

    with pytest.raises(RuntimeError, match="donor policy was already evaluated"):
        candidate.set_donor_policy(_donor_policy())


def test_eligibility_cannot_be_set_twice(tmp_path: Path) -> None:
    """A second eligibility verdict is rejected."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())
    candidate.set_eligibility(_eligibility())

    with pytest.raises(RuntimeError, match="eligibility was already evaluated"):
        candidate.set_eligibility(_eligibility())


def test_geometry_cannot_be_set_twice(tmp_path: Path) -> None:
    """A second geometry verdict is rejected."""
    candidate = _annotated(_neighbor(tmp_path))

    with pytest.raises(RuntimeError, match="geometry was already evaluated"):
        candidate.set_geometry(_geometry())


def test_multi_donor_cannot_be_set_twice(tmp_path: Path) -> None:
    """A second multi-donor verdict is rejected."""
    candidate = _annotated(_neighbor(tmp_path))

    with pytest.raises(RuntimeError, match="multi-donor status was already evaluated"):
        candidate.set_multi_donor(_multi_donor())


def test_donor_policy_before_evaluation_raises(tmp_path: Path) -> None:
    """Reading an unset donor policy raises instead of returning ``None``."""
    candidate = _candidate(_neighbor(tmp_path))

    with pytest.raises(RuntimeError, match="donor policy has not been evaluated"):
        candidate.donor_policy()


def test_eligibility_before_evaluation_raises(tmp_path: Path) -> None:
    """Reading an unset eligibility result raises."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())

    with pytest.raises(RuntimeError, match="eligibility has not been evaluated"):
        candidate.eligibility()


def test_geometry_before_evaluation_raises(tmp_path: Path) -> None:
    """Reading an unset geometry result raises."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())
    candidate.set_eligibility(_eligibility())

    with pytest.raises(RuntimeError, match="geometry has not been evaluated"):
        candidate.geometry()


def test_multi_donor_before_evaluation_raises(tmp_path: Path) -> None:
    """Reading an unset multi-donor result raises."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())
    candidate.set_eligibility(_eligibility())
    candidate.set_geometry(_geometry())

    with pytest.raises(RuntimeError, match="multi-donor status has not been evaluated"):
        candidate.multi_donor()


def test_eligibility_requires_donor_policy_first(tmp_path: Path) -> None:
    """Eligibility reads the donor policy, so it may not run before it."""
    candidate = _candidate(_neighbor(tmp_path))

    with pytest.raises(RuntimeError, match="donor policy has not been evaluated"):
        candidate.set_eligibility(_eligibility())


def test_geometry_requires_eligibility_first(tmp_path: Path) -> None:
    """Geometry is only attached to candidates eligibility already assessed."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())

    with pytest.raises(RuntimeError, match="eligibility has not been evaluated"):
        candidate.set_geometry(_geometry())


def test_multi_donor_requires_geometry_first(tmp_path: Path) -> None:
    """Multi-donor grouping reads each member's geometry, so it runs after it."""
    candidate = _candidate(_neighbor(tmp_path))
    candidate.set_donor_policy(_donor_policy())
    candidate.set_eligibility(_eligibility())

    with pytest.raises(RuntimeError, match="geometry has not been evaluated"):
        candidate.set_multi_donor(_multi_donor())


def test_with_provenance_copies_a_candidate_before_annotation(tmp_path: Path) -> None:
    """``with_provenance`` copies discovery state and owns its new collections.

    ``candidates.merge_candidates`` uses it in place of a bare
    ``dataclasses.replace``, whose ``init=False`` stage fields would be reset
    rather than copied.
    """
    candidate = _candidate(_neighbor(tmp_path))
    candidate.declared_connections.append(_declaration())

    copy = candidate.with_provenance(
        candidate_sources=set(candidate.candidate_sources),
        declared_connections=list(candidate.declared_connections),
    )

    assert copy is not candidate
    assert copy.neighbor is candidate.neighbor
    assert copy.image is candidate.image
    assert copy.donor_class_supported == candidate.donor_class_supported
    assert copy.candidate_sources == candidate.candidate_sources
    assert copy.declared_connections == candidate.declared_connections
    # The copy owns its collections: merging into it cannot reach the original.
    assert copy.candidate_sources is not candidate.candidate_sources
    assert copy.declared_connections is not candidate.declared_connections
    copy.candidate_sources.add(CandidateSource.STRUCT_CONN)
    copy.declared_connections.append(_declaration("second"))
    assert candidate.candidate_sources == {CandidateSource.PROXIMITY_4A}
    assert len(candidate.declared_connections) == 1
    # The copy is unannotated and can still run the stages itself.
    copy.set_donor_policy(_donor_policy())
    assert copy.donor_policy().inferred_allowed is True


@pytest.mark.parametrize("stage_count", [1, 2, 3, 4])
def test_with_provenance_refuses_an_annotated_candidate(
    tmp_path: Path, stage_count: int
) -> None:
    """Copying after any stage ran would drop that stage's verdict, so it raises.

    The ``init=False`` stage fields cannot cross ``__init__``; refusing the
    copy turns "merge provenance before annotation" from a comment into an
    enforced invariant.
    """
    candidate = _candidate(_neighbor(tmp_path))
    if stage_count >= 1:
        candidate.set_donor_policy(_donor_policy())
    if stage_count >= 2:
        candidate.set_eligibility(_eligibility())
    if stage_count >= 3:
        candidate.set_geometry(_geometry())
    if stage_count >= 4:
        candidate.set_multi_donor(_multi_donor())

    with pytest.raises(RuntimeError, match="cannot be copied after an annotation"):
        candidate.with_provenance(
            candidate_sources=set(candidate.candidate_sources),
            declared_connections=list(candidate.declared_connections),
        )

    # The refusal leaves the original untouched.
    assert candidate.donor_policy().rule == "water_oxygen"


def test_replace_still_discards_every_stage_result(tmp_path: Path) -> None:
    """``dataclasses.replace`` silently resets the ``init=False`` stage fields.

    Nothing in ``src`` calls it on a ``Candidate`` any more, but it stays a
    one-line reach for a future caller, so the loss it causes is pinned here
    next to the guarded copy that replaced it.
    """
    candidate = _annotated(_neighbor(tmp_path))

    copy = dataclasses.replace(candidate)

    assert copy.neighbor is candidate.neighbor
    assert copy.candidate_sources == candidate.candidate_sources
    for accessor in (
        copy.donor_policy,
        copy.eligibility,
        copy.geometry,
        copy.multi_donor,
    ):
        with pytest.raises(RuntimeError, match="has not been evaluated"):
            accessor()
    # The original keeps its verdicts; only the copy lost them.
    assert candidate.multi_donor().contact_count == 1
