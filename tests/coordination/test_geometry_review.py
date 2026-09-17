"""Cutoff edges, non-finite inputs and mixed groups in ``coordination.geometry``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from helpers import StructureBuilder, approx

import codes
from codes import EligibilityReason, EligibilityStatus, ReferenceKind
from coordination.contact_record import Candidate, DonorPolicy, EligibilityResult
from coordination.geometry import (
    annotate_contacts,
    annotate_multi_donor_groups,
    residue_image_key,
    zscore,
)
from coordination.policy import FIRST_SPHERE_TOLERANCE, ZSCORE_OUTLIER_CUTOFF
from structure_analysis import AtomSite, ContactImage, load_structure

# DPI 0.12 against the Zn-water spread of 0.05 gives a denominator of exactly
# 0.13, so a target z-score maps to a hand-computable distance.
WATER_DPI = 0.12
WATER_MU = 2.09
WATER_DENOMINATOR = 0.13


def _contact(
    neighbor: AtomSite, *, distance: float = 0.0, image: ContactImage | None = None
) -> Candidate:
    """A ``Candidate`` whose untouched discovery fields carry inert values."""
    if image is None:
        image = ContactImage(
            distance=distance,
            position=(0.0, 0.0, 0.0),
            crystallographic_contact=False,
            strict_ncs_contact=False,
            strict_ncs_operation_id="",
            scope=codes.ContactScope.EXPLICIT,
            image_index=0,
            symmetry_operation="1_555",
            translation=(0, 0, 0),
        )
    candidate = Candidate(
        neighbor=neighbor,
        image=image,
        candidate_sources={codes.CandidateSource.PROXIMITY_4A},
    )
    # ``set_geometry`` requires the eligibility stage to have run, so give the
    # candidate the inert verdicts the real pipeline would have attached.
    candidate.set_donor_policy(
        DonorPolicy(
            inferred_allowed=True,
            rule=codes.InferredDonorRule.TYPICAL_SIDECHAIN_DONOR,
            override="",
        )
    )
    candidate.set_eligibility(
        EligibilityResult(
            status=EligibilityStatus.FIRST_SPHERE_ELIGIBLE,
            reason=EligibilityReason.DISTANCE_WITHIN_TOLERANCE,
            first_sphere_eligible=True,
            inferred_contact_eligible=True,
            assignment_target=WATER_MU,
            assignment_tolerance=FIRST_SPHERE_TOLERANCE,
            first_sphere_cutoff=WATER_MU + FIRST_SPHERE_TOLERANCE,
            assignment_reference_kind=ReferenceKind.EXACT,
            assignment_reference="ZN-HOH-O",
        )
    )
    return candidate


def _water_atom(tmp_path: Path, name: str) -> AtomSite:
    """The single water oxygen of a one-metal, one-water structure."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (WATER_MU, 0.0, 0.0), chain="B")
    context = load_structure("test", builder.write_pdb(tmp_path / name))
    water = next(residue for residue in context.residues if residue.is_water)
    return water.contact_atoms[0]


def _methionine_donors(tmp_path: Path, name: str) -> tuple[AtomSite, AtomSite]:
    """One methionine's backbone O and SD: the same residue image, one group.

    The backbone carbonyl is keyed ``("CA", "O", "ZN")`` and covered by the
    bundled reference; no ``("MET", "S", "ZN")`` distance is bundled, so SD
    can never be scored.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid(
        "MET",
        10,
        chain="A",
        positions={"O": (2.1, 0.0, 0.0), "SD": (0.0, 2.4, 0.0)},
    )
    context = load_structure("test", builder.write_pdb(tmp_path / name))
    residue = next(
        residue for residue in context.residues if residue.residue_name == "MET"
    )
    atoms = {atom.atom_name: atom for atom in residue.contact_atoms}
    return atoms["O"], atoms["SD"]


@pytest.mark.parametrize("dist", [float("inf"), float("-inf"), float("nan")])
def test_zscore_rejects_a_non_finite_distance(dist: float) -> None:
    """A non-finite distance is a missing input, not a huge z-score."""
    assert math.isnan(zscore(dist, WATER_MU, 0.05, WATER_DPI))


@pytest.mark.parametrize("distance", [float("inf"), float("nan")])
def test_annotate_contacts_leaves_a_non_finite_distance_unassessed(
    tmp_path: Path, distance: float
) -> None:
    """A non-finite measured distance yields NaN Zbond and blank verdicts."""
    contact = _contact(_water_atom(tmp_path, "nonfinite.pdb"), distance=distance)

    annotate_contacts([contact], "ZN", WATER_DPI)

    geometry = contact.geometry()
    assert geometry.reference_covered is True
    assert math.isnan(geometry.zscore)
    assert geometry.outlier is None
    assert geometry.consistent is None
    assert geometry.score_eligible is False
    assert (
        geometry.score_exclusion_reason == codes.ScoreExclusionReason.ZSCORE_UNAVAILABLE
    )


@pytest.mark.parametrize(
    ("raw_zscore", "outlier"),
    [
        (6.0, True),  # the cutoff itself is inclusive
        (-6.0, True),
        (6.0 - 1e-13, False),  # a hair inside stays consistent
        (-(6.0 - 1e-13), False),
    ],
)
def test_the_cutoff_is_a_plain_inclusive_comparison(
    tmp_path: Path, raw_zscore: float, outlier: bool
) -> None:
    """No tolerance widens the cutoff: ``|Zbond| >= 6`` decides alone.

    The distances are built from the exact 0.13 denominator, so each case
    lands within one ulp of the requested z-score.
    """
    distance = WATER_MU + raw_zscore * WATER_DENOMINATOR
    contact = _contact(_water_atom(tmp_path, "cutoff.pdb"), distance=distance)

    annotate_contacts([contact], "ZN", WATER_DPI)

    geometry = contact.geometry()
    magnitude = abs(zscore(distance, WATER_MU, 0.05, WATER_DPI))
    assert ZSCORE_OUTLIER_CUTOFF == 6.0
    if outlier:
        assert magnitude >= ZSCORE_OUTLIER_CUTOFF
    else:
        # Inside the cutoff, yet close enough that an absolute tolerance of
        # 1e-12 around it would have called this an outlier.
        assert ZSCORE_OUTLIER_CUTOFF - 1e-12 < magnitude < ZSCORE_OUTLIER_CUTOFF
    assert geometry.zscore == approx(round(raw_zscore, 4))
    assert geometry.outlier is outlier
    assert geometry.consistent is (not outlier)


def test_residue_image_key_is_the_residue_and_its_symmetry_image(
    tmp_path: Path,
) -> None:
    """The key is the deposited residue plus the image it was generated in.

    Scope, NCS operation id and symmetry operation all follow from the image
    index, so the key names the index and the cell translation only.
    """
    neighbor = _water_atom(tmp_path, "key.pdb")
    image = ContactImage(
        distance=2.1,
        position=(1.0, 2.0, 3.0),
        crystallographic_contact=True,
        strict_ncs_contact=True,
        strict_ncs_operation_id="ncs-2",
        scope=codes.ContactScope.STRICT_NCS_AND_CRYSTALLOGRAPHIC,
        image_index=4,
        symmetry_operation="3_565",
        translation=(0, 1, 0),
    )

    assert residue_image_key(_contact(neighbor, image=image)) == (
        neighbor.residue_key,
        4,
        (0, 1, 0),
    )


def test_an_outlier_beside_an_unscorable_donor_marks_the_group_suspect(
    tmp_path: Path,
) -> None:
    """One outlier decides ``suspect`` even when a sibling has no z-score."""
    backbone, sulfur = _methionine_donors(tmp_path, "suspect.pdb")
    # 3.6 A against the 2.07 +/- 0.20 backbone reference is far past the cutoff.
    outlier_contact = _contact(backbone, distance=3.6)
    unscorable = _contact(sulfur, distance=2.4)
    contacts = [outlier_contact, unscorable]

    annotate_contacts(contacts, "ZN", WATER_DPI)
    annotate_multi_donor_groups(contacts)

    assert outlier_contact.geometry().outlier is True
    assert math.isnan(unscorable.geometry().zscore)
    for contact in contacts:
        group = contact.multi_donor()
        assert group.detected is True
        assert group.contact_count == 2
        assert group.geometry_status is codes.MultiDonorStatus.SUSPECT
        assert group.contains_suspect_bond is True
    # Group status never excludes a bond; only the unscorable member drops out.
    assert outlier_contact.geometry().score_eligible is True
    assert outlier_contact.geometry().score_exclusion_reason == ""
    assert unscorable.geometry().score_eligible is False
    assert (
        unscorable.geometry().score_exclusion_reason
        == codes.ScoreExclusionReason.ZSCORE_UNAVAILABLE
    )


def test_a_consistent_bond_beside_an_unscorable_donor_is_indeterminate(
    tmp_path: Path,
) -> None:
    """Without an outlier, one unscorable member makes the group indeterminate."""
    backbone, sulfur = _methionine_donors(tmp_path, "indeterminate.pdb")
    consistent_contact = _contact(backbone, distance=2.1)
    unscorable = _contact(sulfur, distance=2.4)
    contacts = [consistent_contact, unscorable]

    annotate_contacts(contacts, "ZN", WATER_DPI)
    annotate_multi_donor_groups(contacts)

    assert consistent_contact.geometry().consistent is True
    for contact in contacts:
        group = contact.multi_donor()
        assert group.detected is True
        assert group.contact_count == 2
        assert group.geometry_status is codes.MultiDonorStatus.INDETERMINATE
        assert group.contains_suspect_bond is False
    # The scored member keeps contributing to aggregate scoring.
    assert consistent_contact.geometry().score_eligible is True
    assert consistent_contact.geometry().score_exclusion_reason == ""
    assert unscorable.geometry().score_eligible is False
    assert (
        unscorable.geometry().score_exclusion_reason
        == codes.ScoreExclusionReason.ZSCORE_UNAVAILABLE
    )
