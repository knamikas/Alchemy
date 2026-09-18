"""Review coverage for ``coordination.eligibility``.

These tests pin the behaviour of the reference-key lookup, the first-sphere
cutoff clamp, and the promotion rules that decide which declared candidates may
become bond rows and which may report missing assignment evidence.
"""

from __future__ import annotations

import math
from pathlib import Path

import helpers
from helpers import AtomSpec, StructureBuilder, approx

from codes import (
    CandidateSource,
    ContactScope,
    DonorRuleOverride,
    EligibilityReason,
    EligibilityStatus,
    ReasonCode,
    ReferenceKind,
)
from coordination.contact_record import Candidate, DeclaredConnectionRecord
from coordination.eligibility import (
    annotate_donor_policy,
    current_contacts_from_candidates,
    first_sphere_rule,
)
from coordination.metal_distances import distances as distance_reference
from coordination.policy import CANDIDATE_SEARCH_RADIUS, FIRST_SPHERE_TOLERANCE
from structure_analysis import (
    AtomSite,
    ContactImage,
    StructureContext,
    load_structure,
)

Vec3 = tuple[float, float, float]


def _only_candidate(result: helpers.BondAnalysis, atom_name: str) -> dict[str, object]:
    """The single candidate row whose neighbour atom is ``atom_name``."""
    matches = [
        row for row in result.candidate_rows if row["neighbor_atom"] == atom_name
    ]
    assert len(matches) == 1, result.candidate_rows
    return dict(matches[0])


def _atom(context: StructureContext, atom_name: str) -> AtomSite:
    """The single selected atom named ``atom_name`` in a synthetic structure."""
    matches = [atom for atom in context.contact_atoms if atom.atom_name == atom_name]
    assert len(matches) == 1, [atom.atom_name for atom in context.contact_atoms]
    return matches[0]


def _image(distance: float, position: Vec3) -> ContactImage:
    """An explicit contact image at ``distance`` from the metal."""
    return ContactImage(
        distance=distance,
        position=position,
        crystallographic_contact=False,
        strict_ncs_contact=False,
        strict_ncs_operation_id="",
        scope=ContactScope.EXPLICIT,
        image_index=0,
        symmetry_operation="1_555",
        translation=(0, 0, 0),
    )


def _declaration(connection_id: str, distance: float) -> DeclaredConnectionRecord:
    """One source declaration record, as the declaration stage would build it."""
    return DeclaredConnectionRecord(
        source=CandidateSource.LINK,
        connection_id=connection_id,
        connection_type="metalc",
        connection_link_id="",
        connection_asu="same",
        connection_reported_distance=distance,
    )


def _candidate(
    neighbor: AtomSite,
    *,
    distance: float,
    position: Vec3,
    declared: str = "",
    supported: bool = True,
) -> Candidate:
    """A discovery-stage candidate, optionally carrying a declaration."""
    return Candidate(
        neighbor=neighbor,
        image=_image(distance, position),
        candidate_sources=(
            {CandidateSource.PROXIMITY_4A, CandidateSource.LINK}
            if declared
            else {CandidateSource.PROXIMITY_4A}
        ),
        declared_connections=[_declaration(declared, distance)] if declared else [],
        donor_class_supported=supported,
    )


def _metal_and_donor(
    tmp_path: Path,
    *,
    metal_element: str,
    resname: str,
    positions: dict[str, Vec3],
    name: str,
) -> tuple[StructureContext, AtomSite]:
    """Load a one-metal, one-residue probe structure and return it with its metal."""
    builder = StructureBuilder()
    builder.add_metal(metal_element, 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_amino_acid(resname, 10, chain="A", positions=positions)
    path = builder.write_pdb(tmp_path / name)
    context = load_structure("test", path)
    return context, _atom(context, metal_element)


def test_no_reference_target_defines_a_sphere_wider_than_the_candidate_search() -> None:
    """Verify the ``CANDIDATE_SEARCH_RADIUS`` clamp in ``first_sphere_rule`` is inert.

    The cutoff is ``min(CANDIDATE_SEARCH_RADIUS, target + FIRST_SPHERE_TOLERANCE)``.
    If a reference row ever made that clamp bind, the first sphere would reach
    past the radius the candidate search covers, so contacts inside the sphere
    would never be discovered. Widest bundled target: K-O at 2.82 A.
    """
    targets = distance_reference.first_sphere_targets()
    assert targets, "the bundled reference table defines no first-sphere targets"
    over_radius = sorted(
        f"{metal}-{donor}: {target}"
        for (metal, donor), target in targets.items()
        if target + FIRST_SPHERE_TOLERANCE > CANDIDATE_SEARCH_RADIUS
    )
    assert over_radius == [], (
        "reference targets whose first sphere reaches past the candidate "
        f"search radius: {over_radius}"
    )
    exact = sorted(
        f"{residue}-{donor}-{metal}: {mu}"
        for (residue, donor, metal), (
            mu,
            _sd,
        ) in distance_reference.literature_distances().items()
        if mu + FIRST_SPHERE_TOLERANCE > CANDIDATE_SEARCH_RADIUS
    )
    assert exact == []
    assert max(targets.values()) + FIRST_SPHERE_TOLERANCE < CANDIDATE_SEARCH_RADIUS


def test_a_non_oxygen_atom_in_a_water_residue_gets_no_water_oxygen_reference() -> None:
    """Only the oxygen of an HOH residue is the bundled metal-water donor.

    A source LINK can name any atom modeled inside a water residue. Keying such
    an atom as ``HOH:O`` would publish an exact reference, and a z-score, for a
    distance the Zn-water row never described.
    """
    metal = helpers.atom_site("ZN", atom_name="ZN", residue_name="ZN")
    oxygen = helpers.atom_site("O", atom_name="O", residue_name="HOH", is_water=True)
    water_target, _cutoff, kind, key = first_sphere_rule(metal, oxygen)
    assert kind is ReferenceKind.EXACT
    assert key == "HOH:O:ZN"

    nitrogen = helpers.atom_site("N", atom_name="N1", residue_name="HOH", is_water=True)
    target, cutoff, kind, key = first_sphere_rule(metal, nitrogen)
    assert kind is ReferenceKind.ELEMENT_FALLBACK
    assert key == "*:N:ZN"
    assert target == approx(distance_reference.first_sphere_targets()[("ZN", "N")])
    assert target != approx(water_target)
    assert cutoff == approx(target + FIRST_SPHERE_TOLERANCE)


def test_a_water_atom_with_no_element_reference_reports_a_missing_reference() -> None:
    """The fall-through reaches the missing-reference path, not a water target."""
    potassium = helpers.atom_site("K", atom_name="K", residue_name="K")
    sulfur = helpers.atom_site("S", atom_name="S", residue_name="HOH", is_water=True)

    target, cutoff, kind, key = first_sphere_rule(potassium, sulfur)

    assert kind is ReferenceKind.MISSING
    assert key == ""
    assert math.isnan(target) and math.isnan(cutoff)


def test_the_reference_key_normalizes_the_deposited_residue_name() -> None:
    """``bonding_key`` matches the donor rule's ``.strip().upper()`` normalization.

    The donor rule uppercases the residue name before consulting the donor
    table, so a padded or lower-case deposited name must not send the reference
    lookup down the element fallback while the donor rule calls the same atom a
    typical donor.
    """
    metal = helpers.atom_site("ZN", atom_name="ZN", residue_name="ZN")
    canonical = helpers.atom_site("N", atom_name="NE2", residue_name="HIS")
    deposited = helpers.atom_site("N", atom_name="NE2", residue_name=" his ")

    assert first_sphere_rule(metal, deposited) == first_sphere_rule(metal, canonical)
    assert first_sphere_rule(metal, deposited)[2] is ReferenceKind.EXACT
    assert first_sphere_rule(metal, deposited)[3] == "HIS:N:ZN"


def test_a_declared_non_oxygen_water_atom_is_published_without_an_exact_reference(
    tmp_path: Path,
) -> None:
    """End to end: a LINK to a nitrogen in an HOH residue keeps the element fallback.

    The contact is still promoted -- water is a supported donor class -- but it
    carries the ``*:N:ZN`` membership target and no z-score, because no exact
    reference covers it.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    water = builder.add_hetero_residue(
        "HOH", 20, [AtomSpec("N1", "N", (2.10, 0.0, 0.0))], chain="B"
    )
    builder.add_connection(
        metal.ref("ZN"), water.ref("N1"), name="link1", reported_distance=2.10
    )
    path = builder.write_pdb(tmp_path / "water_nitrogen.pdb")

    result = helpers.analyze_bonds(path, connection_path=path)

    candidate = _only_candidate(result, "N1")
    assert candidate["neighbor_resname"] == "HOH"
    assert candidate["declared_connection"] is True
    assert candidate["assignment_reference_kind"] == ReferenceKind.ELEMENT_FALLBACK
    assert candidate["assignment_reference"] == "*:N:ZN"
    assert candidate["assignment_target"] == approx(
        distance_reference.first_sphere_targets()[("ZN", "N")]
    )
    assert candidate["assignment_target"] != approx(
        distance_reference.literature_distances()[("HOH", "O", "ZN")][0]
    )
    assert candidate["inferred_donor_allowed"] is False
    assert candidate["donor_rule_override"] == DonorRuleOverride.DECLARED_CONNECTION

    rows = result.rows_for("N1")
    assert len(rows) == 1
    assert rows[0]["reference_covered"] is False
    assert math.isnan(float(rows[0]["zscore"]))


def test_a_declared_donor_outside_the_supported_classes_reports_no_missing_reference(
    tmp_path: Path,
) -> None:
    """An unsupported declared donor never becomes a bond, so it demands no reference.

    The candidate cannot be promoted by ``current_contacts_from_candidates``,
    so forcing ``missing_assignment_reference`` would make the entry partial
    over a row that could never have used the reference.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("K", 1, chain="B", pos=(0.0, 0.0, 0.0))
    nucleotide = builder.add_hetero_residue(
        "DG", 10, [AtomSpec("N7", "N", (2.90, 0.0, 0.0))], chain="C"
    )
    builder.add_connection(
        metal.ref("K"), nucleotide.ref("N7"), name="metalc1", reported_distance=2.90
    )
    path = builder.write_cif(tmp_path / "k_nucleotide.cif")

    result = helpers.analyze_bonds(path, connection_path=path)

    candidate = _only_candidate(result, "N7")
    assert candidate["eligibility_status"] == EligibilityStatus.NON_TYPICAL_DONOR
    assert candidate["eligibility_reason"] == EligibilityReason.ATOM_NOT_TYPICAL_DONOR
    assert candidate["declared_connection"] is True
    assert result.bond_rows == []
    assert (
        ReasonCode.MISSING_FIRST_SPHERE_REFERENCE
        not in result.metadata.partial_reason_codes
    )
    assert not any("K-N" in message for message in result.metadata.messages)


def test_a_declared_donor_in_a_supported_class_still_reports_a_missing_reference(
    tmp_path: Path,
) -> None:
    """A supported declared donor can be promoted, so a missing reference is partial.

    ASN ND2 is not geometry-inferable, but the declaration admits it and the
    class is one Alchemy assesses; without a K-N reference the entry must say
    the assignment evidence is incomplete.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("K", 1, chain="B", pos=(0.0, 0.0, 0.0))
    residue = builder.add_amino_acid(
        "ASN", 10, chain="A", positions={"ND2": (2.90, 0.0, 0.0)}
    )
    builder.add_connection(
        metal.ref("K"), residue.ref("ND2"), name="metalc1", reported_distance=2.90
    )
    path = builder.write_cif(tmp_path / "k_asn.cif")

    result = helpers.analyze_bonds(path, connection_path=path)

    candidate = _only_candidate(result, "ND2")
    assert (
        candidate["eligibility_status"]
        == EligibilityStatus.MISSING_ASSIGNMENT_REFERENCE
    )
    assert candidate["eligibility_reason"] == EligibilityReason.NO_ASSIGNMENT_REFERENCE
    assert (
        ReasonCode.MISSING_FIRST_SPHERE_REFERENCE
        in result.metadata.partial_reason_codes
    )
    assert any("K-N" in message for message in result.metadata.messages)


def test_a_declared_image_merges_into_the_eligible_image_it_duplicates(
    tmp_path: Path,
) -> None:
    """Near-coincident images collapse once, across inferred and declared contacts.

    The eligible image and the declared-but-not-inferred image are the same
    deposited atom, 0.79 A apart, so one contact survives and carries both
    provenances. Collapsing the eligible list separately first would be
    redundant, and must not change this result.
    """
    context, metal = _metal_and_donor(
        tmp_path,
        metal_element="ZN",
        resname="HIS",
        positions={"NE2": (2.00, 0.0, 0.0)},
        name="his_images.pdb",
    )
    neighbor = _atom(context, "NE2")
    inferred = _candidate(neighbor, distance=2.00, position=(2.00, 0.0, 0.0))
    # 2.79 A is outside the 2.03 + 0.75 A His-Zn sphere, so only the
    # declaration can admit this image.
    declared = _candidate(
        neighbor, distance=2.79, position=(2.79, 0.0, 0.0), declared="link1"
    )
    annotate_donor_policy(context, [inferred, declared])

    contacts, unsupported_pairs = current_contacts_from_candidates(
        [inferred, declared], metal
    )

    assert unsupported_pairs == set()
    assert inferred.eligibility().status is EligibilityStatus.FIRST_SPHERE_ELIGIBLE
    assert declared.eligibility().status is EligibilityStatus.OUTSIDE_FIRST_SPHERE
    assert contacts == [inferred]
    assert contacts[0].image.distance == approx(2.00)
    assert contacts[0].candidate_sources == {
        CandidateSource.PROXIMITY_4A,
        CandidateSource.LINK,
    }
    assert [record.connection_id for record in contacts[0].declared_connections] == [
        "link1"
    ]


def test_only_a_supported_declared_class_overrides_the_donor_rule(
    tmp_path: Path,
) -> None:
    """``annotate_donor_policy`` and the promotion filter agree on the same gate.

    ASN ND2 is inside the 2.03 + 0.75 A Zn-N sphere but is not a geometry
    donor. A declaration promotes it only when the donor class is one Alchemy
    can assess.
    """
    context, metal = _metal_and_donor(
        tmp_path,
        metal_element="ZN",
        resname="ASN",
        positions={"ND2": (2.20, 0.0, 0.0)},
        name="zn_asn.pdb",
    )
    neighbor = _atom(context, "ND2")

    supported = _candidate(
        neighbor, distance=2.20, position=(2.20, 0.0, 0.0), declared="link1"
    )
    annotate_donor_policy(context, [supported])
    assert supported.donor_policy().inferred_allowed is False
    assert supported.donor_policy().override == DonorRuleOverride.DECLARED_CONNECTION
    contacts, pairs = current_contacts_from_candidates([supported], metal)
    assert contacts == [supported]
    assert pairs == set()
    assert supported.eligibility().status is EligibilityStatus.NON_TYPICAL_DONOR

    unsupported = _candidate(
        neighbor,
        distance=2.20,
        position=(2.20, 0.0, 0.0),
        declared="link1",
        supported=False,
    )
    annotate_donor_policy(context, [unsupported])
    assert unsupported.donor_policy().override == ""
    contacts, pairs = current_contacts_from_candidates([unsupported], metal)
    assert contacts == []
    assert pairs == set()


def test_an_unsupported_declared_class_adds_no_unsupported_pair(
    tmp_path: Path,
) -> None:
    """The reported metal-donor pairs cover only candidates that could be promoted.

    With no K-N reference, a supported declared donor is unassignable and says
    so; the same atom in an unsupported class is merely non-typical, and adds
    nothing to the pairs the entry reports as missing evidence.
    """
    context, metal = _metal_and_donor(
        tmp_path,
        metal_element="K",
        resname="ASN",
        positions={"ND2": (2.90, 0.0, 0.0)},
        name="k_asn_unit.pdb",
    )
    neighbor = _atom(context, "ND2")

    supported = _candidate(
        neighbor, distance=2.90, position=(2.90, 0.0, 0.0), declared="link1"
    )
    annotate_donor_policy(context, [supported])
    contacts, pairs = current_contacts_from_candidates([supported], metal)
    assert (
        supported.eligibility().status is EligibilityStatus.MISSING_ASSIGNMENT_REFERENCE
    )
    assert contacts == [supported]
    assert pairs == {("K", "N")}

    unsupported = _candidate(
        neighbor,
        distance=2.90,
        position=(2.90, 0.0, 0.0),
        declared="link1",
        supported=False,
    )
    annotate_donor_policy(context, [unsupported])
    contacts, pairs = current_contacts_from_candidates([unsupported], metal)
    assert unsupported.eligibility().status is EligibilityStatus.NON_TYPICAL_DONOR
    assert contacts == []
    assert pairs == set()
