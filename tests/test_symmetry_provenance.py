"""Symmetry-image provenance: published coordinates, special positions, scope.

Three things Alchemy publishes about a symmetry-generated contact were never
checked anywhere in the suite, and each of them can be wrong in a way that is
invisible in the columns that *were* checked:

* **The image coordinates.** ``transformed_neighbor_x/y/z`` and
  ``cell_translation_x/y/z`` say *where* the donor actually is. The existing
  symmetry tests assert ``distance`` and ``symmetry_operation`` only, so an
  inverted or mis-composed transform could ship a donor position that
  contradicts its own reported distance and nothing would notice. The binding
  invariant is ``position_distance(metal, transformed_neighbor) == distance``,
  and the cell translation must agree with the ``N_klm`` symmetry code.

* **Special-position deduplication.** A metal on a crystallographic axis sees
  several Gemmi images of one deposited donor land on (nearly) the same point.
  ``_deduplicate_special_position_contacts`` collapses them inside Gemmi's
  0.8 A special-position cutoff. Failing to collapse doubles the coordination
  number and fabricates a second multi-donor residue group out of nothing --
  the same corruption class as the already-fixed duplicate-record bugs. The
  much stricter 0.001 A tolerance belongs to *conflicting duplicate coordinate
  records* and must not be confused with this cutoff.

* **Explicit-only versus image-inclusive divergence.**
  ``explicit_contact_count`` / ``image_inclusive_contact_count``,
  ``generated_contact_scope``, ``coordination_depends_on_crystallographic_symmetry``
  and ``coordination_depends_on_strict_ncs`` are how a reader learns that a
  metal's coordination sphere exists only in a symmetry mate or only in an NCS
  copy. That is a headline scientific caveat, and the two scopes must not leak
  into each other.

Everything here is synthetic, offline, and needs neither CCP4 nor the network.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import Sequence, Tuple

import pytest

import gemmi

import bond_analysis as ba
import codes
from contact_record import Candidate
import structure_analysis as sa
from metal_elements import METAL_ELEMENTS

import helpers
from helpers import StructureBuilder, simple_metal_site


#: A deliberately tiny cell: every cell edge is short enough that a donor
#: deposited on the far side of the box has an image inside the metal's first
#: coordination sphere, which is what makes the symmetry branch fire at all.
SMALL_CELL: Tuple[float, ...] = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)

#: The suite's default cell. It is roomy enough that no crystallographic image
#: of a donor reaches a metal placed near its centre, so any generated contact
#: seen with this cell comes from strict NCS and nothing else.
ROOMY_CELL: Tuple[float, ...] = helpers.DEFAULT_CELL

#: An identity-rotation NCS operator is enough to generate strict-NCS images
#: without also having to reason about a rotated donor.
NCS_SHIFT: Tuple[float, float, float] = (-6.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Private helpers
# --------------------------------------------------------------------------- #
def _write_structure(
    builder: StructureBuilder,
    path,
    ncs: Sequence[Tuple[str, Tuple[float, float, float]]] = (),
) -> str:
    """Write ``builder`` as PDB, optionally with MTRIX strict-NCS operators.

    ``helpers.StructureBuilder`` has no NCS hook, so the operators are appended
    to the materialized gemmi structure. Each entry is ``(id, translation)``
    and uses an identity rotation, which keeps the generated image a pure
    translation of the deposited copy and therefore easy to place.
    """
    structure = builder.to_gemmi()
    for identifier, translation in ncs:
        transform = gemmi.Transform()
        transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        transform.vec.fromlist([float(value) for value in translation])
        structure.ncs.append(gemmi.NcsOp(transform, str(identifier), False))
    path = str(path)
    structure.write_pdb(path)
    return path


class _Analysis:
    """The four ``run_bond_analysis`` outputs plus the context they came from."""

    def __init__(self, context, rows, candidates, summaries, metadata):
        self.context = context
        self.rows = rows
        self.candidates = candidates
        self.summaries = summaries
        self.metadata = metadata

    @property
    def summary(self):
        """The single metal site's summary; asserts there is exactly one."""
        assert len(self.summaries) == 1, "expected a one-metal structure"
        return next(iter(self.summaries.values()))

    @property
    def metal_xyz(self) -> Tuple[float, float, float]:
        metals = self.context.metal_atoms(METAL_ELEMENTS, canonical=True)
        assert len(metals) == 1, "expected a one-metal structure"
        return metals[0].xyz


def _analyze(
    builder: StructureBuilder,
    tmp_path,
    name: str = "site",
    *,
    ncs: Sequence[Tuple[str, Tuple[float, float, float]]] = (),
    with_dpi: bool = False,
) -> _Analysis:
    """Write ``builder`` and run the real bond analysis over it.

    ``with_dpi=True`` supplies the refinement metadata that makes the DPI
    finite, which is what turns z-scores and geometry statuses into real
    values instead of ``insufficient data``.
    """
    path = _write_structure(
        builder, os.path.join(str(tmp_path), f"{name}.pdb"), ncs=ncs
    )
    context = sa.load_structure("test", path)
    stats_rows, header, _ = helpers.stats_rows_for_structure(
        context, os.path.join(str(tmp_path), f"{name}.out"), metrics={"ZDm": 2.0}
    )
    if with_dpi:
        data_json = helpers.write_data_json(os.path.join(str(tmp_path), f"{name}.json"))
        dpi_inputs = helpers.dpi_inputs(pdb_path=path, data_json=data_json)
    else:
        dpi_inputs = helpers.dpi_inputs()
    rows, candidates, summaries, metadata = ba.run_bond_analysis(
        "test", path, stats_rows, header, dpi_inputs, structure=context
    )
    return _Analysis(context, rows, candidates, summaries, metadata)


def _transformed_position(row) -> Tuple[float, float, float]:
    return (
        row["transformed_neighbor_x"],
        row["transformed_neighbor_y"],
        row["transformed_neighbor_z"],
    )


def _cell_translation(row) -> Tuple[int, int, int]:
    return (
        row["cell_translation_x"],
        row["cell_translation_y"],
        row["cell_translation_z"],
    )


def _decode_symmetry_code(code: str) -> Tuple[int, Tuple[int, int, int]]:
    """Split a ``N_klm`` symmetry code into its 1-based op number and shift.

    The three trailing digits are unit-cell translations offset by 5, which is
    the PDB/CCP4 convention Gemmi's ``symmetry_code()`` emits.
    """
    number, _, digits = str(code).partition("_")
    assert len(digits) == 3, f"unexpected symmetry code {code!r}"
    translation = (int(digits[0]) - 5, int(digits[1]) - 5, int(digits[2]) - 5)
    return int(number), translation


def _apply_spacegroup_operation(
    context,
    op_number: int,
    translation: Tuple[int, int, int],
    deposited_xyz: Sequence[float],
) -> Tuple[float, float, float]:
    """Independently rebuild an image position from the published provenance.

    Fractionalize the deposited coordinate, apply the ``op_number``-th
    space-group operation, add the reported cell translation, orthogonalize.
    Composing the steps in the wrong order, or applying the inverse operation,
    lands somewhere else, which is exactly what this reconstruction is meant to
    catch.
    """
    cell = context.structure.cell
    spacegroup = gemmi.find_spacegroup_by_name(context.structure.spacegroup_hm)
    operation = list(spacegroup.operations())[op_number - 1]
    fractional = cell.fractionalize(gemmi.Position(*deposited_xyz))
    moved = operation.apply_to_xyz([fractional.x, fractional.y, fractional.z])
    shifted = gemmi.Fractional(
        moved[0] + translation[0], moved[1] + translation[1], moved[2] + translation[2]
    )
    position = cell.orthogonalize(shifted)
    return (position.x, position.y, position.z)


# --------------------------------------------------------------------------- #
# The symmetry cases shared by the coordinate tests
# --------------------------------------------------------------------------- #
#: ``(builder factory, ncs ops, deposited donor xyz, expected code, scope)``.
#: Every case has exactly one Zn and one water; the water's *deposited* copy is
#: always out of range, so any reported contact is the image.
def _case_explicit():
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(10.0, 10.0, 10.0))
    builder.add_water(101, (10.0, 12.09, 10.0), chain="B")
    return builder, (), (10.0, 12.09, 10.0)


def _case_cell_edge():
    """The ``-a`` image of a water deposited near the far cell face."""
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (18.41, 0.5, 0.5), chain="B")
    return builder, (), (18.41, 0.5, 0.5)


def _case_screw_axis():
    """A P 21 21 21 21-screw image, i.e. a genuinely rotated operation."""
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 21 21 21")
    builder.add_metal("ZN", 1, chain="B", pos=(5.0, 0.0, 5.0))
    builder.add_water(101, (5.0, 0.0, -2.91), chain="B")
    return builder, (), (5.0, 0.0, -2.91)


def _case_screw_axis_plus_translation():
    """The same screw image, reached only after an extra ``+a`` cell shift."""
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 21 21 21")
    builder.add_metal("ZN", 1, chain="B", pos=(5.0, 0.0, 5.0))
    builder.add_water(101, (25.0, 0.0, -2.91), chain="B")
    return builder, (), (25.0, 0.0, -2.91)


def _case_strict_ncs():
    """A strict-NCS image, which is not a space-group operation at all."""
    builder = StructureBuilder(cell=ROOMY_CELL, spacegroup="P 21 21 21")
    builder.add_metal("ZN", 1, chain="B", pos=(10.0, 10.0, 10.0))
    builder.add_water(101, (18.09, 10.0, 10.0), chain="B")
    return builder, (("1", NCS_SHIFT),), (18.09, 10.0, 10.0)


#: ``(factory, expected symmetry code, expected scope, crystallographic?)``
SYMMETRY_CASES = [
    (_case_explicit, "1_555", "explicit", True),
    (_case_cell_edge, "1_455", "crystallographic", True),
    (_case_screw_axis, "2_555", "crystallographic", True),
    (_case_screw_axis_plus_translation, "2_655", "crystallographic", True),
    (_case_strict_ncs, "5_555", "strict_ncs", False),
]


def _case_id(factory) -> str:
    name = factory.__name__
    prefix = "_case_"
    return name[len(prefix) :] if name.startswith(prefix) else name


_CASE_IDS = [_case_id(factory) for factory, _, _, _ in SYMMETRY_CASES]

#: The subset whose symmetry code addresses a space-group operation, and can
#: therefore be rebuilt from the space group alone.
CRYSTALLOGRAPHIC_CASES = [
    factory for factory, _, _, crystallographic in SYMMETRY_CASES if crystallographic
]


@pytest.fixture(params=SYMMETRY_CASES, ids=_CASE_IDS)
def symmetry_case(request, tmp_path):
    """One analyzed single-water site per symmetry case, plus its expectations.

    Returns ``(analysis, deposited_xyz, expected_code, expected_scope,
    is_crystallographic)``.
    """
    factory, code, scope, crystallographic = request.param
    builder, ncs, deposited = factory()
    analysis = _analyze(builder, tmp_path, _case_id(factory), ncs=ncs)
    return analysis, deposited, code, scope, crystallographic


# --------------------------------------------------------------------------- #
# G3 -- the published image coordinates
# --------------------------------------------------------------------------- #
def test_symmetry_case_reports_the_expected_operation_and_scope(symmetry_case):
    """Each case really exercises the branch it claims to.

    Without this the coordinate assertions below could all be passing on the
    trivial identity image because a donor was misplaced.
    """
    analysis, _, code, scope, _ = symmetry_case

    assert len(analysis.rows) == 1
    row = analysis.rows[0]
    assert row["symmetry_operation"] == code
    assert row["contact_scope"] == scope
    assert row["distance"] == pytest.approx(2.09, abs=1e-3)


def test_transformed_neighbor_position_reproduces_the_reported_distance(symmetry_case):
    """``|metal - transformed_neighbor|`` must equal the row's own ``distance``.

    This is the invariant that makes the published donor position usable. An
    inverted or mis-composed transform would still report the right distance --
    Gemmi measures that separately -- while writing a position that is
    somewhere else entirely, and a downstream consumer plotting the site would
    silently draw a bond that does not exist.
    """
    analysis, _, _, _, _ = symmetry_case
    row = analysis.rows[0]

    recomputed = sa.position_distance(analysis.metal_xyz, _transformed_position(row))

    # ``distance`` is rounded to 3 decimals and the position to 6, so the
    # comparison tolerance is the rounding of ``distance`` itself.
    assert recomputed == pytest.approx(row["distance"], abs=1e-3)


def test_published_image_positions_are_rounded_in_both_streams(symmetry_case):
    """Both CSVs publish 6-decimal coordinates, and the test above assumes it.

    The rounding is what makes the tolerance in
    ``test_transformed_neighbor_position_reproduces_the_reported_distance``
    correct, and it is what keeps a 17-digit float repr out of a column users
    read. Asserted on the candidate stream as well as the bond stream: they are
    built by two different functions that round separately.
    """
    analysis, _, _, _, _ = symmetry_case

    published = [_transformed_position(row) for row in analysis.rows]
    published += [_transformed_position(row) for row in analysis.candidates]

    assert published, "the fixture must have produced rows for this to mean anything"
    for position in published:
        assert position == tuple(round(value, 6) for value in position)


def test_cell_translation_agrees_with_the_symmetry_code(symmetry_case):
    """The ``klm`` digits of the code and ``cell_translation_*`` say one thing.

    They are two renderings of the same lattice shift. If they disagree, one of
    the two is a lie about which crystal image was measured, and there is no
    way for a reader to tell which.
    """
    analysis, _, _, _, _ = symmetry_case
    row = analysis.rows[0]

    op_number, code_translation = _decode_symmetry_code(row["symmetry_operation"])

    assert _cell_translation(row) == code_translation
    # Gemmi numbers symmetry codes from one and image indices from zero.
    assert row["symmetry_image_index"] == op_number - 1


def test_transformed_neighbor_is_the_image_not_the_deposited_atom(symmetry_case):
    """A generated contact must publish the image, never the deposited copy.

    The whole point of the column is that the contacting atom is somewhere the
    coordinate file does not contain. Publishing the deposited position while
    reporting the image distance is precisely the inconsistency this module
    exists to prevent, so assert the two differ by the amount the geometry
    demands.
    """
    analysis, deposited, _, scope, _ = symmetry_case
    row = analysis.rows[0]
    transformed = _transformed_position(row)
    deposited_distance = sa.position_distance(analysis.metal_xyz, deposited)

    if scope == "explicit":
        assert transformed == pytest.approx(deposited, abs=1e-6)
        assert _cell_translation(row) == (0, 0, 0)
        assert row["symmetry_contact"] is False
        return

    assert sa.position_distance(transformed, deposited) > 1.0
    assert deposited_distance > ba.CUTOFF
    assert row["distance"] < deposited_distance
    assert row["symmetry_contact"] is True


@pytest.mark.parametrize(
    "factory", CRYSTALLOGRAPHIC_CASES, ids=[_case_id(f) for f in CRYSTALLOGRAPHIC_CASES]
)
def test_crystallographic_image_matches_an_independent_transform(factory, tmp_path):
    """Rebuild the image from the code and the cell shift, and compare.

    Fractionalize the deposited atom, apply the named space-group operation,
    add the reported cell translation, orthogonalize. Getting the composition
    order wrong, or applying the inverse operation, changes the answer for the
    screw-axis cases, so this pins the transform itself rather than only its
    length. Strict-NCS images are excluded rather than skipped: their operation
    index addresses an NCS block, not a space-group operation, so there is no
    reconstruction to make.
    """
    builder, ncs, deposited = factory()
    analysis = _analyze(builder, tmp_path, _case_id(factory), ncs=ncs)
    row = analysis.rows[0]

    op_number, _ = _decode_symmetry_code(row["symmetry_operation"])
    expected = _apply_spacegroup_operation(
        analysis.context, op_number, _cell_translation(row), deposited
    )

    assert _transformed_position(row) == pytest.approx(expected, abs=1e-6)


def test_candidate_rows_carry_the_same_image_geometry(symmetry_case):
    """The candidate table must not contradict the bond table.

    ``candidate_distance`` and the transformed position travel together in
    ``CANDIDATE_COLUMNS`` too, and a reader who joins the two tables on the
    donor identity has to get one consistent site.
    """
    analysis, _, code, scope, _ = symmetry_case

    matching = [
        candidate
        for candidate in analysis.candidates
        if candidate["neighbor_atom"] == "O"
    ]
    assert len(matching) == 1
    candidate = matching[0]

    assert candidate["symmetry_operation"] == code
    assert candidate["contact_scope"] == scope
    assert _transformed_position(candidate) == pytest.approx(
        _transformed_position(analysis.rows[0]), abs=1e-9
    )
    assert _cell_translation(candidate) == _cell_translation(analysis.rows[0])
    recomputed = sa.position_distance(
        analysis.metal_xyz, _transformed_position(candidate)
    )
    assert recomputed == pytest.approx(candidate["candidate_distance"], abs=1e-3)


def test_every_bond_row_position_matches_its_distance_in_a_mixed_site(tmp_path):
    """The invariant holds row by row when explicit and generated rows mix.

    A per-row check matters more than a single-contact one: the transformed
    position is written from the candidate that produced the row, so a mix-up
    between two candidates of the same site would leave each individual
    distance plausible while pairing it with the wrong position.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (0.5, 2.59, 0.5), chain="B")  # explicit, 2.09 A
    builder.add_water(102, (18.41, 0.5, 0.5), chain="B")  # -a image, 2.09 A
    builder.add_water(103, (0.5, 0.5, 18.31), chain="B")  # -c image, 2.19 A
    analysis = _analyze(builder, tmp_path, "mixed")

    assert len(analysis.rows) == 3
    scopes = sorted(row["contact_scope"] for row in analysis.rows)
    assert scopes == ["crystallographic", "crystallographic", "explicit"]

    for row in analysis.rows:
        recomputed = sa.position_distance(
            analysis.metal_xyz, _transformed_position(row)
        )
        assert recomputed == pytest.approx(row["distance"], abs=1e-3), row
        _, code_translation = _decode_symmetry_code(row["symmetry_operation"])
        assert _cell_translation(row) == code_translation


# --------------------------------------------------------------------------- #
# G4 -- special-position deduplication
# --------------------------------------------------------------------------- #
def _axis_site(offset: float, *, donor: str = "water") -> StructureBuilder:
    """A Zn on the ``P 1 2 1`` two-fold, with a donor ``offset`` A off the axis.

    The two-fold along **b** maps ``(x, y, z)`` to ``(-x, y, -z)``, so an atom
    at ``x = offset, z = 0`` and its image are ``2 * offset`` apart and both sit
    at the same distance from a metal on the axis. ``offset = 0`` is the exact
    special position; anything under 0.4 A is the near-coincident case the
    0.8 A cutoff exists for.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1 2 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 10.0, 0.0))
    if donor == "water":
        builder.add_water(101, (offset, 12.09, 0.0), chain="B")
    else:
        builder.add_amino_acid(
            "ASP",
            10,
            chain="A",
            positions={"OD1": (offset, 12.09, 0.0), "OD2": (offset, 7.91, 0.0)},
            origin=(30.0, 30.0, 30.0),
        )
    return builder


def _raw_and_deduplicated(context):
    """Return the pre- and post-collapse candidate lists for the one metal.

    ``_collect_proximal_candidates`` is the last stage that still holds one
    candidate per Gemmi image, so it is the only place the collapse can be
    observed as a difference rather than inferred from a count.
    """
    metals = context.metal_atoms(METAL_ELEMENTS, canonical=True)
    assert len(metals) == 1
    search = context.make_neighbor_search(
        ba.CUTOFF + ba.SEARCH_EPSILON,
        include_symmetry=True,
        positive_occupancy_only=True,
    )
    raw = ba._collect_proximal_candidates(context, search, metals[0], True)
    return raw, ba._deduplicate_special_position_contacts(raw)


def test_metal_on_a_two_fold_axis_collapses_the_coincident_donor_image(tmp_path):
    """One deposited water on the axis must yield one contact, not two.

    The two-fold maps the water onto itself, so Gemmi offers two images of the
    same deposited atom at the same point and the same distance. Reporting both
    would claim a two-coordinate site built from a single oxygen.
    """
    path = _write_structure(_axis_site(0.0), tmp_path / "axis.pdb")
    context = sa.load_structure("test", path)
    assert context.crystallographic_operation_count == 2

    raw, collapsed = _raw_and_deduplicated(context)

    assert len(raw) == 2
    assert {candidate.symmetry_operation for candidate in raw} == {"1_555", "2_555"}
    source_keys = {candidate.neighbor.source_key for candidate in raw}
    assert len(source_keys) == 1, "both images are of one deposited atom"
    assert len(collapsed) == 1

    analysis = _analyze(_axis_site(0.0), tmp_path, "axis_full")
    assert len(analysis.rows) == 1
    assert analysis.summary["candidate_contact_count"] == 1


def test_special_position_collapse_keeps_the_explicit_image(tmp_path):
    """The surviving representative is the deposited image, not a generated one.

    ``_special_position_preference`` prefers an explicit image so that a
    fraction-of-an-Angstrom refinement artifact cannot reclassify an ordinary
    deposited contact as symmetry-dependent -- which would then propagate into
    ``generated_contact_scope`` and the ``coordination_depends_on_*`` flags.
    """
    analysis = _analyze(_axis_site(0.025), tmp_path, "prefer")

    assert len(analysis.rows) == 1
    row = analysis.rows[0]
    assert row["contact_scope"] == "explicit"
    assert row["symmetry_operation"] == "1_555"
    assert row["symmetry_contact"] is False
    # The collapse must not manufacture a symmetry dependence for the site.
    assert analysis.summary["generated_contact_scope"] == "none"
    assert (
        analysis.summary["coordination_depends_on_crystallographic_symmetry"] is False
    )


@pytest.mark.parametrize(
    "offset, separation, expected_contacts",
    [
        (0.0, 0.0, 1),  # the exact special position
        (0.001, 0.002, 1),  # just past the 0.001 A duplicate-record tolerance
        (0.025, 0.05, 1),  # 50x that tolerance, still one deposited atom
        (0.25, 0.5, 1),
        (0.395, 0.79, 1),  # just inside the 0.8 A special-position cutoff
        (0.45, 0.9, 2),  # outside it: two genuinely distinct images
        (0.75, 1.5, 2),
    ],
)
def test_images_collapse_only_within_the_special_position_cutoff(
    tmp_path, offset, separation, expected_contacts
):
    """0.8 A is the collapse threshold, and it is applied to image separation.

    Below it the two images are one atom on a special position; above it they
    are two distinct crystal contacts and merging them would *lose* a real
    donor. The separation is asserted alongside the outcome so the case cannot
    quietly stop testing the boundary if the geometry is edited.
    """
    builder = _axis_site(offset)
    path = _write_structure(builder, tmp_path / f"sep_{offset}.pdb")
    context = sa.load_structure("test", path)

    raw, collapsed = _raw_and_deduplicated(context)
    assert len(raw) == 2
    measured = sa.position_distance(
        raw[0].transformed_position, raw[1].transformed_position
    )
    assert measured == pytest.approx(separation, abs=1e-6)
    assert (measured <= ba.SPECIAL_POSITION_DEDUP_CUTOFF) is (expected_contacts == 1)

    assert len(collapsed) == expected_contacts
    analysis = _analyze(builder, tmp_path, f"sep_full_{offset}")
    assert len(analysis.rows) == expected_contacts


def test_images_exactly_at_the_point_eight_angstrom_cutoff_collapse():
    """The published 0.8 A boundary is inclusive, not merely approximately so."""
    neighbor = SimpleNamespace(
        source_key=(0, 0, 0, 0), chain_index=0, residue_index=0, atom_index=0
    )

    def candidate(x, *, symmetry):
        return Candidate(
            neighbor=neighbor,
            distance_raw=2.0,
            transformed_position=(x, 0.0, 0.0),
            symmetry_contact=symmetry,
            crystallographic_contact=symmetry,
            strict_ncs_contact=False,
            strict_ncs_operation_id="",
            contact_scope=(
                codes.ContactScope.CRYSTALLOGRAPHIC
                if symmetry
                else codes.ContactScope.EXPLICIT
            ),
            symmetry_image_index=int(symmetry),
            symmetry_operation="2_555" if symmetry else "1_555",
            translation=(0, 0, 0),
            candidate_sources={
                codes.CandidateSource.STRUCT_CONN
                if symmetry
                else codes.CandidateSource.PROXIMITY_4A
            },
        )

    origin = candidate(0.0, symmetry=False)
    boundary = candidate(0.8, symmetry=True)
    assert (
        sa.position_distance(origin.transformed_position, boundary.transformed_position)
        == 0.8
    )

    collapsed = ba._deduplicate_special_position_contacts([origin, boundary])
    assert len(collapsed) == 1
    assert collapsed[0].candidate_sources == {
        codes.CandidateSource.PROXIMITY_4A,
        codes.CandidateSource.STRUCT_CONN,
    }

    outside = ba._deduplicate_special_position_contacts(
        [candidate(0.0, symmetry=False), candidate(0.800001, symmetry=True)]
    )
    assert len(outside) == 2


def test_special_position_cutoff_is_not_the_duplicate_record_tolerance():
    """The 0.8 A and 0.001 A thresholds are separate constants for separate jobs.

    ``README.md`` promises both: near-coincident *images* collapse at 0.8 A,
    while 0.001 A stays reserved for deciding that two deposited records with
    the same identity disagree about where the atom is. Conflating them would
    either stop collapsing special positions or start silently accepting
    half-Angstrom coordinate conflicts.
    """
    assert ba.SPECIAL_POSITION_DEDUP_CUTOFF == 0.8
    assert sa.DUPLICATE_ATOM_POSITION_TOLERANCE == 0.001
    assert ba.SPECIAL_POSITION_DEDUP_CUTOFF > sa.DUPLICATE_ATOM_POSITION_TOLERANCE * 100


def test_collapsed_images_are_not_reported_as_duplicate_records(tmp_path):
    """A special position is not a deposition error, and vice versa.

    The axis site has one deposited water record whose images coincide: the
    duplicate-record counters must stay at zero. The converse structure -- two
    records for one atom 0.05 A apart -- is a coordinate conflict even though
    0.05 A is far inside the 0.8 A image cutoff, so the wider cutoff does not
    silence the stricter check.
    """
    axis_path = _write_structure(_axis_site(0.025), tmp_path / "axis_dup.pdb")
    axis = sa.load_structure("test", axis_path)
    assert axis.duplicate_atom_records_present is False
    assert axis.duplicate_atom_record_count == 0
    assert axis.duplicate_coordinate_conflict_count == 0

    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1 2 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 10.0, 0.0))
    builder.add_hetero_residue(
        "HOH",
        101,
        [
            helpers.AtomSpec("O", "O", (0.0, 12.09, 0.0), occupancy=0.5),
            helpers.AtomSpec("O", "O", (0.05, 12.09, 0.0), occupancy=0.5),
        ],
        chain="B",
    )
    conflict = sa.load_structure(
        "test", _write_structure(builder, tmp_path / "conflict.pdb")
    )

    assert conflict.duplicate_atom_record_count == 1
    assert conflict.duplicate_coordinate_conflict_count == 1
    assert "duplicate_atom_coordinate_conflict" in conflict.warning_codes


def test_failing_to_collapse_inflates_coordination_and_invents_a_group(
    tmp_path, monkeypatch
):
    """Show the corruption the collapse prevents, by neutralizing the cutoff.

    An aspartate on the two-fold contributes OD1 and OD2, so the site is a
    genuine two-donor chelate. With the special-position cutoff neutralized the
    same deposited pair is counted twice: the coordination number doubles, a
    second multi-donor residue group appears out of nothing, and the site is
    additionally mislabelled as depending on crystallographic symmetry. That is
    the same class of fabricated-coordination corruption as the duplicate-atom
    defects already fixed, which is why the collapse is not cosmetic.
    """
    builder = _axis_site(0.025, donor="aspartate")

    collapsed = _analyze(builder, tmp_path, "chelate")
    assert len(collapsed.rows) == 2
    assert sorted(row["neighbor_atom"] for row in collapsed.rows) == ["OD1", "OD2"]
    assert collapsed.summary["candidate_contact_count"] == 2
    assert collapsed.summary["multi_donor_residue_group_count"] == 1
    assert collapsed.summary["multi_donor_contact_count"] == 2
    assert collapsed.summary["generated_contact_scope"] == "none"
    assert (
        collapsed.summary["coordination_depends_on_crystallographic_symmetry"] is False
    )

    # A zero cutoff still merges exactly coincident images, so this isolates
    # the near-coincident case the 0.8 A value exists for.
    monkeypatch.setattr(ba, "SPECIAL_POSITION_DEDUP_CUTOFF", 0.0)
    uncollapsed = _analyze(builder, tmp_path, "chelate_raw")

    assert len(uncollapsed.rows) == 4
    assert uncollapsed.summary["candidate_contact_count"] == 4
    assert uncollapsed.summary["multi_donor_residue_group_count"] == 2
    assert uncollapsed.summary["multi_donor_contact_count"] == 4
    assert uncollapsed.summary["generated_contact_scope"] == "crystallographic"
    assert (
        uncollapsed.summary["coordination_depends_on_crystallographic_symmetry"] is True
    )


def test_collapse_is_independent_of_the_neighbor_search_order(tmp_path):
    """The retained representative may not depend on Gemmi's mark order.

    ``_deduplicate_special_position_contacts`` sorts each source-atom group
    before comparing, so shuffling the input cannot change which image
    survives. Without that sort the reported scope of a special-position
    contact would be luck of the draw between runs.
    """
    path = _write_structure(_axis_site(0.025), tmp_path / "order.pdb")
    context = sa.load_structure("test", path)
    raw, collapsed = _raw_and_deduplicated(context)

    reversed_result = ba._deduplicate_special_position_contacts(list(reversed(raw)))

    assert len(collapsed) == len(reversed_result) == 1
    assert reversed_result[0].symmetry_operation == collapsed[0].symmetry_operation
    assert reversed_result[0].transformed_position == pytest.approx(
        collapsed[0].transformed_position, abs=1e-9
    )


# --------------------------------------------------------------------------- #
# G8 -- explicit-only versus image-inclusive
# --------------------------------------------------------------------------- #
def test_purely_explicit_site_declares_no_generated_dependence(tmp_path):
    """All donors in the asymmetric unit: the two scopes must agree exactly.

    This is the control. If the generated columns can report a dependence here,
    every claim they make elsewhere is worthless.
    """
    builder = simple_metal_site(
        "ZN",
        [("HIS", "NE2", 2.03), ("ASP", "OD1", 1.99), ("HOH", "O", 2.09)],
        cell=ROOMY_CELL,
    )
    analysis = _analyze(builder, tmp_path, "explicit_only")
    summary = analysis.summary

    assert summary["explicit_contact_count"] == 3
    assert summary["image_inclusive_contact_count"] == 3
    assert summary["symmetry_contact_count"] == 0
    assert summary["crystallographic_contact_count"] == 0
    assert summary["strict_ncs_contact_count"] == 0
    assert summary["generated_contact_scope"] == "none"
    assert summary["coordination_depends_on_crystallographic_symmetry"] is False
    assert summary["coordination_depends_on_strict_ncs"] is False
    assert all(row["contact_scope"] == "explicit" for row in analysis.rows)


def test_purely_generated_site_reports_no_explicit_contacts(tmp_path):
    """A coordination sphere that exists only in a symmetry mate says so.

    ``explicit_contact_count == 0`` with a nonzero image-inclusive count is the
    strongest caveat the summary can carry: nothing in the deposited
    asymmetric unit coordinates this metal. Losing that distinction would let a
    lattice contact be read as a deposited one.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (18.41, 0.5, 0.5), chain="B")
    analysis = _analyze(builder, tmp_path, "generated_only")
    summary = analysis.summary

    assert summary["explicit_contact_count"] == 0
    assert summary["image_inclusive_contact_count"] == 1
    assert summary["symmetry_contact_count"] == 1
    assert summary["crystallographic_contact_count"] == 1
    assert summary["strict_ncs_contact_count"] == 0
    assert summary["combined_ncs_crystallographic_contact_count"] == 0
    assert summary["generated_contact_scope"] == "crystallographic"
    assert summary["coordination_depends_on_crystallographic_symmetry"] is True
    assert summary["coordination_depends_on_strict_ncs"] is False
    # The explicit-only view has nothing to assess, and says so rather than
    # borrowing the image-inclusive coverage.
    assert math.isnan(summary["geometry_coverage_explicit"])
    assert summary["geometry_coverage_image_inclusive"] == pytest.approx(1.0)


def test_mixed_site_keeps_the_two_counts_apart(tmp_path):
    """One deposited donor plus one lattice donor: 1 explicit, 2 image-inclusive.

    The counts must not be the same number, and the primary (image-inclusive)
    bond rows must still be individually attributable to a scope.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (0.5, 2.59, 0.5), chain="B")
    builder.add_water(102, (18.41, 0.5, 0.5), chain="B")
    analysis = _analyze(builder, tmp_path, "mixed_scope")
    summary = analysis.summary

    assert summary["explicit_contact_count"] == 1
    assert summary["image_inclusive_contact_count"] == 2
    assert summary["symmetry_contact_count"] == 1
    assert summary["generated_contact_scope"] == "crystallographic"
    assert summary["coordination_depends_on_crystallographic_symmetry"] is True
    assert summary["coordination_depends_on_strict_ncs"] is False

    scopes = sorted(row["contact_scope"] for row in analysis.rows)
    assert scopes == ["crystallographic", "explicit"]
    assert [row["symmetry_contact"] for row in analysis.rows].count(True) == 1


def test_strict_ncs_site_does_not_claim_a_crystallographic_dependence(tmp_path):
    """An NCS-only contact sets the NCS flag and leaves the crystal flag alone.

    The two provenances answer different questions -- "another copy of the same
    molecule" versus "the neighbouring unit cell" -- and a reader draws
    different conclusions from each, so they must not leak into one another.
    """
    builder = StructureBuilder(cell=ROOMY_CELL, spacegroup="P 21 21 21")
    builder.add_metal("ZN", 1, chain="B", pos=(10.0, 10.0, 10.0))
    builder.add_water(101, (18.09, 10.0, 10.0), chain="B")
    analysis = _analyze(builder, tmp_path, "ncs_only", ncs=[("1", NCS_SHIFT)])
    summary = analysis.summary

    assert analysis.context.strict_ncs_operation_count == 1
    assert summary["explicit_contact_count"] == 0
    assert summary["image_inclusive_contact_count"] == 1
    assert summary["strict_ncs_contact_count"] == 1
    assert summary["crystallographic_contact_count"] == 0
    assert summary["combined_ncs_crystallographic_contact_count"] == 0
    assert summary["generated_contact_scope"] == "strict_ncs"
    assert summary["coordination_depends_on_strict_ncs"] is True
    assert summary["coordination_depends_on_crystallographic_symmetry"] is False

    row = analysis.rows[0]
    assert row["contact_scope"] == "strict_ncs"
    assert row["strict_ncs_contact"] is True
    assert row["crystallographic_contact"] is False
    assert row["strict_ncs_operation_id"] == "1"


def test_both_provenances_present_report_the_combined_scope(tmp_path):
    """One NCS donor and one lattice donor set both flags and the joint scope.

    ``strict_ncs_and_crystallographic`` here describes the *site*: two
    separately generated contacts. It is not the same as
    ``combined_ncs_crystallographic_contact_count``, which counts single
    contacts needing both transforms at once and stays zero.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (18.5, 0.5, 0.5), chain="B")  # -a image, 2.0 A
    builder.add_water(102, (10.5, 0.5, 0.5), chain="B")  # NCS image, 2.0 A
    analysis = _analyze(
        builder, tmp_path, "ncs_and_xtal", ncs=[("1", (-8.0, 0.0, 0.0))]
    )
    summary = analysis.summary

    assert summary["explicit_contact_count"] == 0
    assert summary["image_inclusive_contact_count"] == 2
    assert summary["crystallographic_contact_count"] == 1
    assert summary["strict_ncs_contact_count"] == 1
    assert summary["combined_ncs_crystallographic_contact_count"] == 0
    assert summary["generated_contact_scope"] == "strict_ncs_and_crystallographic"
    assert summary["coordination_depends_on_crystallographic_symmetry"] is True
    assert summary["coordination_depends_on_strict_ncs"] is True

    assert sorted(row["contact_scope"] for row in analysis.rows) == [
        "crystallographic",
        "strict_ncs",
    ]


def test_generated_columns_are_blank_without_symmetry_metadata(tmp_path):
    """No cell means no image search, and the flags must abstain, not say False.

    A blank is "not determined"; ``False`` would assert that the coordination
    provably does not depend on symmetry, which nobody checked.
    """
    builder = simple_metal_site(
        "ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)], cell=None
    )
    analysis = _analyze(builder, tmp_path, "nocell")
    summary = analysis.summary

    assert analysis.context.symmetry_search_available is False
    assert summary["explicit_contact_count"] == 2
    assert math.isnan(summary["image_inclusive_contact_count"])
    assert math.isnan(summary["symmetry_contact_count"])
    assert summary["generated_contact_scope"] == ""
    assert summary["coordination_depends_on_crystallographic_symmetry"] == ""
    assert summary["coordination_depends_on_strict_ncs"] == ""
    assert "symmetry_search_unavailable" in summary[
        "geometry_not_assessed_reason"
    ].split("|")
    assert "symmetry_search_unavailable" in analysis.metadata["partial_reason_codes"]


def test_generated_images_can_change_the_geometry_verdict(tmp_path):
    """The headline caveat: the verdict differs between the two scopes.

    The deposited histidine alone looks plausible; the water reached across a
    cell edge sits at 1.35 A, far too short for Zn-O, and makes the
    image-inclusive assessment suspect. Both verdicts are published, and
    ``geometry_classification_changes_with_generated_images`` is the flag that
    tells a reader the difference is real rather than a rounding artifact --
    a user who reads only the primary result would otherwise blame the
    deposited model for a lattice contact.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (0.5, 2.53, 0.5)},
        origin=(10.0, 10.0, 10.0),
    )
    builder.add_water(101, (19.15, 0.5, 0.5), chain="B")
    analysis = _analyze(builder, tmp_path, "verdict", with_dpi=True)
    summary = analysis.summary

    assert math.isfinite(summary["dpi"])
    assert summary["explicit_contact_count"] == 1
    assert summary["image_inclusive_contact_count"] == 2
    assert summary["geometry_outlier_count_explicit"] == 0
    assert summary["geometry_outlier_count_image_inclusive"] == 1
    assert summary["explicit_geometry_status"] == "plausible"
    assert summary["image_inclusive_geometry_status"] == "suspect"
    assert summary["geometry_classification_changes_with_generated_images"] is True

    (outlier,) = [row for row in analysis.rows if row["geometry_outlier"]]
    assert outlier["contact_scope"] == "crystallographic"
    assert outlier["distance"] == pytest.approx(1.35, abs=1e-3)


def test_count_divergence_alone_does_not_flip_the_classification_flag(tmp_path):
    """The flag tracks the verdict, not the contact count.

    Adding a well-behaved lattice donor changes both counts but leaves both
    scopes classified the same way, so the flag must stay ``False``. Wiring it
    to a count comparison would fire on almost every crystal structure and stop
    meaning anything.
    """
    builder = StructureBuilder(cell=SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (0.5, 2.53, 0.5)},
        origin=(10.0, 10.0, 10.0),
    )
    builder.add_water(101, (18.41, 0.5, 0.5), chain="B")  # -a image at 2.09 A
    analysis = _analyze(builder, tmp_path, "no_flip", with_dpi=True)
    summary = analysis.summary

    assert summary["explicit_contact_count"] == 1
    assert summary["image_inclusive_contact_count"] == 2
    assert (
        summary["explicit_geometry_status"]
        == summary["image_inclusive_geometry_status"]
        == "plausible"
    )
    assert summary["geometry_classification_changes_with_generated_images"] is False
    assert summary["coordination_depends_on_crystallographic_symmetry"] is True
