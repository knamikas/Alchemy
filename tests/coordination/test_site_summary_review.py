"""Per-site summary columns in ``src/coordination/site_summary.py``.

These cover the parts of the summary that its own callers cannot: the
deposited atom count it repeats from the entry, the occupancy assessment of
the metal's own chemical site, the donor B-factor comparison when a donor has
no usable B factor, and the generated-image counts and scope vocabulary.
Fixtures are synthetic and require neither CCP4 nor network access.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import gemmi
import helpers
import pytest
from helpers import StructureBuilder, approx, simple_metal_site

import structure_analysis as sa
from codes import ContactScope, ReasonCode
from coordination.contact_record import Candidate
from coordination.dpi import DpiComponents
from coordination.site_environment import (
    EntryModelStatistics,
    MetalProximity,
    MetalSpecialPosition,
)
from coordination.site_summary import (
    SiteSummary,
    _donor_b_factors,  # pyright: ignore[reportPrivateUsage]
    _image_count,  # pyright: ignore[reportPrivateUsage]
    site_summary,
    unassessable_site_summary,
)
from metal_elements import METAL_ELEMENTS
from structure_analysis import NAN, AtomSite, ContactImage, StructureContext

# PDB occupancy, columns 55-60 (0-based 54:60). Gemmi always writes a
# well-formed occupancy, so an unreadable one is injected at the text level.
_OCCUPANCY_COLUMN = 54

# A 20 A cube leaves a donor across the boundary within the first sphere.
_SMALL_CELL: tuple[float, ...] = (20.0, 20.0, 20.0, 90.0, 90.0, 90.0)


def _blank_occupancy(source: str, destination: Path, atom_name: str) -> str:
    """Copy a PDB file with every ``atom_name`` record's occupancy blanked."""
    lines: list[str] = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == atom_name:
                line = (
                    line[:_OCCUPANCY_COLUMN] + " " * 6 + line[_OCCUPANCY_COLUMN + 6 :]
                )
            lines.append(line)
    destination.write_text("".join(lines), encoding="utf-8")
    return str(destination)


def _one_metal(context: StructureContext) -> AtomSite:
    """The single analyzed metal of a one-metal structure."""
    metals = context.metal_atoms(METAL_ELEMENTS, canonical=True)
    assert len(metals) == 1, "expected a one-metal structure"
    return metals[0]


def _summary_for(
    context: StructureContext,
    metal: AtomSite,
    *,
    model_statistics: EntryModelStatistics | None = None,
) -> SiteSummary:
    """Summarize ``metal`` with no contacts, so the caller's inputs are the test."""
    return site_summary(
        metal,
        context,
        [],
        [],
        [],
        DpiComponents(
            dpi=NAN,
            resolution=NAN,
            reason_code="",
            r_free=NAN,
            reflection_count=NAN,
            asu_volume=NAN,
        ),
        (
            EntryModelStatistics.from_structure(context)
            if model_statistics is None
            else model_statistics
        ),
        MetalProximity.unavailable(),
        MetalSpecialPosition.unavailable(),
    )


def test_the_summary_repeats_the_deposited_atom_count_not_the_analyzed_one(
    tmp_path: Path,
) -> None:
    """``deposited_...`` counts the records; the other count adds the NCS copies.

    Strict NCS multiplies the asymmetric unit's contents without adding a
    record to the file, so the two columns must differ for such an entry or
    one of them is reporting the other's quantity.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    structure = builder.to_gemmi()
    transform = gemmi.Transform()
    transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([-6.0, 0.0, 0.0])
    structure.ncs.append(gemmi.NcsOp(transform, "1", False))
    path = helpers.write_pdb(structure, str(tmp_path / "ncs.pdb"))
    context = sa.load_structure("test", path)
    summary = _summary_for(context, _one_metal(context))

    deposited = sa.count_deposited_ni(context)
    assert deposited == approx(sa.count_ni(context) / 2.0)
    assert summary["deposited_occupancy_weighted_atom_count"] == approx(deposited)
    assert summary["occupancy_weighted_atom_count"] == approx(sa.count_ni(context))
    assert summary["deposited_occupancy_weighted_atom_count"] == round(deposited, 6)


def test_a_non_finite_atom_count_is_published_as_nan_not_rounded(
    tmp_path: Path,
) -> None:
    """Both atom counts abstain when the entry could not establish them.

    ``round()`` accepts a NaN and returns one, so the guard is what keeps an
    infinite count -- which ``round()`` raises on -- from ending the entry.
    """
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = builder.write_pdb(tmp_path / "counts.pdb")
    context = sa.load_structure("test", path)
    summary = _summary_for(
        context,
        _one_metal(context),
        model_statistics=EntryModelStatistics(
            occupancy_weighted_atom_count=math.inf,
            deposited_occupancy_weighted_atom_count=NAN,
            nonwater_median_b_iso=NAN,
        ),
    )

    assert math.isnan(summary["occupancy_weighted_atom_count"])
    assert math.isnan(summary["deposited_occupancy_weighted_atom_count"])


def test_overfull_occupancy_is_blank_when_the_metal_records_cannot_be_read(
    tmp_path: Path,
) -> None:
    """An unreadable alternate occupancy leaves the sum unknown, so abstain.

    The entry-wide survey skips any chemical site with an unusable occupancy,
    so the site is absent from ``overfull_site_keys`` because nobody judged it.
    Reporting ``False`` there would assert the deposition is sound on evidence
    that was never read.
    """
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    metal_residue = builder.residues[0]
    builder.add_conformers(
        metal_residue, [("A", 0.46, {}), ("B", 0.55, {})], atom_names=["ZN"]
    )
    clean = builder.write_pdb(tmp_path / "clean.pdb")
    path = _blank_occupancy(clean, tmp_path / "unreadable.pdb", "ZN")
    context = sa.load_structure("test", path)
    metal = _one_metal(context)

    assert context.occupancy.missing_count == 2
    assert metal.chemical_site_identity not in context.occupancy.overfull_site_keys
    assert _summary_for(context, metal)["metal_overfull_occupancy"] == ""


def test_overfull_occupancy_is_a_verdict_when_the_metal_records_are_readable(
    tmp_path: Path,
) -> None:
    """Readable alternates get a real verdict either way.

    The blank above must be reserved for the unreadable case, so the same
    fixture with its deposited occupancies intact has to answer ``True``, and
    a metal with no alternates at all has to answer ``False``.
    """
    overfull = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    overfull.add_conformers(
        overfull.residues[0], [("A", 0.46, {}), ("B", 0.55, {})], atom_names=["ZN"]
    )
    overfull_path = overfull.write_pdb(tmp_path / "overfull.pdb")
    overfull_context = sa.load_structure("test", overfull_path)
    overfull_summary = _summary_for(overfull_context, _one_metal(overfull_context))

    single = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    single_path = single.write_pdb(tmp_path / "single.pdb")
    single_context = sa.load_structure("test", single_path)
    single_summary = _summary_for(single_context, _one_metal(single_context))

    assert overfull_summary["metal_overfull_occupancy"] is True
    assert single_summary["metal_overfull_occupancy"] is False


def test_a_donor_without_a_usable_b_factor_is_left_out_of_the_median(
    tmp_path: Path,
) -> None:
    """The donor B count is the donors that had one, not the contacts.

    A non-finite deposited B factor cannot enter a median, so the count falls
    below ``candidate_contact_count`` and the ratio still describes only the
    donors that were measured.
    """
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)])
    path = builder.write_pdb(tmp_path / "donor_b.pdb")
    context = sa.load_structure("test", path)
    water = next(
        atom
        for atom in context.contact_atoms
        if atom.is_water and atom.atom_name == "O"
    )
    water.gemmi_atom.b_iso = NAN
    analysis = helpers.analyze_bonds(path, structure=context)
    summary = analysis.summary

    assert summary["candidate_contact_count"] == 2
    assert summary["donor_b_iso_count"] == 1
    # The one remaining donor is its own median, and both B factors are 20.0.
    assert summary["donor_median_b_iso"] == approx(20.0)
    assert summary["metal_donor_b_ratio"] == approx(1.0)


def test_the_donor_b_comparison_ignores_non_finite_donors_entirely() -> None:
    """Directly: a NaN donor changes neither the count nor the median."""
    metal = helpers.atom_site("ZN", atom_name="ZN", residue_name="ZN")
    metal.gemmi_atom.b_iso = 30.0
    measured: list[Candidate] = []
    for index, b_iso in enumerate((10.0, 20.0, NAN)):
        neighbor = helpers.atom_site(
            "O", atom_name="O", residue_name="HOH", is_water=True, source_order=index
        )
        neighbor.gemmi_atom.b_iso = b_iso
        measured.append(
            Candidate(
                neighbor=neighbor,
                image=ContactImage.explicit(metal, neighbor),
                candidate_sources=set(),
            )
        )

    factors = _donor_b_factors(metal, measured)
    finite_only = _donor_b_factors(metal, measured[:2])

    assert factors.donor_count == 2
    assert factors.donor_median_b_iso == approx(15.0)
    assert factors == finite_only


def test_a_metal_with_no_usable_coordinates_abstains_without_a_symmetry_search(
    tmp_path: Path,
) -> None:
    """No cell means ``image_contacts`` is ``None``: the scope stays blank.

    ``unassessable_site_summary`` reports zero contacts wherever the search
    was possible, so the two structures differ only in whether the search
    could run, and only the searchable one may claim ``none``.
    """
    without_cell = simple_metal_site("ZN", [("HIS", "NE2", 2.03)], cell=None)
    no_cell_path = without_cell.write_pdb(tmp_path / "nocell.pdb")
    no_cell_context = sa.load_structure("test", no_cell_path)

    with_cell = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    cell_path = with_cell.write_pdb(tmp_path / "cell.pdb")
    cell_context = sa.load_structure("test", cell_path)

    summaries = {
        name: unassessable_site_summary(
            _one_metal(context),
            context,
            DpiComponents(
                dpi=NAN,
                resolution=NAN,
                reason_code="",
                r_free=NAN,
                reflection_count=NAN,
                asu_volume=NAN,
            ),
            EntryModelStatistics.from_structure(context),
            MetalProximity.unavailable(),
            MetalSpecialPosition.unavailable(),
        )
        for name, context in (("nocell", no_cell_context), ("cell", cell_context))
    }

    assert no_cell_context.symmetry.search_available is False
    assert summaries["nocell"]["generated_contact_scope"] == ""
    assert summaries["nocell"]["coordination_depends_on_strict_ncs"] == ""
    assert math.isnan(summaries["nocell"]["image_inclusive_contact_count"])
    assert math.isnan(summaries["nocell"]["symmetry_contact_count"])

    assert cell_context.symmetry.search_available is True
    assert summaries["cell"]["generated_contact_scope"] == ContactScope.NONE
    assert summaries["cell"]["coordination_depends_on_strict_ncs"] is False
    assert summaries["cell"]["image_inclusive_contact_count"] == 0
    assert summaries["cell"]["symmetry_contact_count"] == 0

    for summary in summaries.values():
        assert summary["explicit_contact_count"] == 0
        assert summary["geometry_not_assessed_reason"] == (
            ReasonCode.NON_FINITE_METAL_COORDINATES
        )


def test_the_searched_but_empty_scope_is_a_vocabulary_member(
    tmp_path: Path,
) -> None:
    """``none`` is published from ``codes.py``, not spelled at the call site.

    The value is documented and consumed like every other scope, so it has to
    come from the vocabulary a consumer reads.
    """
    builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03), ("HOH", "O", 2.09)])
    path = builder.write_pdb(tmp_path / "explicit_only.pdb")
    scope = helpers.analyze_bonds(path).summary["generated_contact_scope"]

    assert scope == "none"
    assert scope is ContactScope.NONE


def test_the_image_counts_agree_with_the_scopes_of_the_published_rows(
    tmp_path: Path,
) -> None:
    """Each generated-image count is the count of rows in that scope.

    The four counts share one accumulator, so they are checked together
    against the bond rows, which carry each contact's scope independently.
    """
    builder = StructureBuilder(cell=_SMALL_CELL, spacegroup="P 1")
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (0.5, 2.59, 0.5), chain="B")
    builder.add_water(102, (18.41, 0.5, 0.5), chain="B")
    path = builder.write_pdb(tmp_path / "mixed_scope.pdb")
    analysis = helpers.analyze_bonds(path)
    summary = analysis.summary
    scopes = [row["contact_scope"] for row in analysis.bond_rows]

    assert sorted(scopes) == ["crystallographic", "explicit"]
    assert summary["symmetry_contact_count"] == sum(
        scope != ContactScope.EXPLICIT for scope in scopes
    )
    assert summary["crystallographic_contact_count"] == sum(
        scope
        in (
            ContactScope.CRYSTALLOGRAPHIC,
            ContactScope.STRICT_NCS_AND_CRYSTALLOGRAPHIC,
        )
        for scope in scopes
    )
    assert summary["strict_ncs_contact_count"] == sum(
        scope in (ContactScope.STRICT_NCS, ContactScope.STRICT_NCS_AND_CRYSTALLOGRAPHIC)
        for scope in scopes
    )
    assert summary["combined_ncs_crystallographic_contact_count"] == sum(
        scope == ContactScope.STRICT_NCS_AND_CRYSTALLOGRAPHIC for scope in scopes
    )


_IMAGE_PREDICATES: list[Callable[[ContactImage], bool]] = [
    lambda image: image.symmetry_contact,
    lambda image: image.crystallographic_contact,
    lambda image: image.strict_ncs_contact,
]


@pytest.mark.parametrize("predicate", _IMAGE_PREDICATES)
def test_an_unavailable_search_counts_nan_rather_than_zero(
    predicate: Callable[[ContactImage], bool],
) -> None:
    """No search is not an empty search, whatever is being counted."""
    assert math.isnan(_image_count(None, predicate))
    assert _image_count([], predicate) == 0
