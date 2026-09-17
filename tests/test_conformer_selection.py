"""Behavioural tests for ``src/conformer_selection.py``.

Every residue is built from explicitly constructed ``AtomSite`` records, so
malformed occupancies the structure builder cannot express are covered too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import helpers
import pytest
from helpers import approx

import conformer_selection
from structure_model import AtomSite, ResidueSelection


def _site(
    atom_name: str,
    *,
    altloc: str = "",
    occupancy: float = 1.0,
    source_order: int = 0,
    element: str = "C",
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    **overrides: Any,
) -> AtomSite:
    """One atom of ``HIS A 10`` built directly.

    ``residue_index``, ``occupancy_valid`` and ``occupancy_status`` pass through
    as overrides.
    """
    return helpers.atom_site(
        element,
        atom_name=atom_name,
        residue_name="HIS",
        occupancy=occupancy,
        altloc=altloc,
        pos=pos,
        source_order=source_order,
        residue_number=10,
        resnum="10",
        **overrides,
    )


def _altloc_option_map(selection: ResidueSelection) -> dict[str, str]:
    """Parse ``ResidueSelection.altloc_options`` into ``{label: text}``."""
    if not selection.altloc_options:
        return {}
    return dict(part.split(":", 1) for part in selection.altloc_options.split("|"))


def _contact(selection: ResidueSelection, atom_name: str) -> AtomSite:
    matches = [atom for atom in selection.contact_atoms if atom.atom_name == atom_name]
    assert len(matches) == 1, (
        f"expected exactly one contact atom named {atom_name!r}, "
        f"got {[atom.altloc for atom in matches]}"
    )
    return matches[0]


@pytest.mark.parametrize(
    "candidate, current, expected, why",
    [
        (
            _site("O", occupancy=0.1, source_order=5),
            _site("O", occupancy=float("nan"), source_order=0),
            True,
            "a valid occupancy always beats an invalid one",
        ),
        (
            _site("O", occupancy=float("nan"), source_order=0),
            _site("O", occupancy=0.1, source_order=5),
            False,
            "an invalid occupancy never beats a valid one",
        ),
        (
            _site("O", occupancy=0.9, source_order=5),
            _site("O", occupancy=0.4, source_order=0),
            True,
            "higher valid occupancy wins regardless of file order",
        ),
        (
            _site("O", occupancy=0.4, source_order=0),
            _site("O", occupancy=0.9, source_order=5),
            False,
            "lower valid occupancy loses regardless of file order",
        ),
        (
            _site("O", occupancy=0.5, source_order=1),
            _site("O", occupancy=0.5, source_order=3),
            True,
            "an exact tie keeps the earlier source record",
        ),
        (
            _site("O", occupancy=0.5, source_order=3),
            _site("O", occupancy=0.5, source_order=1),
            False,
            "an exact tie does not displace the earlier source record",
        ),
        (
            _site("O", occupancy=2.0, source_order=1),
            _site("O", occupancy=5.0, source_order=3),
            True,
            "two invalid records fall back to source order, not magnitude",
        ),
    ],
)
def test_site_is_better_ranks_validity_then_occupancy_then_order(
    candidate: AtomSite, current: AtomSite, expected: bool, why: str
) -> None:
    assert conformer_selection.site_is_better(candidate, current) is expected, why


def test_select_residue_without_alternates_shares_every_blank_atom() -> None:
    atoms = [
        _site("N", occupancy=1.0, source_order=0),
        _site("CA", occupancy=0.8, source_order=1),
        _site("C", occupancy=0.6, source_order=2),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == ""
    assert selection.alternative_conformers_present is False
    assert selection.altloc_selection_fallback is False
    assert [atom.atom_name for atom in selection.contact_atoms] == ["N", "CA", "C"]
    assert selection.selected_conformer_mean_occupancy == approx((1.0 + 0.8 + 0.6) / 3)


def test_select_residue_shares_blank_atoms_with_the_selected_conformer() -> None:
    atoms = [
        _site("N", occupancy=1.0, source_order=0),
        _site("CA", occupancy=1.0, source_order=1),
        _site("NE2", altloc="A", occupancy=0.3, source_order=2),
        _site("NE2", altloc="B", occupancy=0.7, source_order=3),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert [(atom.atom_name, atom.altloc) for atom in selection.contact_atoms] == [
        ("N", ""),
        ("CA", ""),
        ("NE2", "B"),
    ]


def test_select_residue_uses_the_conformer_mean_not_the_single_best_atom() -> None:
    """Selection compares mean occupancy per conformer, per the README.

    Conformer A holds the highest single atom (0.9) but the lower mean (0.5);
    per-atom maxima would build a chimeric residue.
    """
    atoms = [
        _site("CB", altloc="A", occupancy=0.9, source_order=0),
        _site("CG", altloc="A", occupancy=0.1, source_order=1),
        _site("CB", altloc="B", occupancy=0.6, source_order=2),
        _site("CG", altloc="B", occupancy=0.6, source_order=3),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert selection.selected_conformer_mean_occupancy == approx(0.6)
    assert {atom.altloc for atom in selection.contact_atoms} == {"B"}


@pytest.mark.parametrize("order", ["ab", "ba"])
def test_select_residue_breaks_occupancy_ties_by_altloc_label(order: str) -> None:
    a_atoms = [
        _site("CB", altloc="A", occupancy=0.5, source_order=0),
        _site("CG", altloc="A", occupancy=0.5, source_order=1),
    ]
    b_atoms = [
        _site("CB", altloc="B", occupancy=0.5, source_order=2),
        _site("CG", altloc="B", occupancy=0.5, source_order=3),
    ]
    atoms = a_atoms + b_atoms if order == "ab" else b_atoms + a_atoms

    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == "A"
    assert {atom.altloc for atom in selection.contact_atoms} == {"A"}


def test_select_residue_averages_only_valid_occupancies() -> None:
    """Verify conformer selection averages only valid occupancies.

    Conformer A is 0.4 plus an out-of-range 5.0: averaging raw values gives 2.7
    and selects A, while the valid subset gives 0.4 and selects B.
    """
    atoms = [
        _site("CB", altloc="A", occupancy=0.4, source_order=0),
        _site("CG", altloc="A", occupancy=5.0, source_order=1),
        _site("CB", altloc="B", occupancy=0.6, source_order=2),
        _site("CG", altloc="B", occupancy=0.6, source_order=3),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert selection.selected_conformer_mean_occupancy == approx(0.6)
    assert float(_altloc_option_map(selection)["A"]) == approx(0.4)


def test_select_residue_records_every_available_alternative() -> None:
    atoms = [
        _site("CB", altloc="A", occupancy=0.25, source_order=0),
        _site("CB", altloc="B", occupancy=0.35, source_order=1),
        _site("CB", altloc="C", occupancy=0.40, source_order=2),
    ]
    selection = conformer_selection.select_residue(atoms)

    options = _altloc_option_map(selection)
    assert set(options) == {"A", "B", "C"}
    assert [float(options[label]) for label in ("A", "B", "C")] == approx(
        [0.25, 0.35, 0.40]
    )
    assert selection.selected_altloc == "C"


def test_select_residue_keeps_one_coherent_conformer_across_all_atoms() -> None:
    """Mixing conformers would hand the contact search an invented residue."""
    names = ("CB", "CG", "ND1", "CD2", "CE1", "NE2")
    atoms: list[AtomSite] = []
    for index, name in enumerate(names):
        atoms.append(_site(name, altloc="A", occupancy=0.45, source_order=2 * index))
        atoms.append(
            _site(name, altloc="B", occupancy=0.55, source_order=2 * index + 1)
        )
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == "B"
    assert {atom.altloc for atom in selection.contact_atoms} == {"B"}
    assert [atom.atom_name for atom in selection.contact_atoms] == list(names)
    assert selection.chemical_atom_site_count == len(names)
    assert len(selection.source_atoms) == 2 * len(names)


def test_select_residue_falls_back_when_no_conformer_has_valid_occupancy() -> None:
    atoms = [
        _site("CB", altloc="B", occupancy=2.5, source_order=0),
        _site("CB", altloc="A", occupancy=float("nan"), source_order=1),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.altloc_selection_fallback is True
    assert selection.selected_altloc == "A"  # lowest label, not file order
    assert selection.selected_conformer_mean_occupancy is None
    assert _altloc_option_map(selection) == {"A": "NA", "B": "NA"}
    assert {atom.altloc for atom in selection.contact_atoms} == {"A"}


def test_select_residue_treats_a_lone_altloc_label_as_no_alternative() -> None:
    """A side chain deposited only as altloc A offers nothing to choose from."""
    atoms = [
        _site("CA", occupancy=1.0, source_order=0),
        _site("CB", altloc="A", occupancy=5.0, source_order=1),
        _site("CG", altloc="A", occupancy=5.0, source_order=2),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_altloc == "A"
    assert selection.alternative_conformers_present is False
    assert selection.altloc_selection_fallback is False
    assert selection.selected_conformer_mean_occupancy is None
    assert _altloc_option_map(selection) == {"A": "NA"}
    assert [(atom.atom_name, atom.altloc) for atom in selection.contact_atoms] == [
        ("CA", ""),
        ("CB", "A"),
        ("CG", "A"),
    ]


def test_select_residue_reports_the_selected_conformer_mean_only() -> None:
    """The mean covers the atoms carrying the selected label, not shared ones."""
    atoms = [
        _site("N", occupancy=1.0, source_order=0),
        _site("NE2", altloc="A", occupancy=0.3, source_order=1),
        _site("NE2", altloc="B", occupancy=0.7, source_order=2),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.selected_conformer_mean_occupancy == approx(0.7)


def test_select_residue_counts_chemical_sites_with_a_validated_element_only() -> None:
    """A duplicate record with an unknown element must not make an ion a cluster."""
    atoms = [
        _site("ZN", element="ZN", occupancy=1.0, source_order=0),
        _site("ZN", element="X", element_known=False, occupancy=1.0, source_order=1),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert selection.chemical_atom_site_count == 1
    assert len(selection.source_atoms) == 2


def test_select_residue_rejects_an_empty_residue() -> None:
    with pytest.raises(ValueError, match="at least one atom"):
        conformer_selection.select_residue([])


def test_select_residue_prefers_a_selected_conformer_over_a_blank_duplicate() -> None:
    atoms = [
        _site("NE2", altloc="", occupancy=1.0, source_order=0),
        _site("NE2", altloc="A", occupancy=0.4, source_order=1),
        _site("NE2", altloc="B", occupancy=0.6, source_order=2),
    ]
    selection = conformer_selection.select_residue(atoms)

    contact = _contact(selection, "NE2")
    assert contact.altloc == "B"
    assert selection.malformed_duplicate_atom_name_count == 1


def test_select_residue_counts_every_shadowed_blank_duplicate() -> None:
    """Two blank records under a selected named atom are two extra records."""
    atoms = [
        _site("NE2", altloc="", occupancy=1.0, source_order=0),
        _site("NE2", altloc="", occupancy=1.0, source_order=1),
        _site("NE2", altloc="B", occupancy=0.6, source_order=2),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert _contact(selection, "NE2").altloc == "B"
    assert selection.malformed_duplicate_atom_name_count == 2


def test_select_residue_resolves_repeated_atom_names_by_occupancy() -> None:
    atoms = [
        _site("NE2", altloc="A", occupancy=0.4, source_order=0, pos=(1.0, 0.0, 0.0)),
        _site("NE2", altloc="A", occupancy=0.9, source_order=1, pos=(2.0, 0.0, 0.0)),
    ]
    selection = conformer_selection.select_residue(atoms)

    contact = _contact(selection, "NE2")
    assert contact.occupancy == approx(0.9)
    assert contact.x == approx(2.0)
    assert selection.malformed_duplicate_atom_name_count == 1


def test_select_residue_orders_contact_atoms_by_source_order() -> None:
    """Deposited order keeps the downstream atom indices stable."""
    atoms = [
        _site("C", occupancy=1.0, source_order=7),
        _site("N", occupancy=1.0, source_order=2),
        _site("CA", occupancy=1.0, source_order=5),
    ]
    selection = conformer_selection.select_residue(atoms)

    assert [atom.atom_name for atom in selection.contact_atoms] == ["N", "CA", "C"]
    assert [atom.source_order for atom in selection.source_atoms] == [2, 5, 7]
