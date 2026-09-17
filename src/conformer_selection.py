"""Choose one conformer per residue for contact searches.

Alternate conformers are compared by their mean valid occupancy, so a residue
is never assembled from atoms of different conformers; blank-altloc atoms are
shared by every conformer. Means within ``CONFORMER_MEAN_TIE_TOLERANCE`` of
the best are one tie, broken by the lowest altloc label, and duplicate records
of one atom are resolved by ``site_is_better``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

from structure_model import AtomSite, ResidueSelection

#: Conformer means closer than this are one tie, broken by altloc label.
CONFORMER_MEAN_TIE_TOLERANCE = 1e-12
#: Rendered in ``ResidueSelection.altloc_options`` for a conformer without a mean.
UNAVAILABLE_MEAN_TEXT = "NA"
ALTLOC_OPTION_SEPARATOR = "|"


def _format_number(value: float | None) -> str:
    if value is None:
        return UNAVAILABLE_MEAN_TEXT
    return f"{value:.6g}"


def _mean_valid_occupancy(atoms: Iterable[AtomSite]) -> float | None:
    """Mean of the valid occupancies, or ``None`` when no occupancy is valid."""
    valid = [atom.occupancy for atom in atoms if atom.occupancy_valid]
    return sum(valid) / len(valid) if valid else None


def site_is_better(candidate: AtomSite, current: AtomSite) -> bool:
    """Highest valid occupancy wins; exact ties retain source-file order."""
    if candidate.occupancy_valid != current.occupancy_valid:
        return candidate.occupancy_valid
    if candidate.occupancy_valid and candidate.occupancy != current.occupancy:
        return candidate.occupancy > current.occupancy
    return candidate.source_order < current.source_order


def select_residue(atoms: Sequence[AtomSite]) -> ResidueSelection:
    """Select one deterministic conformer for a residue's contact atoms.

    ``atoms`` are every deposited record of one residue, in any order; the
    result lists them by source order.

    Raises:
        ValueError: If ``atoms`` is empty.
    """
    if not atoms:
        raise ValueError("select_residue needs at least one atom record")
    source_atoms = tuple(sorted(atoms, key=lambda atom: atom.source_order))
    first = source_atoms[0]
    named: dict[str, list[AtomSite]] = defaultdict(list)
    for atom in source_atoms:
        if atom.altloc:
            named[atom.altloc].append(atom)

    option_means = {
        label: _mean_valid_occupancy(named[label]) for label in sorted(named)
    }
    valid_options = [
        (mean, label) for label, mean in option_means.items() if mean is not None
    ]

    # A lone label is the only conformer, so choosing it is not a fallback.
    fallback = False
    selected_altloc = ""
    if valid_options:
        best_mean = max(mean for mean, _ in valid_options)
        selected_altloc = min(
            label
            for mean, label in valid_options
            if math.isclose(
                mean, best_mean, rel_tol=0.0, abs_tol=CONFORMER_MEAN_TIE_TOLERANCE
            )
        )
    elif option_means:
        selected_altloc = min(option_means)
        fallback = len(option_means) > 1
    # Blank atoms are the whole conformer when no label is present, so this is
    # the mean over the atoms carrying ``selected_altloc`` in either case.
    selected_mean = _mean_valid_occupancy(
        atom for atom in source_atoms if atom.altloc == selected_altloc
    )

    # A selected named atom supersedes a malformed blank record of the same
    # atom name; any remaining duplicate name is resolved by occupancy.
    by_name: dict[str, list[AtomSite]] = defaultdict(list)
    for atom in source_atoms:
        if not atom.altloc or atom.altloc == selected_altloc:
            by_name[atom.atom_name].append(atom)
    contact_atoms: list[AtomSite] = []
    malformed_duplicates = 0
    for group in by_name.values():
        malformed_duplicates += len(group) - 1
        pool = [atom for atom in group if atom.altloc] or group
        winner = pool[0]
        for atom in pool[1:]:
            if site_is_better(atom, winner):
                winner = atom
        contact_atoms.append(winner)
    # A group's winner may follow a later group's first record.
    contact_atoms.sort(key=lambda atom: atom.source_order)

    options = ALTLOC_OPTION_SEPARATOR.join(
        f"{label}:{_format_number(mean)}" for label, mean in option_means.items()
    )
    # Records without a validated element carry no chemical identity to count.
    chemical_sites = len(
        {atom.chemical_site_identity for atom in source_atoms if atom.element_known}
    )
    return ResidueSelection(
        identity=first.identity,
        is_water=first.is_water,
        source_atoms=source_atoms,
        contact_atoms=tuple(contact_atoms),
        selected_altloc=selected_altloc,
        selected_conformer_mean_occupancy=selected_mean,
        altloc_options=options,
        alternative_conformers_present=len(option_means) > 1,
        altloc_selection_fallback=fallback,
        malformed_duplicate_atom_name_count=malformed_duplicates,
        chemical_atom_site_count=chemical_sites,
    )
