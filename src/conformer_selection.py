"""Choose one conformer per residue for contact searches.

Alternate conformers are compared by their mean valid occupancy, so a residue
is never assembled from atoms of different conformers; blank-altloc atoms are
shared by every conformer. Exact ties fall back to the lowest altloc label,
and duplicate records of one atom are resolved by ``site_is_better``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from structure_model import AtomSite, ResidueSelection

#: Conformer means closer than this are one tie, broken by altloc label.
CONFORMER_MEAN_TIE_TOLERANCE = 1e-12
#: Rendered in ``ResidueSelection.altloc_options`` for a conformer without a mean.
UNAVAILABLE_MEAN_TEXT = "NA"
ALTLOC_OPTION_SEPARATOR = "|"


def _format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return UNAVAILABLE_MEAN_TEXT
    return f"{value:.6g}"


def site_is_better(candidate: AtomSite, current: AtomSite) -> bool:
    """Highest valid occupancy wins; exact ties retain source-file order."""
    if candidate.occupancy_valid != current.occupancy_valid:
        return candidate.occupancy_valid
    if candidate.occupancy_valid and candidate.occupancy != current.occupancy:
        return candidate.occupancy > current.occupancy
    return candidate.source_order < current.source_order


def select_residue(atoms: Sequence[AtomSite]) -> ResidueSelection:
    """Select one deterministic conformer for a residue's contact atoms."""
    first = atoms[0]
    named: dict[str, list[AtomSite]] = defaultdict(list)
    blank: list[AtomSite] = []
    for atom in atoms:
        (named[atom.altloc] if atom.altloc else blank).append(atom)

    option_means: dict[str, float | None] = {}
    for label in sorted(named):
        valid = [atom.occupancy for atom in named[label] if atom.occupancy_valid]
        option_means[label] = sum(valid) / len(valid) if valid else None

    fallback = False
    selected_altloc = ""
    selected_mean: float | None
    if option_means:
        valid_options = [
            (mean, label)
            for label, mean in option_means.items()
            if mean is not None and math.isfinite(mean)
        ]
        if valid_options:
            best_mean = max(mean for mean, _ in valid_options)
            selected_altloc = min(
                label
                for mean, label in valid_options
                if math.isclose(
                    mean,
                    best_mean,
                    rel_tol=0.0,
                    abs_tol=CONFORMER_MEAN_TIE_TOLERANCE,
                )
            )
            selected_mean = option_means[selected_altloc]
        else:
            selected_altloc = min(option_means)
            selected_mean = None
            fallback = True
    else:
        valid_blank = [atom.occupancy for atom in blank if atom.occupancy_valid]
        selected_mean = sum(valid_blank) / len(valid_blank) if valid_blank else None

    candidates = list(blank)
    if selected_altloc:
        candidates.extend(named[selected_altloc])

    # A selected named atom supersedes a malformed blank record of the same
    # atom name; any remaining duplicate name is resolved by occupancy.
    by_name: dict[str, list[AtomSite]] = defaultdict(list)
    for atom in candidates:
        by_name[atom.atom_name].append(atom)
    contact_atoms: list[AtomSite] = []
    selected_over_blank = 0
    malformed_duplicates = 0
    for atom_name in sorted(
        by_name, key=lambda name: min(atom.source_order for atom in by_name[name])
    ):
        group = by_name[atom_name]
        selected_named = [
            atom for atom in group if selected_altloc and atom.altloc == selected_altloc
        ]
        pool = selected_named or group
        if selected_named:
            selected_over_blank += sum(1 for atom in group if not atom.altloc)
        if len(pool) > 1:
            malformed_duplicates += len(pool) - 1
        winner = pool[0]
        for atom in pool[1:]:
            if site_is_better(atom, winner):
                winner = atom
        contact_atoms.append(winner)
    contact_atoms.sort(key=lambda atom: atom.source_order)

    options = ALTLOC_OPTION_SEPARATOR.join(
        f"{label}:{_format_number(option_means[label])}"
        for label in sorted(option_means)
    )
    chemical_sites = len({atom.chemical_site_identity for atom in atoms})
    return ResidueSelection(
        identity=first.identity,
        is_water=first.is_water,
        source_atoms=tuple(sorted(atoms, key=lambda atom: atom.source_order)),
        contact_atoms=tuple(contact_atoms),
        selected_altloc=selected_altloc,
        selected_conformer_mean_occupancy=selected_mean,
        altloc_options=options,
        alternative_conformers_present=bool(option_means),
        altloc_selection_fallback=fallback,
        selected_over_blank_duplicate_count=selected_over_blank,
        malformed_duplicate_atom_name_count=malformed_duplicates,
        chemical_atom_site_count=chemical_sites,
    )
