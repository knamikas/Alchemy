"""What a site summary reports about a metal independently of its contacts.

Entry-wide model statistics, the other modeled metals near a site, the
crystallographic site symmetry of the metal, and the component class it
belongs to are all fixed before any contact is discovered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import gemmi

from codes import ParentType
from coordination.policy import (
    NEARBY_METAL_RADIUS,
    SEARCH_EPSILON,
    SPECIAL_POSITION_DEDUP_CUTOFF,
    SPECIAL_POSITION_OCCUPANCY_TOLERANCE,
)
from coordination.schema import metal_site_identifier
from metallocofactors.catalog import cluster_ids, heme_ids
from structure_analysis import (
    NAN,
    AtomKey,
    AtomSite,
    StructureContext,
    count_deposited_ni,
    count_ni,
    position_distance,
    spacegroup_or_none,
)


@dataclass(frozen=True, slots=True)
class EntryModelStatistics:
    """Entry-wide model statistics that every site summary of the entry repeats."""

    occupancy_weighted_atom_count: float
    deposited_occupancy_weighted_atom_count: float
    nonwater_median_b_iso: float

    @classmethod
    def from_structure(cls, structure: StructureContext) -> EntryModelStatistics:
        """Count the analyzed and deposited models and take the entry's median B."""
        return cls(
            occupancy_weighted_atom_count=count_ni(structure),
            deposited_occupancy_weighted_atom_count=count_deposited_ni(structure),
            nonwater_median_b_iso=entry_nonwater_median_b_iso(structure),
        )


@dataclass(frozen=True, slots=True)
class MetalProximity:
    """Other modeled metals near one site, without crystallographic expansion."""

    nearest_distance: float
    nearest_element: str
    nearest_site_id: str
    count_within_6a: int | float

    @classmethod
    def unavailable(cls) -> MetalProximity:
        """The proximity of a metal whose coordinates cannot be searched."""
        return cls(
            nearest_distance=NAN,
            nearest_element="",
            nearest_site_id="",
            count_within_6a=NAN,
        )


@dataclass(frozen=True, slots=True)
class MetalSpecialPosition:
    """Crystallographic site symmetry of one metal and its occupancy expectation."""

    special_position: bool | str
    site_symmetry_order: int | float
    expected_occupancy: float
    occupancy_matches_site_symmetry: bool | str

    @classmethod
    def unavailable(cls) -> MetalSpecialPosition:
        """The site symmetry of a metal that could not be assessed."""
        return cls(
            special_position="",
            site_symmetry_order=NAN,
            expected_occupancy=NAN,
            occupancy_matches_site_symmetry="",
        )


def _note_failure(
    messages: list[str] | None, exc: Exception, site_id: str = ""
) -> None:
    """Record why a special-position evaluation could not be completed."""
    if messages is None:
        return
    scope = f" for {site_id}" if site_id else ""
    messages.append(
        f"special position evaluation failed{scope}: {type(exc).__name__}: {exc}"
    )


def metal_proximity_summaries(
    pdb_id: str, metals: Sequence[AtomSite]
) -> dict[AtomKey, MetalProximity]:
    """Summarize other modeled metals without crystallographic expansion.

    ``metals`` is the canonical analyzed-model selection, which already omits
    explicitly zero-occupancy sites. Non-finite coordinates remain in the result
    with unavailable proximity fields so their site rows keep a fixed schema.
    The nearest neighbor is chosen in one pass over the other searchable metals,
    ordered by ``(distance, source_key)`` so equidistant neighbors resolve to the
    lowest source key. A lone searchable metal reports blank nearest fields with
    a count of zero, which is a measurement rather than a failed search.
    """
    summaries: dict[AtomKey, MetalProximity] = {}
    spatial = [metal for metal in metals if metal.coordinates_valid]
    for metal in metals:
        if not metal.coordinates_valid:
            summaries[metal.source_key] = MetalProximity.unavailable()
            continue
        nearest: AtomSite | None = None
        nearest_distance = NAN
        count_within_6a = 0
        for neighbor in spatial:
            if neighbor.source_key == metal.source_key:
                continue
            distance = position_distance(metal.xyz, neighbor.xyz)
            if distance <= NEARBY_METAL_RADIUS + SEARCH_EPSILON:
                count_within_6a += 1
            if nearest is None or (distance, neighbor.source_key) < (
                nearest_distance,
                nearest.source_key,
            ):
                nearest = neighbor
                nearest_distance = distance
        summaries[metal.source_key] = MetalProximity(
            nearest_distance=NAN if nearest is None else round(nearest_distance, 3),
            nearest_element="" if nearest is None else nearest.element,
            nearest_site_id=(
                "" if nearest is None else metal_site_identifier(pdb_id, nearest)
            ),
            count_within_6a=count_within_6a,
        )
    return summaries


def metal_special_position_summaries(
    structure: StructureContext,
    metals: Sequence[AtomSite],
    messages: list[str] | None = None,
) -> dict[AtomKey, MetalSpecialPosition]:
    """Report crystallographic site symmetry and its occupancy expectation.

    A fresh Gemmi cell is populated with space-group images only.  This keeps
    strict-NCS transforms, which are also present in ``structure.cell`` after
    ``setup_cell_images()``, from being mistaken for crystallographic site
    symmetry.

    A site symmetry order that does not divide the space-group operation count
    is crystallographically impossible: it comes from a metal modeled slightly
    off an axis, where the deduplication cutoff merges only some of the images
    that meet there. Such a site keeps ``special_position=True``, which the
    coincident image directly observed, and blanks the order, the expected
    occupancy, and the occupancy agreement, none of which can be trusted.

    ``messages`` collects the entry-level messages of a caller; when it is
    given, a symmetry evaluation that raises appends the failure to it instead
    of being indistinguishable from an entry with no special positions.
    """
    summaries: dict[AtomKey, MetalSpecialPosition] = {
        metal.source_key: MetalSpecialPosition.unavailable() for metal in metals
    }
    if not structure.symmetry.search_available:
        return summaries

    try:
        spacegroup = spacegroup_or_none(structure.structure)
        if spacegroup is None:
            return summaries
        source_cell = structure.structure.cell
        crystallographic_structure = gemmi.Structure()
        crystallographic_structure.cell = gemmi.UnitCell(
            source_cell.a,
            source_cell.b,
            source_cell.c,
            source_cell.alpha,
            source_cell.beta,
            source_cell.gamma,
        )
        crystallographic_structure.spacegroup_hm = spacegroup.xhm()
        crystallographic_structure.setup_cell_images()
    except Exception as exc:
        _note_failure(messages, exc)
        return summaries

    for metal in metals:
        if not metal.coordinates_valid:
            continue
        try:
            coincident_nonidentity_images = int(
                crystallographic_structure.cell.is_special_position(
                    metal.pos, SPECIAL_POSITION_DEDUP_CUTOFF
                )
            )
        except Exception as exc:
            _note_failure(messages, exc, metal_site_identifier(structure.pdb_id, metal))
            continue
        site_symmetry_order = coincident_nonidentity_images + 1
        operation_count = structure.symmetry.crystallographic_operation_count
        if operation_count > 0 and operation_count % site_symmetry_order != 0:
            summaries[metal.source_key] = MetalSpecialPosition(
                special_position=True,
                site_symmetry_order=NAN,
                expected_occupancy=NAN,
                occupancy_matches_site_symmetry="",
            )
            continue
        expected_occupancy = 1.0 / site_symmetry_order
        occupancy_matches: bool | str = ""
        if metal.occupancy_valid:
            occupancy_matches = math.isclose(
                metal.occupancy,
                expected_occupancy,
                rel_tol=0.0,
                abs_tol=SPECIAL_POSITION_OCCUPANCY_TOLERANCE,
            )
        summaries[metal.source_key] = MetalSpecialPosition(
            special_position=coincident_nonidentity_images > 0,
            site_symmetry_order=site_symmetry_order,
            expected_occupancy=round(expected_occupancy, 6),
            occupancy_matches_site_symmetry=occupancy_matches,
        )
    return summaries


def entry_nonwater_median_b_iso(structure: StructureContext) -> float:
    """Median B for canonical, non-water, non-H modeled atoms in this entry.

    Atoms with a valid occupancy of exactly zero are modeled absence and are
    excluded, as are atoms whose B factor is not finite. An entry with no
    remaining atom has no median and reports it as unavailable.
    """
    values = [
        atom.b_iso
        for atom in structure.contact_atoms
        if not atom.is_water
        and not atom.is_hydrogen
        and not (atom.occupancy_valid and atom.occupancy == 0.0)
        and math.isfinite(atom.b_iso)
    ]
    return round(float(median(values)), 3) if values else NAN


def parent_type(structure: StructureContext, metal: AtomSite) -> ParentType:
    """Classify the component a metal belongs to for the bond rows.

    ``metal`` comes from ``metal_atoms(METAL_ELEMENTS)``, so its element is a
    metal by construction and only the component identity needs deciding.
    ``AtomSite.residue_name`` arrives already whitespace-stripped and
    upper-cased, which is how the catalogued component identifiers are keyed.
    """
    residue_name = metal.residue_name
    if residue_name in cluster_ids():
        return ParentType.CLUSTER
    if residue_name in heme_ids():
        return ParentType.HEME
    residue = structure.residue_for_atom(metal)
    if residue.chemical_atom_site_count == 1:
        return ParentType.ION
    return ParentType.OTHER
