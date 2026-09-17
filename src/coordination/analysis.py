"""Identify metal-donor contacts and assess their bond distances.

Orchestrate the coordination stages for one entry: ``coordination.candidates``
discovers donor-like images within 4 A and merges them with source
struct_conn and LINK declarations, ``coordination.eligibility`` assigns
first-sphere contacts, ``coordination.geometry`` scores their distances, and
``coordination.site_summary`` condenses each site. Image-inclusive results are
primary wherever symmetry metadata is available. See docs/method.md for the
scientific policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import gemmi

from codes import ReasonCode
from coordination.candidates import (
    AtomKey as AtomKey,
    collect_proximal_candidates,
    merge_candidates,
)
from coordination.contact_record import Candidate
from coordination.declared_connections import collect_declared_candidates
from coordination.density_zscores import DensityZScoreIndex
from coordination.dpi import DpiComponents, DpiInputs, calculate_dpi_components
from coordination.eligibility import (
    annotate_donor_policy,
    current_contacts_from_candidates,
)
from coordination.geometry import annotate_contacts, annotate_multi_donor_groups
from coordination.policy import CANDIDATE_SEARCH_RADIUS, SEARCH_EPSILON
from coordination.schema import (
    BondRow,
    CandidateRow,
    bond_row,
    candidate_row,
    contact_identifier,
)
from coordination.site_environment import (
    EntryModelStatistics,
    MetalProximity,
    MetalSpecialPosition,
    metal_proximity_summaries,
    metal_special_position_summaries,
    parent_type,
)
from coordination.site_summary import (
    SiteSummary,
    context_warning_reasons,
    site_summary,
    unassessable_site_summary,
)
from metal_elements import METAL_ELEMENTS
from output_rows import MetalStatsRow
from structure_analysis import AtomSite, StructureContext, load_structure


@dataclass(frozen=True, slots=True)
class _MetalAnalysisResult:
    """Everything the analysis of one metal site contributes to the entry."""

    bond_rows: list[BondRow]
    candidate_rows: list[CandidateRow]
    summary: SiteSummary
    unsupported_pairs: set[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class _EntryContext:
    """Entry-level inputs that every metal site of one entry shares."""

    pdb_id: str
    structure: StructureContext
    explicit_search: gemmi.NeighborSearch
    image_search: gemmi.NeighborSearch | None
    dpi_components: DpiComponents
    density_z_scores: DensityZScoreIndex
    model_statistics: EntryModelStatistics
    metal_proximity: Mapping[AtomKey, MetalProximity]
    metal_special_positions: Mapping[AtomKey, MetalSpecialPosition]


def _assess_scope(
    entry: _EntryContext,
    metal: AtomSite,
    search: gemmi.NeighborSearch,
    declarations: Sequence[Candidate],
    include_symmetry: bool,
) -> tuple[list[Candidate], list[Candidate], set[tuple[str, str]]]:
    """Discover, filter, and score the contacts of one metal in one scope.

    Returns ``(candidates, contacts, unsupported_pairs)``: every annotated
    candidate, the subset assigned as contacts with geometry and multi-donor
    results attached, and the metal-donor element pairs no reference covers.
    """
    candidates = merge_candidates(
        collect_proximal_candidates(entry.structure, search, metal, include_symmetry),
        declarations,
    )
    annotate_donor_policy(entry.structure, candidates)
    contacts, unsupported_pairs = current_contacts_from_candidates(candidates, metal)
    annotate_contacts(contacts, metal.element, entry.dpi_components.dpi)
    annotate_multi_donor_groups(contacts)
    return candidates, contacts, unsupported_pairs


def _analyze_metal_site(
    entry: _EntryContext,
    metal: AtomSite,
    metal_declarations: Sequence[Candidate],
) -> _MetalAnalysisResult:
    """Assess one metal in both scopes and serialize its rows and summary."""
    pdb_id = entry.pdb_id
    structure = entry.structure
    dpi_components = entry.dpi_components
    explicit_declarations = [
        candidate
        for candidate in metal_declarations
        if not candidate.image.symmetry_contact
    ]
    explicit_candidates, explicit_contacts, unsupported_pairs = _assess_scope(
        entry, metal, entry.explicit_search, explicit_declarations, False
    )

    image_candidates: list[Candidate] | None = None
    image_contacts: list[Candidate] | None = None
    if entry.image_search is not None:
        image_candidates, image_contacts, image_unsupported = _assess_scope(
            entry, metal, entry.image_search, metal_declarations, True
        )
        unsupported_pairs.update(image_unsupported)

    primary_contacts = (
        image_contacts if image_contacts is not None else explicit_contacts
    )
    primary_candidates = (
        image_candidates if image_candidates is not None else explicit_candidates
    )
    summary = site_summary(
        metal,
        structure,
        explicit_contacts,
        image_contacts,
        primary_candidates,
        dpi_components,
        entry.model_statistics,
        entry.metal_proximity[metal.source_key],
        entry.metal_special_positions[metal.source_key],
    )

    sigma = entry.density_z_scores.lookup(
        metal.source_key, (metal.residue_name, metal.chain_id, metal.resnum)
    )
    metal_parent_type = parent_type(structure, metal)
    assigned_contact_ids = {
        contact_identifier(pdb_id, metal, contact) for contact in primary_contacts
    }
    return _MetalAnalysisResult(
        bond_rows=[
            bond_row(
                pdb_id,
                structure,
                metal,
                contact,
                dpi_components.dpi,
                dpi_components.resolution,
                sigma,
                metal_parent_type,
                context_warning_reasons(contact, multi_donor=contact.multi_donor()),
            )
            for contact in primary_contacts
        ],
        candidate_rows=[
            candidate_row(
                pdb_id,
                structure,
                metal,
                candidate,
                assigned_as_bond=(
                    contact_identifier(pdb_id, metal, candidate) in assigned_contact_ids
                ),
                context_reasons=context_warning_reasons(
                    candidate, include_proximal=True
                ),
            )
            for candidate in primary_candidates
        ],
        summary=summary,
        unsupported_pairs=unsupported_pairs,
    )


@dataclass(slots=True)
class BondAnalysisMetadata:
    """Collect non-row outcomes from coordination analysis."""

    partial_reason_codes: list[str]
    warning_codes: list[str]
    messages: list[str]


@dataclass(frozen=True, slots=True)
class BondAnalysisResult:
    """Collect coordination rows, site summaries, and run metadata."""

    bond_rows: list[BondRow]
    candidate_rows: list[CandidateRow]
    site_summaries: dict[AtomKey, SiteSummary]
    metadata: BondAnalysisMetadata


def _declared_candidates_by_metal(
    structure: StructureContext,
    connection_path: str,
    spatial_metals: Sequence[AtomSite],
    metadata: BondAnalysisMetadata,
) -> dict[AtomKey, list[Candidate]]:
    """Resolve deposited connections, recording unresolved ones on ``metadata``."""
    declared_candidates, declared_issues, declared_warnings = (
        collect_declared_candidates(structure, connection_path, spatial_metals)
    )
    if declared_issues:
        metadata.partial_reason_codes.append(
            ReasonCode.DECLARED_CONNECTION_RESOLUTION_INCOMPLETE
        )
        metadata.messages.extend(declared_issues)
    metadata.warning_codes.extend(declared_warnings)
    declared_by_metal: dict[AtomKey, list[Candidate]] = {}
    for candidate in declared_candidates:
        # Every declaration-derived candidate carries the metal it was resolved
        # against; only proximity discovery leaves the field unset.
        declared_by_metal.setdefault(
            cast(AtomSite, candidate.metal).source_key, []
        ).append(candidate)
    return declared_by_metal


def _entry_context(
    pdb_id: str,
    structure: StructureContext,
    metals: Sequence[AtomSite],
    stats_rows: Sequence[MetalStatsRow],
    header: Sequence[str] | None,
    dpi_inputs: DpiInputs,
    metadata: BondAnalysisMetadata,
) -> _EntryContext:
    """Compute what every site shares, recording entry-level limitations."""
    dpi_components = calculate_dpi_components(structure, dpi_inputs)
    if dpi_components.reason_code:
        metadata.partial_reason_codes.append(dpi_components.reason_code)
        metadata.messages.append(f"DPI unavailable: {dpi_components.reason_code}")
    if not structure.symmetry.search_available:
        metadata.partial_reason_codes.append(ReasonCode.SYMMETRY_SEARCH_UNAVAILABLE)
        metadata.messages.append(
            "symmetry search unavailable: "
            + (structure.symmetry.search_failure_reason or "unknown reason")
        )
    image_search = None
    if structure.symmetry.search_available:
        image_search = structure.make_neighbor_search(
            CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON,
            include_symmetry=True,
            positive_occupancy_only=True,
        )
    return _EntryContext(
        pdb_id=pdb_id,
        structure=structure,
        explicit_search=structure.make_neighbor_search(
            CANDIDATE_SEARCH_RADIUS + SEARCH_EPSILON,
            include_symmetry=False,
            positive_occupancy_only=True,
        ),
        image_search=image_search,
        dpi_components=dpi_components,
        density_z_scores=DensityZScoreIndex.from_stats_rows(stats_rows, header),
        model_statistics=EntryModelStatistics.from_structure(structure),
        metal_proximity=metal_proximity_summaries(pdb_id, metals),
        metal_special_positions=metal_special_position_summaries(structure, metals),
    )


def _note_unsupported_pairs(
    metadata: BondAnalysisMetadata, unsupported_pairs: set[tuple[str, str]]
) -> None:
    """Record metal-donor element pairs the reference table cannot score."""
    if not unsupported_pairs:
        return
    metadata.partial_reason_codes.append(ReasonCode.MISSING_FIRST_SPHERE_REFERENCE)
    pairs = ", ".join(
        f"{metal_element}-{donor_element}"
        for metal_element, donor_element in sorted(unsupported_pairs)
    )
    metadata.messages.append(f"first-sphere reference unavailable for {pairs}")


def record_non_finite_metals(
    metadata: BondAnalysisMetadata, non_finite_metals: Sequence[AtomSite]
) -> None:
    """Record selected metals whose coordinates are NaN or infinite.

    No distance-based evidence can be collected for these sites, so the entry
    is partial whether or not the geometry stage runs; the worker records them
    through this same function under ``--no-bonds``.
    """
    if not non_finite_metals:
        return
    metadata.partial_reason_codes.append(ReasonCode.NON_FINITE_METAL_COORDINATES)
    metadata.messages.append(
        "geometry unavailable for selected metal site(s) with non-finite "
        "coordinates: "
        + ", ".join(
            f"{metal.residue_name}/{metal.chain_id or '_'}/{metal.resnum}/"
            f"{metal.atom_name}"
            for metal in non_finite_metals
        )
    )


def run_bond_analysis(
    pdb_id: str,
    pdb_path: str,
    stats_rows: Sequence[MetalStatsRow],
    header: Sequence[str] | None,
    dpi_inputs: DpiInputs,
    structure: StructureContext | None = None,
    connection_path: str | None = None,
) -> BondAnalysisResult:
    """Return contact rows, candidate rows, site summaries, and metadata.

    Bond rows come from first-sphere-eligible candidates and from source
    ``struct_conn``/``LINK`` declarations, which are evaluated even outside the
    4 A discovery radius. Image-inclusive results are primary wherever symmetry
    metadata is available.
    """
    if structure is None:
        structure = load_structure(pdb_id, pdb_path)

    metals_in_model = structure.metal_atoms(METAL_ELEMENTS, canonical=True)
    spatial_metals = [metal for metal in metals_in_model if metal.coordinates_valid]
    non_finite_metals = [
        metal for metal in metals_in_model if not metal.coordinates_valid
    ]
    metadata = BondAnalysisMetadata(
        partial_reason_codes=[],
        warning_codes=list(structure.warning_codes),
        messages=[],
    )
    if not metals_in_model:
        return BondAnalysisResult([], [], {}, metadata)

    declared_by_metal = _declared_candidates_by_metal(
        structure, connection_path or pdb_path, spatial_metals, metadata
    )
    record_non_finite_metals(metadata, non_finite_metals)
    entry = _entry_context(
        pdb_id, structure, metals_in_model, stats_rows, header, dpi_inputs, metadata
    )

    rows: list[BondRow] = []
    candidate_rows: list[CandidateRow] = []
    summaries: dict[AtomKey, SiteSummary] = {}
    for metal in metals_in_model:
        if not metal.coordinates_valid:
            summaries[metal.source_key] = unassessable_site_summary(
                metal,
                structure,
                entry.dpi_components,
                entry.model_statistics,
                entry.metal_proximity[metal.source_key],
                entry.metal_special_positions[metal.source_key],
            )
            continue
        site_result = _analyze_metal_site(
            entry, metal, declared_by_metal.get(metal.source_key, ())
        )
        _note_unsupported_pairs(metadata, site_result.unsupported_pairs)
        rows.extend(site_result.bond_rows)
        candidate_rows.extend(site_result.candidate_rows)
        summaries[metal.source_key] = site_result.summary

    metadata.partial_reason_codes = list(dict.fromkeys(metadata.partial_reason_codes))
    metadata.warning_codes = list(dict.fromkeys(metadata.warning_codes))
    metadata.messages = list(dict.fromkeys(metadata.messages))
    return BondAnalysisResult(rows, candidate_rows, summaries, metadata)
