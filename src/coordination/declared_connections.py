"""Resolve source struct_conn and LINK contacts onto the analysis model.

Match author identities rather than serials, which PDB conversion can change.
Use each residue's selected conformer and retain unresolved declarations as
issues or warnings. See resolve_declared_partners for metal identification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from typing import NamedTuple, Protocol

import gemmi

from codes import CandidateSource, WarningCode
from coordination.contact_record import Candidate, DeclaredConnectionRecord
from coordination.donor_chemistry import AA, DONOR_ELEMENTS
from metal_elements import METAL_ELEMENTS, UNAMBIGUOUS_METAL_COMPONENT_IDS
from structure_analysis import (
    NAN,
    AtomKey,
    AtomSite,
    ContactImage,
    StructureContext,
    blank_if_missing,
)


class PartnerLocator(Protocol):
    """Define the source-model address lookup used during declaration resolution.

    The address is positional-only to match Gemmi's binding.
    """

    def find_cra(
        self, address: gemmi.AtomAddress, /, ignore_segment: bool = ...
    ) -> gemmi.CRA:
        """Find the chain, residue, and atom at a source-model address."""
        ...


def connection_source(path: str | None) -> CandidateSource:
    """Infer declaration provenance from the source coordinate format."""
    lower = str(path or "").lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    return (
        CandidateSource.STRUCT_CONN
        if lower.endswith((".cif", ".mmcif"))
        else CandidateSource.LINK
    )


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", value)).lower()


def analysis_chain_names(
    connection_path: str, declared_structure: gemmi.Structure | None = None
) -> dict[str, str]:
    """Map source chain names onto the analysis model's chain names.

    Replaying Gemmi's ``setup_entities`` and ``shorten_chain_names`` recovers
    the conversion mapping. A source PDB is extracted textually and keeps its
    chain names, so its mapping is empty.

    Pass ``declared_structure`` when the caller already parsed
    ``connection_path``; the replay then runs on a clone instead of parsing the
    file a second time. The mapping is positional, so a replay that does not
    return the chains it was given cannot be aligned and yields an empty map.
    """
    if connection_source(connection_path) != CandidateSource.STRUCT_CONN:
        return {}
    parsed = (
        gemmi.read_structure(connection_path)
        if declared_structure is None
        else declared_structure
    )
    if len(parsed) == 0:
        return {}
    source_names = [str(chain.name) for chain in parsed[0]]
    copy = parsed.clone()
    copy.setup_entities()
    copy.shorten_chain_names()
    try:
        renamed = list(zip(source_names, copy[0], strict=True))
    except ValueError:
        return {}
    return {
        source: str(chain.name)
        for source, chain in renamed
        if source != str(chain.name)
    }


def analysis_atom_for_partner(
    structure: StructureContext, cra: gemmi.CRA, chain_names: Mapping[str, str]
) -> AtomSite | None:
    """Resolve one declared partner to an analysis atom by author identity.

    Returns ``None`` when the identity matches no residue, matches more than
    one, or names an atom the analyzed model does not hold.
    """
    # Gemmi's stub declares these members non-optional although the binding
    # returns ``None`` for an unmatched address, so a plain ``is None`` test
    # would be flagged as unreachable.
    chain = getattr(cra, "chain", None)
    residue = getattr(cra, "residue", None)
    atom = getattr(cra, "atom", None)
    if chain is None or residue is None or atom is None:
        return None

    source_chain_id = str(chain.name)
    resnum = f"{residue.seqid.num}{blank_if_missing(residue.seqid.icode)}"
    matches = structure.residues_for_source_author(
        str(residue.name), source_chain_id, resnum
    )
    if len(matches) != 1:
        # Fallback for an analysis PDB converted before chain renames were
        # recorded in its provenance: retry under Gemmi's shortened chain name.
        chain_id = chain_names.get(source_chain_id, source_chain_id)
        if chain_id != source_chain_id:
            matches = structure.residues_for_author(str(residue.name), chain_id, resnum)
    if len(matches) != 1:
        return None

    atom_name = str(atom.name).strip()
    altloc = blank_if_missing(atom.altloc)
    named = [site for site in matches[0].source_atoms if site.atom_name == atom_name]
    if altloc:
        named = [site for site in named if site.altloc == altloc]
    elif len(named) > 1:
        # A declaration naming no conformer names the residue, not one of its
        # alternates, so answer with the conformer per-residue selection chose.
        # Taking the deposition-order-first alternate instead would report a
        # conformer substitution the declaration never asked for.
        selected = matches[0].selected_altloc
        named = [site for site in named if site.altloc == selected] or named
    return named[0] if named else None


def selected_conformer_atom(
    structure: StructureContext, atom: AtomSite | None
) -> AtomSite | None:
    """Return the selected-conformer record for ``atom``'s chemical site.

    Contacts are measured on the selected conformer only. Returns ``None``
    when that conformer has no atom of this name and element, because the
    declared contact then has no counterpart in the analyzed model.
    """
    if atom is None:
        return None
    selected = structure.atom_for_indices(
        atom.chain_index, atom.residue_index, atom.atom_index
    )
    if selected is not None:
        return selected
    for candidate in structure.residue_for_atom(atom).contact_atoms:
        if candidate.atom_name == atom.atom_name and candidate.element == atom.element:
            return candidate
    return None


def declared_partner_is_metal(
    address: gemmi.AtomAddress, cra: gemmi.CRA | None
) -> bool:
    """Whether a source connection partner unambiguously names a metal.

    A resolved atom's element is authoritative. If resolution failed, only a
    metal residue name is sufficient evidence: atom names such as ``CA``, ``CD``,
    ``CE`` and ``SG`` are also ordinary macromolecular atom names.

    The name fallback excludes the symbols that are also non-metal component
    ids, so an unresolvable declaration naming RNA ``U`` or nitric oxide ``NO``
    is not reported as having named a metal.
    """
    source_atom = getattr(cra, "atom", None)
    if source_atom is not None:
        element = str(getattr(source_atom.element, "name", "")).upper()
        return element in METAL_ELEMENTS

    residue_id = getattr(address, "res_id", None)
    residue_name = str(getattr(residue_id, "name", "")).strip().upper()
    return residue_name in UNAMBIGUOUS_METAL_COMPONENT_IDS


def declared_candidate_geometry(
    structure: StructureContext,
    metal: AtomSite,
    neighbor: AtomSite,
    connection: gemmi.Connection,
) -> ContactImage:
    """Return the contact image for a resolved declared connection."""
    asu = connection.asu
    if asu == gemmi.Asu.Same or (
        asu == gemmi.Asu.Any and not structure.symmetry.search_available
    ):
        return ContactImage.explicit(metal, neighbor)

    if not structure.symmetry.search_available:
        raise ValueError(
            structure.symmetry.search_failure_reason or "symmetry metadata unavailable"
        )
    cell = structure.structure.cell
    nearest = cell.find_nearest_image(metal.pos, neighbor.pos, asu)
    transformed_fractional = cell.fract_image(nearest, cell.fractionalize(neighbor.pos))
    transformed = cell.orthogonalize(transformed_fractional)
    # ``same_asu()`` cannot classify this: after ``setup_cell_images`` the
    # image list holds the strict-NCS transforms too, so an NCS image is "not
    # the same ASU" and reads as crystallographic with no NCS operation id.
    # ``from_nearest_image`` asks ``image_provenance`` instead.
    return ContactImage.from_nearest_image(structure, nearest, transformed)


class _PartnerResolution(NamedTuple):
    """Both partners of one declaration resolved onto the analysis model."""

    #: The partners in declaration order, each ``None`` if it did not resolve;
    #: both ``None`` when the source-model lookup itself failed.
    atoms: tuple[AtomSite | None, AtomSite | None]
    declares_metal: bool
    conformer_deselected: bool
    conformer_substituted: bool
    #: Class name of the exception a source-model lookup raised; empty when
    #: both lookups completed.
    failure_exception_name: str = ""


def resolve_declared_partners(
    structure: StructureContext,
    source_model: PartnerLocator,
    connection: gemmi.Connection,
    chain_names: Mapping[str, str],
) -> _PartnerResolution:
    """Resolve both declared partners to selected-conformer atoms.

    Check declaration identifiers for metal evidence before find_cra can fail,
    then widen that evidence with the elements of the atoms that resolved: a
    resolved metal adds evidence, never removes it. Return lookup failures in
    failure_exception_name.

    Only the source-model lookup is guarded: a failure there is a property of
    the declaration, while an exception from the analysis-side resolution below
    is a bug that must not be relabelled as an unresolvable declaration.
    """
    addresses = (connection.partner1, connection.partner2)
    declares_metal = any(
        declared_partner_is_metal(address, None) for address in addresses
    )
    try:
        source_cras = [
            source_model.find_cra(address, ignore_segment=True) for address in addresses
        ]
    except Exception as exc:
        return _PartnerResolution(
            (None, None), declares_metal, False, False, type(exc).__name__
        )
    declares_metal = declares_metal or any(
        declared_partner_is_metal(address, cra)
        for address, cra in zip(addresses, source_cras, strict=True)
    )
    atoms: list[AtomSite | None] = []
    deselected = False
    substituted = False
    for cra in source_cras:
        declared_atom = analysis_atom_for_partner(structure, cra, chain_names)
        selected_atom = selected_conformer_atom(structure, declared_atom)
        if declared_atom is not None:
            if selected_atom is None:
                deselected = True
            elif selected_atom is not declared_atom:
                substituted = True
        atoms.append(selected_atom)
    first, second = atoms
    return _PartnerResolution((first, second), declares_metal, deselected, substituted)


def _is_selected_metal(
    atom: AtomSite | None, selected_metal_keys: Set[AtomKey]
) -> bool:
    """Whether a resolved partner is one of the metal sites under analysis."""
    return atom is not None and atom.source_key in selected_metal_keys


def _metal_and_neighbor(
    first: AtomSite, second: AtomSite, selected_metal_keys: Set[AtomKey]
) -> tuple[AtomSite, AtomSite] | None:
    """Order two resolved partners as ``(metal, neighbor)``.

    Returns ``None`` when neither or both partners are selected metals: the
    former is a declaration unrelated to the analyzed sites and the latter a
    metal-metal contact, and neither is a coordination candidate.
    """
    first_is_metal = _is_selected_metal(first, selected_metal_keys)
    if first_is_metal == _is_selected_metal(second, selected_metal_keys):
        return None
    return (first, second) if first_is_metal else (second, first)


class DeclaredCandidate(NamedTuple):
    """One resolved declaration: the metal it names and the donor candidate.

    The metal travels beside the candidate rather than on it: only this
    resolution path knows which metal a declaration was resolved against, and
    ``coordination.analysis`` only needs it to group the candidates by site.
    """

    metal: AtomSite
    candidate: Candidate


def _declared_connection_record(
    connection: gemmi.Connection, connection_id: str, source: CandidateSource
) -> DeclaredConnectionRecord:
    """Serialize the provenance a candidate keeps from its source declaration."""
    reported_distance = float(connection.reported_distance)
    if not math.isfinite(reported_distance) or reported_distance <= 0:
        reported_distance = NAN
    return DeclaredConnectionRecord(
        source=source,
        connection_id=connection_id,
        connection_type=_enum_name(connection.type),
        connection_link_id=str(connection.link_id).strip(),
        connection_asu=_enum_name(connection.asu),
        connection_reported_distance=reported_distance,
    )


def declared_candidate_for_connection(
    structure: StructureContext,
    connection: gemmi.Connection,
    connection_id: str,
    source: CandidateSource,
    resolved: _PartnerResolution,
    selected_metal_keys: Set[AtomKey],
) -> tuple[DeclaredCandidate | None, list[str], list[str]]:
    """Return a resolved candidate, issues, and warnings for one declaration.

    Return candidate=None for unmeasurable contacts. Report unresolved metal
    declarations as issues and ignore declarations unrelated to metals.

    Warning codes are entry-level, so a code is only returned when this
    declaration leaves a trace the reader can attach it to -- a candidate, or
    an issue. A declaration that is discarded silently, such as the
    intra-cofactor links a deposition writes between a cluster's own atoms,
    returns none. Per code:

    * ``DECLARED_CONNECTION_ZERO_OCCUPANCY_PARTNER`` and
      ``DECLARED_DONOR_OUTSIDE_SUPPORTED_CLASSES`` describe the candidate that
      was produced, so they are returned only alongside one.
    * ``DECLARED_DONOR_ELEMENT_UNSUPPORTED`` describes the declaration itself:
      the deposition declared a metal contact to an element that is never a
      donor. No candidate can carry it, so it is returned on its early exit.
    * ``DECLARED_CONNECTION_CONFORMER_SUBSTITUTED`` records how the
      declaration was resolved, so it accompanies a candidate and also the
      issues raised for a declaration that resolution could not complete.
    """
    issues: list[str] = []
    if resolved.failure_exception_name:
        if resolved.declares_metal:
            issues.append(
                f"{source} {connection_id} resolution failed: "
                f"{resolved.failure_exception_name}"
            )
        return None, issues, []

    first, second = resolved.atoms
    connection_involves_metal = resolved.declares_metal or any(
        _is_selected_metal(atom, selected_metal_keys) for atom in resolved.atoms
    )
    substituted: list[str] = (
        [WarningCode.DECLARED_CONNECTION_CONFORMER_SUBSTITUTED]
        if resolved.conformer_substituted and connection_involves_metal
        else []
    )
    if resolved.conformer_deselected:
        if not resolved.declares_metal:
            return None, [], []
        issues.append(
            f"{source} {connection_id} partner names a conformer whose "
            f"selected alternative has no matching atom"
        )
        return None, issues, substituted
    if first is None or second is None:
        if not connection_involves_metal:
            return None, [], []
        issues.append(
            f"{source} {connection_id} neither partner resolved"
            if first is None and second is None
            else f"{source} {connection_id} partner unresolved"
        )
        return None, issues, substituted
    pair = _metal_and_neighbor(first, second, selected_metal_keys)
    if pair is None:
        return None, [], []

    metal, neighbor = pair
    if not (metal.coordinates_valid and neighbor.coordinates_valid):
        issues.append(
            f"{source} {connection_id} geometry unavailable: "
            "partner has non-finite coordinates"
        )
        return None, issues, substituted
    if neighbor.element not in DONOR_ELEMENTS:
        return None, [], [WarningCode.DECLARED_DONOR_ELEMENT_UNSUPPORTED]
    residue = structure.residue_for_atom(neighbor)
    warnings = list(substituted)
    if neighbor.occupancy_valid and neighbor.occupancy == 0.0:
        warnings.append(WarningCode.DECLARED_CONNECTION_ZERO_OCCUPANCY_PARTNER)
    # Keep unsupported donor classes as candidates with measured distances.
    # They cannot be scored or promoted to bond rows without a reference.
    donor_class_supported = bool(residue.is_water or residue.residue_name in AA)
    if not donor_class_supported:
        warnings.append(WarningCode.DECLARED_DONOR_OUTSIDE_SUPPORTED_CLASSES)
    try:
        image = declared_candidate_geometry(structure, metal, neighbor, connection)
    except Exception as exc:
        issues.append(
            f"{source} {connection_id} geometry unresolved: {type(exc).__name__}: {exc}"
        )
        return None, issues, substituted
    if neighbor.residue_key == metal.residue_key and not image.symmetry_contact:
        return None, [], []

    candidate = Candidate(
        neighbor=neighbor,
        image=image,
        candidate_sources={source},
        declared_connections=[
            _declared_connection_record(connection, connection_id, source)
        ],
        donor_class_supported=donor_class_supported,
    )
    return DeclaredCandidate(metal, candidate), issues, warnings


def collect_declared_candidates(
    structure: StructureContext,
    connection_path: str | None,
    metals: Sequence[AtomSite],
) -> tuple[list[DeclaredCandidate], list[str], list[str]]:
    """Resolve source ``struct_conn``/``LINK`` claims to analysis atoms.

    Partners are matched by author identity -- chain, sequence number,
    insertion code, component, atom name, altloc -- and re-pointed onto their
    residue's selected conformer. Each resolved candidate is returned paired
    with the metal site it was resolved against, in declaration order.
    """
    if not connection_path:
        return [], [], []
    source = connection_source(connection_path)
    try:
        declared_structure = gemmi.read_structure(connection_path)
        if len(declared_structure) == 0:
            return [], [f"{source} contains no coordinate model"], []
        # Inside the guard: replaying the conversion is Gemmi work too, and a
        # failure there must degrade this entry to a declaration issue rather
        # than escape and fail the whole bond stage.
        chain_names = analysis_chain_names(connection_path, declared_structure)
    except Exception as exc:
        return [], [f"{source} parse failed: {type(exc).__name__}: {exc}"], []

    source_model = declared_structure[0]
    selected_metal_keys = {metal.source_key for metal in metals}
    candidates: list[DeclaredCandidate] = []
    issues: list[str] = []
    warnings: list[str] = []
    connection_ids: set[str] = set()
    for index, connection in enumerate(declared_structure.connections, start=1):
        connection_id = str(connection.name).strip() or f"{source}_{index}"
        # Candidate merging keys declarations on (source, connection_id), so a
        # deposition that repeats an id would collapse distinct records.
        while connection_id in connection_ids:
            connection_id = f"{connection_id}_{index}"
        connection_ids.add(connection_id)
        resolved = resolve_declared_partners(
            structure, source_model, connection, chain_names
        )
        candidate, connection_issues, connection_warnings = (
            declared_candidate_for_connection(
                structure,
                connection,
                connection_id,
                source,
                resolved,
                selected_metal_keys,
            )
        )
        issues.extend(connection_issues)
        warnings.extend(connection_warnings)
        if candidate is not None:
            candidates.append(candidate)
    return candidates, issues, warnings
