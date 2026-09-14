"""Resolve source struct_conn and LINK contacts onto the analysis model.

Match author identities rather than serials, which PDB conversion can change.
Use each residue's selected conformer and retain unresolved declarations as
issues or warnings. See resolve_declared_partners for metal identification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set
from typing import (
    TYPE_CHECKING,
    NamedTuple,
    Protocol,
    TypedDict,
    cast,
)

from codes import CandidateSource, ContactScope, WarningCode
from coordination.contact_record import Candidate, DeclaredConnectionRecord
from coordination.donor_chemistry import AA, DONOR_ELEMENTS
from metal_elements import METAL_ELEMENTS, UNAMBIGUOUS_METAL_COMPONENT_IDS
from structure_analysis import (
    NAN,
    AtomSite,
    StructureContext,
    blank_if_missing,
    pbc_translation,
    position_distance,
)

if TYPE_CHECKING:
    import gemmi


class PartnerLocator(Protocol):
    """Define the source-model address lookup used during declaration resolution.

    The address is positional-only to match Gemmi's binding.
    """

    def find_cra(
        self, address: gemmi.AtomAddress, /, ignore_segment: bool = ...
    ) -> gemmi.CRA:
        """Find the chain, residue, and atom at a source-model address."""
        ...


#: Identity of one selected metal atom, as ``AtomSite.source_key`` reports it.
_MetalKey = tuple[int, int, int, int]


class _CandidateGeometry(TypedDict):
    """Geometry fields passed directly into ``Candidate`` construction."""

    distance_raw: float
    transformed_position: tuple[float, float, float]
    symmetry_contact: bool
    crystallographic_contact: bool
    strict_ncs_contact: bool
    strict_ncs_operation_id: str
    contact_scope: ContactScope
    symmetry_image_index: int
    symmetry_operation: str
    translation: tuple[int, int, int]


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


def analysis_chain_names(connection_path: str) -> dict[str, str]:
    """Map source chain names onto the analysis model's chain names.

    Replaying Gemmi's ``setup_entities`` and ``shorten_chain_names`` recovers
    the conversion mapping. A source PDB is extracted textually and keeps its
    chain names, so its mapping is empty.
    """
    import gemmi

    if connection_source(connection_path) != CandidateSource.STRUCT_CONN:
        return {}
    copy = gemmi.read_structure(connection_path)
    if len(copy) == 0:
        return {}
    source_names = [str(chain.name) for chain in copy[0]]
    copy.setup_entities()
    copy.shorten_chain_names()
    return {
        source: str(chain.name)
        for source, chain in zip(source_names, copy[0], strict=False)
        if source != str(chain.name)
    }


def analysis_atom_for_partner(
    structure: StructureContext, cra: gemmi.CRA, chain_names: Mapping[str, str]
) -> AtomSite | None:
    """Resolve one declared partner to an analysis atom by author identity.

    Returns ``None`` when the identity matches no residue, matches more than
    one, or names an atom the analyzed model does not hold.
    """
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
        # Fallback for an analysis PDB carrying no residue provenance: retry
        # under Gemmi's shortened chain name.
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
) -> _CandidateGeometry:
    """Return contact geometry for a resolved declared connection."""
    import gemmi

    asu = connection.asu
    if asu == gemmi.Asu.Same or (
        asu == gemmi.Asu.Any and not structure.symmetry_search_available
    ):
        return {
            "distance_raw": position_distance(metal.xyz, neighbor.xyz),
            "transformed_position": neighbor.xyz,
            "symmetry_contact": False,
            "crystallographic_contact": False,
            "strict_ncs_contact": False,
            "strict_ncs_operation_id": "",
            "contact_scope": ContactScope.EXPLICIT,
            "symmetry_image_index": 0,
            "symmetry_operation": "1_555",
            "translation": (0, 0, 0),
        }

    if not structure.symmetry_search_available:
        raise ValueError(
            structure.symmetry_search_failure_reason or "symmetry metadata unavailable"
        )
    cell = structure.structure.cell
    nearest = cell.find_nearest_image(metal.pos, neighbor.pos, asu)
    transformed_fractional = cell.fract_image(nearest, cell.fractionalize(neighbor.pos))
    transformed = cell.orthogonalize(transformed_fractional)
    translation = pbc_translation(nearest)
    image_index = int(nearest.sym_idx)
    # ``same_asu()`` cannot classify this: after ``setup_cell_images`` the
    # image list holds the strict-NCS transforms too, so an NCS image is "not
    # the same ASU" and reads as crystallographic with no NCS operation id.
    (
        crystallographic_contact,
        strict_ncs_contact,
        strict_ncs_operation_id,
        contact_scope,
    ) = structure.image_provenance(image_index, translation)
    return {
        "distance_raw": float(nearest.dist()),
        "transformed_position": (
            float(transformed.x),
            float(transformed.y),
            float(transformed.z),
        ),
        "symmetry_contact": crystallographic_contact or strict_ncs_contact,
        "crystallographic_contact": crystallographic_contact,
        "strict_ncs_contact": strict_ncs_contact,
        "strict_ncs_operation_id": strict_ncs_operation_id,
        # ``image_provenance`` is intentionally broad at the structure layer;
        # every value it returns here is a ``ContactScope`` member.
        "contact_scope": cast(ContactScope, contact_scope),
        "symmetry_image_index": image_index,
        "symmetry_operation": nearest.symmetry_code(),
        "translation": translation,
    }


class _PartnerResolution(NamedTuple):
    """Results and failure details from resolving a declaration's two partners."""

    #: ``None`` exactly when ``failure_exception_name`` is set; otherwise the
    #: two partners in declaration order, each ``None`` if it did not resolve.
    selected_conformer_atoms: list[AtomSite | None] | None
    declares_metal: bool
    conformer_deselected: bool
    conformer_substituted: bool
    failure_exception_name: str | None


def resolve_declared_partners(
    structure: StructureContext,
    source_model: PartnerLocator,
    connection: gemmi.Connection,
    chain_names: Mapping[str, str],
) -> _PartnerResolution:
    """Resolve both declared partners to selected-conformer atoms.

    Check declaration identifiers for metal evidence before find_cra can fail,
    then refine that evidence using resolved atom elements. Return lookup failures
    in failure_exception_name.
    """
    addresses = (connection.partner1, connection.partner2)
    declares_metal = any(
        declared_partner_is_metal(address, None) for address in addresses
    )
    try:
        source_cras = [
            source_model.find_cra(address, ignore_segment=True) for address in addresses
        ]
        declares_metal = declares_metal or any(
            declared_partner_is_metal(address, cra)
            for address, cra in zip(addresses, source_cras, strict=False)
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
    except Exception as exc:
        return _PartnerResolution(
            None, declares_metal, False, False, type(exc).__name__
        )
    return _PartnerResolution(atoms, declares_metal, deselected, substituted, None)


def declared_candidate_for_connection(
    structure: StructureContext,
    connection: gemmi.Connection,
    connection_id: str,
    source: CandidateSource,
    resolved: _PartnerResolution,
    selected_metal_keys: Set[_MetalKey],
) -> tuple[Candidate | None, list[str], list[str]]:
    """Return a candidate, issues, and warnings for one resolved declaration.

    Return candidate=None for unmeasurable contacts. Report unresolved metal
    declarations as issues and ignore declarations unrelated to metals.
    """
    issues: list[str] = []
    warnings: list[str] = []
    if resolved.failure_exception_name is not None:
        if resolved.declares_metal:
            issues.append(
                f"{source} {connection_id} resolution failed: "
                f"{resolved.failure_exception_name}"
            )
        return None, issues, warnings

    if resolved.conformer_deselected:
        if resolved.declares_metal:
            issues.append(
                f"{source} {connection_id} partner names a conformer whose "
                f"selected alternative has no matching atom"
            )
        return None, issues, warnings

    # Bound whenever no failure name was recorded, which the guard above has
    # already returned on.
    first, second = cast("list[AtomSite | None]", resolved.selected_conformer_atoms)
    first_is_metal = first is not None and first.source_key in selected_metal_keys
    second_is_metal = second is not None and second.source_key in selected_metal_keys
    connection_involves_metal = (
        resolved.declares_metal or first_is_metal or second_is_metal
    )
    if resolved.conformer_substituted and connection_involves_metal:
        warnings.append(WarningCode.DECLARED_CONNECTION_CONFORMER_SUBSTITUTED)
    if first is None and second is None:
        if connection_involves_metal:
            issues.append(f"{source} {connection_id} neither partner resolved")
        return None, issues, warnings
    if not (first_is_metal or second_is_metal):
        if connection_involves_metal and (first is None or second is None):
            issues.append(f"{source} {connection_id} partner unresolved")
        return None, issues, warnings
    if first is None or second is None:
        issues.append(f"{source} {connection_id} partner unresolved")
        return None, issues, warnings
    if first_is_metal and second_is_metal:
        return None, issues, warnings

    metal, neighbor = (first, second) if first_is_metal else (second, first)
    if not (metal.coordinates_valid and neighbor.coordinates_valid):
        issues.append(
            f"{source} {connection_id} geometry unavailable: "
            "partner has non-finite coordinates"
        )
        return None, issues, warnings
    if neighbor.occupancy_valid and neighbor.occupancy == 0.0:
        warnings.append(WarningCode.DECLARED_CONNECTION_ZERO_OCCUPANCY_PARTNER)
    residue = structure.residue_for_atom(neighbor)
    if neighbor.element not in DONOR_ELEMENTS:
        warnings.append(WarningCode.DECLARED_DONOR_ELEMENT_UNSUPPORTED)
        return None, issues, warnings
    # Keep unsupported donor classes as candidates with measured distances.
    # They cannot be scored or promoted to bond rows without a reference.
    donor_class_supported = bool(residue.is_water or residue.residue_name in AA)
    if not donor_class_supported:
        warnings.append(WarningCode.DECLARED_DONOR_OUTSIDE_SUPPORTED_CLASSES)
    try:
        geometry = declared_candidate_geometry(structure, metal, neighbor, connection)
    except Exception as exc:
        issues.append(
            f"{source} {connection_id} geometry unresolved: {type(exc).__name__}: {exc}"
        )
        return None, issues, warnings
    if neighbor.residue_key == metal.residue_key and not geometry["symmetry_contact"]:
        return None, issues, warnings

    reported_distance = float(connection.reported_distance)
    if not math.isfinite(reported_distance) or reported_distance <= 0:
        reported_distance = NAN

    record: DeclaredConnectionRecord = {
        "source": source,
        "connection_id": connection_id,
        "connection_type": _enum_name(connection.type),
        "connection_link_id": str(connection.link_id).strip(),
        "connection_asu": _enum_name(connection.asu),
        "connection_reported_distance": reported_distance,
    }
    candidate = Candidate(
        metal=metal,
        neighbor=neighbor,
        **geometry,
        candidate_sources={source},
        declared_connections=[record],
        donor_class_supported=donor_class_supported,
    )
    return candidate, issues, warnings


def collect_declared_candidates(
    structure: StructureContext,
    connection_path: str | None,
    metals: Sequence[AtomSite],
) -> tuple[list[Candidate], list[str], list[str]]:
    """Resolve source ``struct_conn``/``LINK`` claims to analysis atoms.

    Partners are matched by author identity -- chain, sequence number,
    insertion code, component, atom name, altloc -- and re-pointed onto their
    residue's selected conformer.
    """
    import gemmi

    if not connection_path:
        return [], [], []
    source = connection_source(connection_path)
    try:
        declared_structure = gemmi.read_structure(connection_path)
    except Exception as exc:
        return [], [f"{source} parse failed: {type(exc).__name__}: {exc}"], []
    if len(declared_structure) == 0:
        return [], [f"{source} contains no coordinate model"], []

    source_model = declared_structure[0]
    chain_names = analysis_chain_names(connection_path)
    selected_metal_keys = {metal.source_key for metal in metals}
    candidates: list[Candidate] = []
    issues: list[str] = []
    warnings: list[str] = []
    for index, connection in enumerate(declared_structure.connections, start=1):
        connection_id = str(connection.name).strip() or f"{source}_{index}"
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
