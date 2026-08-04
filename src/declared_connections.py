"""Source ``struct_conn`` / ``LINK`` declarations, resolved onto the model.

A deposition states its own coordination: ``_struct_conn`` in an mmCIF, ``LINK``
records in a legacy PDB. Alchemy trusts those claims, but it measures geometry
on the *analysis* model, which for an mmCIF entry is a converted PDB. Binding
one to the other is the whole job of this module, and it is provenance work
rather than chemistry -- it shares no vocabulary with z-scores, reference
distances or coordination counts, which is why it lives apart from
``bond_analysis``.

Three properties of that join are load-bearing, and each is defended by a test:

* **Partners are matched by author identity**, never by atom serial. Gemmi's
  PDB writer emits a TER record after each polymer and every TER consumes a
  serial, so a converted model's serials run ahead of the source
  ``_atom_site.id``; conversion provenance restores the identities the legacy
  format could not hold.
* **Both partners are re-pointed onto their residue's selected conformer**, so
  a declaration cannot introduce a second record for a chemical site the
  proximity search already reports.
* **Whether a declaration names a metal is decided twice** -- once from the
  declaration's own identifiers and again from the resolved atom's element.
  See ``_resolve_declared_partners``.

``_collect_declared_candidates`` is the entry point. It never raises: every
failure becomes a message in its ``issues`` list, which the caller turns into
``declared_connection_resolution_incomplete``, or a code in its ``warnings``
list. A declaration Alchemy cannot bind must leave an audit trail, because
dropping it silently is indistinguishable from a metal that has no
coordination at all.
"""

import math
from typing import NamedTuple, Optional

from codes import CandidateSource, ContactScope, WarningCode
from contact_record import Candidate
from donor_chemistry import AA, DONOR_ELEMENTS
from metal_elements import METAL_ELEMENTS
from structure_analysis import NAN, blank_if_missing, position_distance


def _connection_source(path):
    lower = str(path or "").lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    return (
        CandidateSource.STRUCT_CONN
        if lower.endswith((".cif", ".mmcif"))
        else CandidateSource.LINK
    )


def _enum_name(value):
    return str(getattr(value, "name", value)).lower()


def _analysis_chain_names(connection_path):
    """Map source chain names onto the analysis model's chain names.

    For ordinary structures, replaying Gemmi's ``setup_entities`` and
    ``shorten_chain_names`` calls recovers the conversion mapping instead of
    assuming it is the identity. Oversized structures use residue-level
    provenance and resolve source identities before this fallback is reached.
    A source PDB is extracted textually and keeps its chain names, so its
    mapping is empty.
    """
    import gemmi

    if _connection_source(connection_path) != CandidateSource.STRUCT_CONN:
        return {}
    copy = gemmi.read_structure(connection_path)
    if len(copy) == 0:
        return {}
    source_names = [str(chain.name) for chain in copy[0]]
    copy.setup_entities()
    copy.shorten_chain_names()
    return {
        source: str(chain.name)
        for source, chain in zip(source_names, copy[0])
        if source != str(chain.name)
    }


def _analysis_atom_for_partner(structure, cra, chain_names):
    """Resolve one declared partner to an analysis atom by author identity.

    The declaration's author identifiers survive conversion: only chain names
    are shortened, which ``chain_names`` reverses, and a component identifier
    too long for the legacy residue field is indexed under both its source and
    its converted name. Returns ``None`` when the identity matches no residue,
    matches more than one, or names an atom the analyzed model does not hold.
    """
    chain = getattr(cra, "chain", None)
    residue = getattr(cra, "residue", None)
    atom = getattr(cra, "atom", None)
    if chain is None or residue is None or atom is None:
        return None

    source_chain_id = str(chain.name)
    resnum = f"{residue.seqid.num}{blank_if_missing(residue.seqid.icode)}"
    source_lookup = (
        structure.residues_for_source_author
        if hasattr(type(structure), "residues_for_source_author")
        else structure.residues_for_author
    )
    matches = source_lookup(str(residue.name), source_chain_id, resnum)
    if len(matches) != 1:
        # Older analysis PDBs retain only Gemmi's chain-name shortening.  New
        # packed PDBs embed full residue provenance, so the source lookup above
        # succeeds without guessing how sequence numbers were remapped.
        chain_id = chain_names.get(source_chain_id, source_chain_id)
        if chain_id != source_chain_id:
            # The combined legacy index restores >3-character residue names
            # while using Gemmi's shortened chain name.  Packed structures do
            # not need this fallback because their full source identity is
            # indexed above.
            matches = structure.residues_for_author(str(residue.name), chain_id, resnum)
    if len(matches) != 1:
        return None

    atom_name = str(atom.name).strip()
    altloc = blank_if_missing(atom.altloc)
    named = [site for site in matches[0].source_atoms if site.atom_name == atom_name]
    if altloc:
        named = [site for site in named if site.altloc == altloc]
    return named[0] if named else None


def _selected_conformer_atom(structure, atom):
    """Return the selected-conformer record for ``atom``'s chemical site.

    A declaration names one deposited atom record, which may belong to an
    alternate conformer that per-residue selection did not choose. Contacts are
    measured on the selected conformer only, so re-point the declaration onto
    the same-named atom of that conformer rather than admitting a second record
    for a chemical site the proximity search already reports. Returns ``None``
    when the selected conformer has no atom of that name and element, because
    the declared contact then has no counterpart in the analyzed model.
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


def _declared_partner_is_metal(address, cra):
    """Whether a source connection partner unambiguously names a metal.

    Prefer the element of the atom resolved in the source model.  A malformed
    declaration may omit that atom or name a record no longer present, so fall
    back only to unambiguous component/atom identifiers such as ``ZN``.  Do not
    guess elements from prefixes (for example ``CA`` is commonly a protein
    alpha carbon): an indeterminate non-metal declaration is outside Alchemy's
    coordination-analysis scope.
    """
    source_atom = getattr(cra, "atom", None)
    if source_atom is not None:
        element = str(getattr(source_atom.element, "name", "")).upper()
        if element in METAL_ELEMENTS:
            return True

    residue_id = getattr(address, "res_id", None)
    residue_name = str(getattr(residue_id, "name", "")).strip().upper()
    atom_name = str(getattr(address, "atom_name", "")).strip().upper()
    return residue_name in METAL_ELEMENTS or atom_name in METAL_ELEMENTS


def _declared_candidate_geometry(structure, metal, neighbor, connection):
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
    shift_a, shift_b, shift_c = nearest.pbc_shift
    translation = (int(shift_a), int(shift_b), int(shift_c))
    image_index = int(nearest.sym_idx)
    # Classified exactly as the proximity path does. ``same_asu()`` alone
    # cannot tell the two apart: after ``setup_cell_images`` the image list
    # holds the strict-NCS transforms as well, so an NCS image is "not the same
    # ASU" and would otherwise be reported as crystallographic with no NCS
    # operation identifier.
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
        "contact_scope": contact_scope,
        "symmetry_image_index": image_index,
        "symmetry_operation": nearest.symmetry_code(),
        "translation": translation,
    }


class _PartnerResolution(NamedTuple):
    """What binding one declaration's two partners to the model produced."""

    #: One selected-conformer atom per partner, ``None`` where the identity did
    #: not resolve -- or the whole list is ``None`` when resolution raised.
    atoms: Optional[list]
    #: Whether either partner names a metal. Meaningful even when ``atoms`` is
    #: ``None``: it is what decides whether a failure is worth reporting.
    declares_metal: bool
    #: A partner named a conformer whose selected alternative has no such atom.
    deselected: bool
    #: A partner was re-pointed onto a different conformer's atom record.
    substituted: bool
    #: Exception type name, when resolving the partners raised.
    failure: Optional[str]


def _resolve_declared_partners(structure, source_model, connection, chain_names):
    """Bind both partners of one declaration to selected-conformer atoms.

    **The metal test runs twice, and both calls are deliberate.** The first
    reads the declaration's own identifiers and runs before ``find_cra``, which
    can raise: without it, a declaration Alchemy failed to resolve would lose
    the one piece of evidence that says the failure matters, and a metal site
    would be dropped with no audit trail. The second is widened by ``or`` from
    the atom ``find_cra`` returned, whose element is the better evidence when
    it is available. Neither subsumes the other.

    Never raises. A failure is returned as ``failure`` so the caller can decide
    whether this declaration was one worth reporting.
    """
    addresses = (connection.partner1, connection.partner2)
    declares_metal = any(
        _declared_partner_is_metal(address, None) for address in addresses
    )
    try:
        source_cras = [
            source_model.find_cra(address, ignore_segment=True) for address in addresses
        ]
        declares_metal = declares_metal or any(
            _declared_partner_is_metal(address, cra)
            for address, cra in zip(addresses, source_cras)
        )
        atoms = []
        deselected = False
        substituted = False
        for cra in source_cras:
            declared_atom = _analysis_atom_for_partner(structure, cra, chain_names)
            selected_atom = _selected_conformer_atom(structure, declared_atom)
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


def _declared_candidate_for_connection(
    structure, connection, connection_id, source, resolved, selected_metal_keys
):
    """Return ``(candidate, issues, warnings)`` for one resolved declaration.

    ``candidate`` is ``None`` whenever the declaration does not describe a
    metal-donor contact this model can measure. Each way that can happen is one
    guard with its own early return, and each decides separately whether the
    outcome is an issue (reportable: the declaration named a metal, so silence
    would understate a coordination number), a warning code, or nothing at all
    (a link between two amino acids is simply not Alchemy's subject).
    """
    issues = []
    warnings = []
    if resolved.failure is not None:
        if resolved.declares_metal:
            issues.append(
                f"{source} {connection_id} resolution failed: {resolved.failure}"
            )
        return None, issues, warnings

    if resolved.deselected:
        if resolved.declares_metal:
            issues.append(
                f"{source} {connection_id} partner names a conformer whose "
                f"selected alternative has no matching atom"
            )
        return None, issues, warnings

    first, second = resolved.atoms
    first_is_metal = first is not None and first.source_key in selected_metal_keys
    second_is_metal = second is not None and second.source_key in selected_metal_keys
    connection_involves_metal = (
        resolved.declares_metal or first_is_metal or second_is_metal
    )
    if resolved.substituted and connection_involves_metal:
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
    residue = structure.residue_for_atom(neighbor)
    if neighbor.element not in DONOR_ELEMENTS:
        # Not a donor-like atom at all, so there is no coordination
        # evidence to retain. Recorded rather than dropped in silence.
        warnings.append(WarningCode.DECLARED_DONOR_ELEMENT_UNSUPPORTED)
        return None, issues, warnings
    # Nucleic acids, modified residues and organic ligands are genuine
    # metal donors, but no bundled literature reference covers them, so
    # their geometry can never be z-scored. Retain them as candidate
    # evidence carrying the measured distance and the declaration's own
    # provenance: dropping them is indistinguishable from a metal with no
    # coordination at all, which is how a fully coordinated nucleic-acid
    # site came to report no contacts. They are deliberately not promoted
    # to bond rows -- that would raise coordination counts and enlarge the
    # confidence geometry-coverage denominator on the strength of a contact
    # nothing can assess.
    donor_class_supported = bool(residue.is_water or residue.residue_name in AA)
    if not donor_class_supported:
        warnings.append(WarningCode.DECLARED_DONOR_OUTSIDE_SUPPORTED_CLASSES)
    try:
        geometry = _declared_candidate_geometry(structure, metal, neighbor, connection)
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

    record = {
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


def _collect_declared_candidates(structure, connection_path, metals):
    """Resolve source ``struct_conn``/``LINK`` claims to analysis atoms.

    Partners are resolved by author identity -- chain, sequence number,
    insertion code, component, atom name, and altloc -- against the analysis
    model. Atom serials cannot carry this join: Gemmi's PDB writer emits a TER
    record after each polymer and every TER consumes a serial number, so the
    serials of an mmCIF-converted model run ahead of the source
    ``_atom_site.id`` by one for each preceding TER, and a partner past the
    first TER would resolve to a neighbouring atom. Conversion provenance
    restores source residue identities when the legacy PDB representation
    shortens a chain, truncates a component name, or packs an oversized model
    into synthetic chain and residue identifiers.

    A declaration names one deposited record, so either partner may be an
    alternate conformer that per-residue selection did not choose. Both are
    re-pointed onto their residue's selected conformer before use, so a
    declaration cannot introduce a second record for a chemical site the
    proximity search already reports and cannot inflate a coordination number
    with a conformer that is absent from the analyzed model.
    """
    import gemmi

    if not connection_path:
        return [], [], []
    source = _connection_source(connection_path)
    try:
        declared_structure = gemmi.read_structure(connection_path)
    except Exception as exc:
        return [], [f"{source} parse failed: {type(exc).__name__}: {exc}"], []
    if len(declared_structure) == 0:
        return [], [f"{source} contains no coordinate model"], []

    source_model = declared_structure[0]
    chain_names = _analysis_chain_names(connection_path)
    selected_metal_keys = {metal.source_key for metal in metals}
    candidates = []
    issues = []
    warnings = []
    for index, connection in enumerate(declared_structure.connections, start=1):
        connection_id = str(connection.name).strip() or f"{source}_{index}"
        resolved = _resolve_declared_partners(
            structure, source_model, connection, chain_names
        )
        candidate, connection_issues, connection_warnings = (
            _declared_candidate_for_connection(
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
