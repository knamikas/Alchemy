"""Shared Gemmi structure preprocessing for Alchemy.

The pipeline deliberately uses two atom sets from the first coordinate model:

* ``source_atoms`` retains every deduplicated alternate position for the
  occupancy-weighted DPI atom count.
* ``contact_atoms`` contains one coherent, occupancy-selected conformer per
  residue for metal-contact searches.

Keeping these sets explicit avoids depending on a parser's implicit alternate-
location selection and makes the model, occupancy, and deduplication policies
identical for single-entry and PDB-wide runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gemmi

from codes import ContactScope, ElementStatus, OccupancyStatus, WarningCode


NAN = float("nan")
DUPLICATE_ATOM_POSITION_TOLERANCE = 0.001
RESNAME_REMARK_PREFIX = "REMARK 950 ALCHEMY RESNAME"
RESIDUE_REMARK_PREFIX = "REMARK 950 ALCHEMY RESIDUE"


# Blank PDB columns, Gemmi's NUL altloc, and the two mmCIF null tokens all mean
# "no value here". Altloc labels, insertion codes, and main.py's mmCIF
# conversion bookkeeping share this one definition.
MISSING_VALUE_TOKENS = ("", " ", "\x00", ".", "?")
PDB_HYBRID36_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def blank_if_missing(value: str) -> str:
    """Return ``value``, or blank when it is a missing-value marker."""
    return "" if value in MISSING_VALUE_TOKENS else str(value)


def _decode_pdb_resseq(value: str) -> int:
    """Decode a four-column decimal or hybrid-36 PDB resSeq value.

    Gemmi exposes the decoded integer through ``Residue.seqid.num``. Raw PDB
    and EDSTATS identifiers must use the same representation for stable joins.
    Gemmi treats letter case equivalently for PDB resSeq fields, so normalize to
    upper case before decoding the base-36 range.
    """
    text = str(value)
    if len(text) > 4:
        raise ValueError(f"PDB residue sequence field is wider than 4: {value!r}")
    text = text.rjust(4)
    first = text[0]
    if first in (" ", "-") or first.isdigit():
        try:
            return int(text)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"invalid decimal PDB residue sequence: {value!r}"
            ) from exc

    encoded = text.upper()
    if any(character not in PDB_HYBRID36_DIGITS for character in encoded):
        raise ValueError(f"invalid hybrid-36 PDB residue sequence: {value!r}")
    decoded = 0
    for character in encoded:
        decoded = decoded * 36 + PDB_HYBRID36_DIGITS.index(character)
    return decoded - 10 * 36**3 + 10**4


def canonical_pdb_residue_id(value: str) -> str:
    """Return a decimal residue identifier with any insertion code appended.

    EDSTATS separates an insertion code with ``:``. Accept the equivalent
    compact form as well, then decode the four-column PDB residue number using
    the same integer representation Gemmi exposes.
    """
    text = str(value).strip()
    insertion = ""
    if ":" in text:
        number_text, insertion = text.split(":", 1)
    elif len(text) > 4:
        number_text, insertion = text[:4], text[4:]
    elif text and text[-1].isalpha() and not text[0].isalpha() and text[:-1]:
        number_text, insertion = text[:-1], text[-1]
    else:
        number_text = text
    if len(insertion) > 1:
        raise ValueError(f"invalid PDB insertion code in residue id: {value!r}")
    return f"{_decode_pdb_resseq(number_text)}{insertion}"


def position_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two ``(x, y, z)`` triples."""
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _residue_number(residue: gemmi.Residue) -> int:
    number = residue.seqid.num
    if number is None:
        raise ValueError(f"residue {residue.name!r} has no author sequence number")
    return number


def _valid_occupancy(value: object) -> bool:
    if not isinstance(value, (int, float, str)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def _parse_pdb_element(value: str) -> Tuple[str, str]:
    """Return the canonical deposited PDB element and its validation status."""
    deposited = value.strip()
    if not deposited:
        return "", ElementStatus.MISSING
    try:
        element = gemmi.Element(deposited)
    except (RuntimeError, ValueError):
        return "", ElementStatus.INVALID
    canonical = str(element.name).upper()
    if int(element.atomic_number) <= 0 or canonical in ("", "X"):
        return "", ElementStatus.INVALID
    return canonical, ElementStatus.VALID


def _format_number(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.6g}"


@dataclass(frozen=True)
class RawOccupancy:
    value: Optional[float]
    status: str
    element: str = ""
    element_status: str = ElementStatus.MISSING
    atom_name: str = ""
    altloc: str = ""
    chain_id: str = ""
    residue_name: str = ""
    residue_number: str = ""
    insertion_code: str = ""
    serial: Optional[int] = None
    source_order: int = 0

    @property
    def valid(self) -> bool:
        return self.status == OccupancyStatus.VALID

    @property
    def stable_identity(self) -> Tuple[str, str, str, str, str, str]:
        return (
            self.chain_id,
            self.residue_name,
            self.residue_number,
            self.insertion_code,
            self.atom_name,
            self.altloc,
        )


@dataclass
class AtomSite:
    """One deposited atom record with stable source-model indices."""

    pdb_id: str
    model_index: int
    model_id: str
    chain_index: int
    chain_id: str
    residue_index: int
    residue_name: str
    coordinate_residue_name: str
    residue_number: int
    insertion_code: str
    resnum: str
    atom_index: int
    source_order: int
    atom_name: str
    altloc: str
    element: str
    element_known: bool
    occupancy: float
    occupancy_valid: bool
    occupancy_status: str
    serial: Optional[int]
    x: float
    y: float
    z: float
    is_water: bool
    is_hydrogen: bool
    gemmi_atom: gemmi.Atom = field(repr=False, compare=False)
    # The Gemmi atom belongs to the legacy-PDB identity EDSTATS sees.  Usually
    # that identity is also the source author identity.  Very large mmCIF
    # structures need a packed one-character namespace, in which case these
    # fields retain the coordinate-side half of the reversible mapping while
    # ``chain_id``/``resnum`` above remain the source identity reported in CSVs.
    coordinate_chain_id: str = ""
    coordinate_residue_number: Optional[int] = None
    coordinate_insertion_code: str = ""
    coordinate_resnum: str = ""
    source_polymer_position: str = ""
    source_chain_index: Optional[int] = None
    source_residue_index: Optional[int] = None

    @property
    def pos(self) -> gemmi.Position:
        return self.gemmi_atom.pos

    @property
    def xyz(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def residue_key(self) -> Tuple[int, int, int]:
        return self.model_index, self.chain_index, self.residue_index

    @property
    def output_chain_index(self) -> int:
        return (
            self.chain_index
            if self.source_chain_index is None
            else self.source_chain_index
        )

    @property
    def output_residue_index(self) -> int:
        return (
            self.residue_index
            if self.source_residue_index is None
            else self.source_residue_index
        )

    @property
    def source_key(self) -> Tuple[int, int, int, int]:
        return (*self.residue_key, self.atom_index)

    @property
    def exact_identity(self) -> Tuple[object, ...]:
        return (*self.residue_key, self.atom_name, self.altloc, self.element)

    @property
    def chemical_site_identity(self) -> Tuple[object, ...]:
        return (*self.residue_key, self.atom_name, self.element)


@dataclass
class ResidueSelection:
    """All source sites and the selected canonical sites for one residue."""

    model_index: int
    model_id: str
    chain_index: int
    chain_id: str
    residue_index: int
    residue_name: str
    coordinate_residue_name: str
    residue_number: int
    insertion_code: str
    resnum: str
    is_water: bool
    source_atoms: Tuple[AtomSite, ...]
    contact_atoms: Tuple[AtomSite, ...]
    selected_altloc: str
    selected_conformer_mean_occupancy: Optional[float]
    altloc_options: str
    alternative_conformers_present: bool
    altloc_selection_fallback: bool
    selected_over_blank_duplicate_count: int
    malformed_duplicate_atom_name_count: int
    chemical_atom_site_count: int
    coordinate_chain_id: str = ""
    coordinate_residue_number: Optional[int] = None
    coordinate_insertion_code: str = ""
    coordinate_resnum: str = ""
    source_polymer_position: str = ""
    source_chain_index: Optional[int] = None
    source_residue_index: Optional[int] = None

    @property
    def key(self) -> Tuple[int, int, int]:
        return self.model_index, self.chain_index, self.residue_index

    @property
    def author_key(self) -> Tuple[str, str, str]:
        return self.residue_name, self.chain_id, self.resnum

    @property
    def coordinate_author_key(self) -> Tuple[str, str, str]:
        return (
            self.coordinate_residue_name,
            self.coordinate_chain_id or self.chain_id,
            self.coordinate_resnum or self.resnum,
        )

    @property
    def elements(self) -> frozenset[str]:
        return frozenset(
            atom.element for atom in self.source_atoms if atom.element_known
        )


@dataclass
class StructureContext:
    """Gemmi structure plus deterministic first-model analysis metadata."""

    pdb_id: str
    path: str
    structure: gemmi.Structure
    model: gemmi.Model
    model_index: int
    model_policy: str
    input_model_count: int
    model_analyzed: int
    analyzed_model_id: str
    multi_model_structure: bool
    source_atoms: Tuple[AtomSite, ...]
    contact_atoms: Tuple[AtomSite, ...]
    residues: Tuple[ResidueSelection, ...]
    occupancy_validation_failed: bool
    missing_occupancy_count: int
    invalid_occupancy_count: int
    zero_occupancy_atom_count: int
    duplicate_atom_records_present: bool
    duplicate_atom_record_count: int
    duplicate_coordinate_conflict_count: int
    malformed_duplicate_atom_name_count: int
    unknown_element_atom_count: int
    element_validation_warning: str
    raw_occupancy_mapping_failed: bool
    raw_occupancy_mapping_failure_reason: str
    symmetry_search_available: bool
    symmetry_search_failure_reason: str
    crystallographic_operation_count: int
    strict_ncs_operation_ids: Tuple[str, ...]
    analysis_coordinate_format: str
    warning_codes: Tuple[str, ...]
    _atom_by_indices: Mapping[Tuple[int, int, int], AtomSite] = field(
        repr=False, default_factory=dict
    )
    _residue_by_key: Mapping[Tuple[int, int, int], ResidueSelection] = field(
        repr=False, default_factory=dict
    )
    _residues_by_author: Mapping[Tuple[str, str, str], Tuple[ResidueSelection, ...]] = (
        field(repr=False, default_factory=dict)
    )
    _residues_by_source_author: Mapping[
        Tuple[str, str, str], Tuple[ResidueSelection, ...]
    ] = field(repr=False, default_factory=dict)
    _residues_by_coordinate_author: Mapping[
        Tuple[str, str, str], Tuple[ResidueSelection, ...]
    ] = field(repr=False, default_factory=dict)

    @property
    def strict_ncs_operation_count(self) -> int:
        return len(self.strict_ncs_operation_ids)

    @property
    def dpi_atom_count_multiplier(self) -> int:
        """Number of full coordinate copies represented in the asymmetric unit."""
        return 1 + self.strict_ncs_operation_count

    def image_provenance(
        self,
        image_index: int,
        cell_translation: Tuple[int, int, int],
    ) -> Tuple[bool, bool, str, str]:
        """Classify a Gemmi cell image as crystallographic and/or strict NCS.

        ``setup_cell_images()`` orders the identity first, followed by the
        remaining space-group operations. Each strict-NCS transform then adds
        one complete block of space-group operations. A nonzero unit-cell
        translation is also crystallographic, including when combined with NCS.
        See https://gemmi.readthedocs.io/en/stable/analysis.html.
        """
        if image_index < 0:
            raise ValueError("Gemmi image index cannot be negative")
        operation_count = self.crystallographic_operation_count
        if operation_count <= 0:
            raise ValueError("crystallographic operation count is unavailable")

        strict_ncs = image_index >= operation_count
        crystallographic = image_index % operation_count != 0 or cell_translation != (
            0,
            0,
            0,
        )
        ncs_operation_id = ""
        if strict_ncs:
            ncs_index = image_index // operation_count - 1
            if 0 <= ncs_index < len(self.strict_ncs_operation_ids):
                ncs_operation_id = self.strict_ncs_operation_ids[ncs_index]
            else:
                raise ValueError(
                    f"Gemmi image index {image_index} has no strict-NCS operation"
                )

        if strict_ncs and crystallographic:
            scope = ContactScope.STRICT_NCS_AND_CRYSTALLOGRAPHIC
        elif strict_ncs:
            scope = ContactScope.STRICT_NCS
        elif crystallographic:
            scope = ContactScope.CRYSTALLOGRAPHIC
        else:
            scope = ContactScope.EXPLICIT
        return crystallographic, strict_ncs, ncs_operation_id, scope

    def atom_for_indices(
        self, chain_index: int, residue_index: int, atom_index: int
    ) -> Optional[AtomSite]:
        return self._atom_by_indices.get((chain_index, residue_index, atom_index))

    def atom_for_mark(self, mark: gemmi.NeighborSearch.Mark) -> Optional[AtomSite]:
        return self.atom_for_indices(mark.chain_idx, mark.residue_idx, mark.atom_idx)

    def residue_for_atom(self, atom: AtomSite) -> ResidueSelection:
        return self._residue_by_key[atom.residue_key]

    def residues_for_author(
        self, residue_name: str, chain_id: str, resnum: str
    ) -> Tuple[ResidueSelection, ...]:
        return self._residues_by_author.get(
            (str(residue_name), str(chain_id), str(resnum)), ()
        )

    def residues_for_source_author(
        self, residue_name: str, chain_id: str, resnum: str
    ) -> Tuple[ResidueSelection, ...]:
        return self._residues_by_source_author.get(
            (str(residue_name), str(chain_id), str(resnum)), ()
        )

    def residues_for_coordinate_author(
        self, residue_name: str, chain_id: str, resnum: str
    ) -> Tuple[ResidueSelection, ...]:
        return self._residues_by_coordinate_author.get(
            (str(residue_name), str(chain_id), str(resnum)), ()
        )

    def metal_atoms(
        self, elements: Iterable[str], canonical: bool = True
    ) -> List[AtomSite]:
        wanted = {str(element).upper() for element in elements}
        atoms = self.contact_atoms if canonical else self.source_atoms
        return [atom for atom in atoms if atom.element_known and atom.element in wanted]

    def make_neighbor_search(
        self,
        radius: float,
        include_symmetry: bool = True,
        positive_occupancy_only: bool = True,
    ) -> gemmi.NeighborSearch:
        if include_symmetry:
            if not self.symmetry_search_available:
                raise ValueError(
                    self.symmetry_search_failure_reason
                    or "symmetry image search is unavailable"
                )
            cell = self.structure.cell
        else:
            cell = gemmi.UnitCell()
        search = gemmi.NeighborSearch(self.model, cell, radius)
        for atom in self.contact_atoms:
            if positive_occupancy_only and not (
                atom.occupancy_valid and atom.occupancy > 0
            ):
                continue
            search.add_atom(
                atom.gemmi_atom, atom.chain_index, atom.residue_index, atom.atom_index
            )
        return search


def _raw_pdb_occupancies(path: str) -> Tuple[List[List[RawOccupancy]], str]:
    """Read PDB occupancy and element fields without losing provenance."""
    records: List[List[RawOccupancy]] = [[]]
    model_index = 0
    saw_model = False
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = line[:6].strip().upper()
                if record == "MODEL":
                    if saw_model:
                        model_index += 1
                        records.append([])
                    else:
                        saw_model = True
                    continue
                if record not in ("ATOM", "HETATM"):
                    continue
                while model_index >= len(records):
                    records.append([])
                source_order = len(records[model_index])
                serial_text = line[6:11].strip()
                try:
                    serial = int(serial_text)
                except (TypeError, ValueError, OverflowError):
                    serial = None
                atom_name = line[12:16].strip()
                altloc = blank_if_missing(line[16:17])
                residue_name = line[17:20].strip()
                chain_id = line[21:22].strip()
                residue_number = str(_decode_pdb_resseq(line[22:26]))
                insertion_code = blank_if_missing(line[26:27])
                element, element_status = _parse_pdb_element(line[76:78])
                text = line[54:60] if len(line) >= 55 else ""
                stripped = text.strip()
                if not stripped:
                    value = None
                    status = OccupancyStatus.MISSING
                else:
                    try:
                        value = float(stripped)
                    except (TypeError, ValueError, OverflowError):
                        value = None
                        status = OccupancyStatus.INVALID_NON_NUMERIC
                    else:
                        if not math.isfinite(value):
                            status = OccupancyStatus.INVALID_NON_FINITE
                        elif value < 0.0 or value > 1.0:
                            status = OccupancyStatus.INVALID_RANGE
                        else:
                            status = OccupancyStatus.VALID
                records[model_index].append(
                    RawOccupancy(
                        value=value,
                        status=status,
                        element=element,
                        element_status=element_status,
                        atom_name=atom_name,
                        altloc=altloc,
                        chain_id=chain_id,
                        residue_name=residue_name,
                        residue_number=residue_number,
                        insertion_code=insertion_code,
                        serial=serial,
                        source_order=source_order,
                    )
                )
    except OSError as exc:
        return [], str(exc)
    return records, ""


@dataclass(frozen=True)
class SourceResidueIdentity:
    """Source-mmCIF identity for one residue in an analysis PDB."""

    residue_name: str
    chain_id: str
    residue_number: Optional[int] = None
    insertion_code: str = ""
    polymer_position: str = ""
    chain_index: Optional[int] = None
    residue_index: Optional[int] = None


def _raw_pdb_residue_mapping(
    path: str,
) -> Dict[Tuple[int, str, str, str], SourceResidueIdentity]:
    """Read reversible mmCIF residue mappings embedded during conversion.

    ``RESNAME`` is the original, name-only format.  ``RESIDUE`` is emitted for
    a structure whose residues had to be packed into a PDB-safe namespace and
    restores the source component, chain, sequence number and insertion code.
    """
    mapping: Dict[Tuple[int, str, str, str], SourceResidueIdentity] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            prefix = " ".join(fields[:4])
            if prefix not in (RESNAME_REMARK_PREFIX, RESIDUE_REMARK_PREFIX):
                continue
            expected_fields = 9 if prefix == RESNAME_REMARK_PREFIX else 15
            if len(fields) != expected_fields:
                raise ValueError(f"malformed Alchemy residue mapping: {line.rstrip()}")
            try:
                model_index = int(fields[4]) - 1
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid model in residue mapping: {fields[4]!r}"
                ) from exc
            if model_index < 0:
                raise ValueError("residue mapping model must be positive")
            coordinate_chain = "" if fields[5] == "_" else fields[5]
            coordinate_resnum, coordinate_name = fields[6:8]
            key = (
                model_index,
                coordinate_name,
                coordinate_chain,
                coordinate_resnum,
            )
            if prefix == RESNAME_REMARK_PREFIX:
                source = SourceResidueIdentity(
                    residue_name=fields[8],
                    chain_id=coordinate_chain,
                )
            else:
                source_chain = "" if fields[8] == "_" else fields[8]
                try:
                    source_number = int(fields[9])
                    source_chain_index = int(fields[12])
                    source_residue_index = int(fields[13])
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "invalid source numeric field in residue mapping: "
                        f"{line.rstrip()}"
                    ) from exc
                source = SourceResidueIdentity(
                    residue_name=fields[11],
                    chain_id=source_chain,
                    residue_number=source_number,
                    insertion_code=("" if fields[10] == "_" else fields[10]),
                    chain_index=source_chain_index,
                    residue_index=source_residue_index,
                    polymer_position=fields[14],
                )
            previous = mapping.get(key)
            # A full RESIDUE record deliberately supersedes a preceding legacy
            # RESNAME record for the same coordinate residue.
            if previous is not None:
                compatible_legacy_upgrade = (
                    prefix == RESIDUE_REMARK_PREFIX
                    and previous.residue_number is None
                    and previous.residue_name == source.residue_name
                )
                if previous != source and not compatible_legacy_upgrade:
                    raise ValueError(
                        "conflicting Alchemy residue mappings for "
                        f"{coordinate_name}/{coordinate_chain}/"
                        f"{coordinate_resnum}"
                    )
            mapping[key] = source
    return mapping


def _gemmi_atom_identity(
    chain: gemmi.Chain, residue: gemmi.Residue, atom: gemmi.Atom
) -> Tuple[str, str, str, str, str, str]:
    return (
        str(chain.name),
        str(residue.name),
        str(_residue_number(residue)),
        blank_if_missing(residue.seqid.icode),
        str(atom.name).strip(),
        blank_if_missing(atom.altloc),
    )


def _match_raw_occupancies(
    model: gemmi.Model,
    raw_records: Sequence[RawOccupancy],
) -> Tuple[List[Optional[RawOccupancy]], int, int]:
    """Match PDB records to Gemmi atoms without relying on traversal order.

    Gemmi may merge repeated chain segments when reading a PDB file. Stable
    author identifiers locate each record after that regrouping; atom serials
    disambiguate malformed duplicate identifiers when possible.
    """
    raw_by_identity: Dict[Tuple[str, str, str, str, str, str], List[RawOccupancy]] = (
        defaultdict(list)
    )
    for raw in raw_records:
        raw_by_identity[raw.stable_identity].append(raw)

    matches: List[Optional[RawOccupancy]] = []
    unmatched_gemmi = 0
    for chain in model:
        for residue in chain:
            for atom in residue:
                candidates = raw_by_identity.get(
                    _gemmi_atom_identity(chain, residue, atom), []
                )
                if not candidates:
                    matches.append(None)
                    unmatched_gemmi += 1
                    continue

                match_index = 0
                serial = atom.serial
                if serial is not None:
                    for index, candidate in enumerate(candidates):
                        if candidate.serial == serial:
                            match_index = index
                            break
                matches.append(candidates.pop(match_index))

    unmatched_raw = sum(len(records) for records in raw_by_identity.values())
    return matches, unmatched_gemmi, unmatched_raw


def _occupancy_for_atom(
    atom: gemmi.Atom, raw: Optional[RawOccupancy]
) -> Tuple[float, bool, str]:
    value = float(atom.occ)
    if raw is not None:
        if raw.valid and raw.value is not None:
            return raw.value, True, OccupancyStatus.VALID
        return value, False, raw.status
    valid = _valid_occupancy(value)
    return (
        value,
        valid,
        OccupancyStatus.VALID if valid else OccupancyStatus.INVALID_VALUE,
    )


def _element_for_atom(
    atom: gemmi.Atom, analysis_format: str, raw: Optional[RawOccupancy]
) -> Tuple[str, bool]:
    """Use deposited element provenance rather than a PDB atom-name guess."""
    if analysis_format == "pdb":
        if raw is not None and raw.element_status == ElementStatus.VALID:
            return raw.element, True
        return "", False
    element = str(atom.element.name).upper()
    return element, element not in ("", "X")


def _site_is_better(candidate: AtomSite, current: AtomSite) -> bool:
    """Highest valid occupancy wins; exact ties retain source-file order."""
    if candidate.occupancy_valid != current.occupancy_valid:
        return candidate.occupancy_valid
    if candidate.occupancy_valid and candidate.occupancy != current.occupancy:
        return candidate.occupancy > current.occupancy
    return candidate.source_order < current.source_order


def _select_residue(atoms: Sequence[AtomSite]) -> ResidueSelection:
    first = atoms[0]
    named: Dict[str, List[AtomSite]] = defaultdict(list)
    blank: List[AtomSite] = []
    for atom in atoms:
        (named[atom.altloc] if atom.altloc else blank).append(atom)

    option_means: Dict[str, Optional[float]] = {}
    for label in sorted(named):
        valid = [atom.occupancy for atom in named[label] if atom.occupancy_valid]
        option_means[label] = sum(valid) / len(valid) if valid else None

    fallback = False
    selected_altloc = ""
    selected_mean: Optional[float]
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
                if math.isclose(mean, best_mean, rel_tol=0.0, abs_tol=1e-12)
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
    # atom name. Any remaining duplicate name is resolved by valid occupancy.
    by_name: Dict[str, List[AtomSite]] = defaultdict(list)
    for atom in candidates:
        by_name[atom.atom_name].append(atom)
    contact_atoms: List[AtomSite] = []
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
            if _site_is_better(atom, winner):
                winner = atom
        contact_atoms.append(winner)
    contact_atoms.sort(key=lambda atom: atom.source_order)

    options = "|".join(
        f"{label}:{_format_number(option_means[label])}"
        for label in sorted(option_means)
    )
    chemical_sites = len({atom.chemical_site_identity for atom in atoms})
    return ResidueSelection(
        model_index=first.model_index,
        model_id=first.model_id,
        chain_index=first.chain_index,
        chain_id=first.chain_id,
        residue_index=first.residue_index,
        residue_name=first.residue_name,
        coordinate_residue_name=first.coordinate_residue_name,
        residue_number=first.residue_number,
        insertion_code=first.insertion_code,
        resnum=first.resnum,
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
        coordinate_chain_id=first.coordinate_chain_id,
        coordinate_residue_number=first.coordinate_residue_number,
        coordinate_insertion_code=first.coordinate_insertion_code,
        coordinate_resnum=first.coordinate_resnum,
        source_polymer_position=first.source_polymer_position,
        source_chain_index=first.source_chain_index,
        source_residue_index=first.source_residue_index,
    )


def _symmetry_metadata(
    structure: gemmi.Structure,
) -> Tuple[bool, str, int, Tuple[str, ...]]:
    strict_ncs_ids = tuple(
        str(operation.id).strip() or f"strict_ncs_{index}"
        for index, operation in enumerate(
            (operation for operation in structure.ncs if not bool(operation.given)),
            start=1,
        )
    )
    try:
        cell = structure.cell
        if cell is None or not cell.is_crystal() or cell.volume <= 0:
            return False, "missing_or_invalid_unit_cell", 0, strict_ncs_ids
        spacegroup = structure.find_spacegroup()
        if spacegroup is None:
            return False, "missing_or_invalid_space_group", 0, strict_ncs_ids
        operation_count = len(list(spacegroup.operations()))
        if operation_count <= 0:
            return False, "missing_or_invalid_space_group", 0, strict_ncs_ids
        structure.setup_cell_images()
        return True, "", operation_count, strict_ncs_ids
    except Exception as exc:  # one malformed entry must not stop a batch
        return (
            False,
            f"symmetry_setup_failed:{type(exc).__name__}",
            0,
            strict_ncs_ids,
        )


def load_structure(
    pdb_id: str,
    path: str,
    source_model_count: Optional[int] = None,
) -> StructureContext:
    """Parse ``path`` with Gemmi and build Alchemy's first-model atom sets."""
    structure = gemmi.read_structure(path)
    if len(structure) == 0:
        raise ValueError("coordinate file contains no models")
    input_model_count = (
        len(structure) if source_model_count is None else source_model_count
    )
    if input_model_count < len(structure):
        raise ValueError(
            "source model count cannot be smaller than the analysis model count"
        )
    model = structure[0]
    model_id = str(model.num)
    analysis_format = (
        "mmcif" if os.path.splitext(path)[1].lower() in (".cif", ".mmcif") else "pdb"
    )

    raw_models: List[List[RawOccupancy]] = []
    raw_error = ""
    source_residue_identities: Dict[
        Tuple[int, str, str, str], SourceResidueIdentity
    ] = {}
    if analysis_format == "pdb":
        raw_models, raw_error = _raw_pdb_occupancies(path)
        source_residue_identities = _raw_pdb_residue_mapping(path)
    legacy_identifiers_packed = any(
        identity.residue_number is not None
        for identity in source_residue_identities.values()
    )
    raw_first = raw_models[0] if raw_models else []
    gemmi_atoms = [atom for chain in model for residue in chain for atom in residue]
    gemmi_atom_count = len(gemmi_atoms)
    raw_matches: List[Optional[RawOccupancy]]
    unmatched_gemmi = 0
    unmatched_raw = 0
    if analysis_format == "pdb":
        raw_matches, unmatched_gemmi, unmatched_raw = _match_raw_occupancies(
            model, raw_first
        )
    else:
        raw_matches = [None] * gemmi_atom_count
    mapping_failed = analysis_format == "pdb" and bool(
        raw_error or unmatched_gemmi or unmatched_raw
    )
    mapping_reason = raw_error
    if mapping_failed and not mapping_reason:
        if len(raw_first) != gemmi_atom_count:
            mapping_reason = (
                f"raw_atom_count_mismatch:{len(raw_first)}!="
                f"{gemmi_atom_count};unmatched_gemmi="
                f"{unmatched_gemmi};unmatched_raw={unmatched_raw}"
            )
        else:
            mapping_reason = (
                f"raw_atom_identity_mismatch:unmatched_gemmi="
                f"{unmatched_gemmi};unmatched_raw={unmatched_raw}"
            )

    all_sites: List[AtomSite] = []
    gemmi_order = 0
    for chain_index, chain in enumerate(model):
        for residue_index, residue in enumerate(chain):
            coordinate_number = _residue_number(residue)
            coordinate_insertion = blank_if_missing(residue.seqid.icode)
            coordinate_resnum = f"{coordinate_number}{coordinate_insertion}"
            coordinate_residue_name = str(residue.name)
            coordinate_chain_id = str(chain.name)
            source_identity = source_residue_identities.get(
                (0, coordinate_residue_name, coordinate_chain_id, coordinate_resnum),
                SourceResidueIdentity(
                    residue_name=coordinate_residue_name,
                    chain_id=coordinate_chain_id,
                    residue_number=coordinate_number,
                    insertion_code=coordinate_insertion,
                ),
            )
            source_number = (
                coordinate_number
                if source_identity.residue_number is None
                else source_identity.residue_number
            )
            source_insertion = (
                coordinate_insertion
                if source_identity.residue_number is None
                else source_identity.insertion_code
            )
            source_resnum = f"{source_number}{source_insertion}"
            for atom_index, atom in enumerate(residue):
                raw = (
                    raw_matches[gemmi_order] if gemmi_order < len(raw_matches) else None
                )
                occupancy, occupancy_valid, occupancy_status = _occupancy_for_atom(
                    atom, raw
                )
                if analysis_format == "pdb" and raw is None:
                    occupancy_valid = False
                    occupancy_status = OccupancyStatus.RAW_MAPPING_FAILED
                source_order = (
                    raw.source_order
                    if raw is not None
                    else len(raw_first) + gemmi_order
                )
                element, element_known = _element_for_atom(atom, analysis_format, raw)
                all_sites.append(
                    AtomSite(
                        pdb_id=pdb_id,
                        model_index=0,
                        model_id=model_id,
                        chain_index=chain_index,
                        chain_id=source_identity.chain_id,
                        residue_index=residue_index,
                        residue_name=source_identity.residue_name,
                        coordinate_residue_name=coordinate_residue_name,
                        residue_number=source_number,
                        insertion_code=source_insertion,
                        resnum=source_resnum,
                        atom_index=atom_index,
                        source_order=source_order,
                        atom_name=str(atom.name).strip(),
                        altloc=blank_if_missing(atom.altloc),
                        element=element,
                        element_known=element_known,
                        occupancy=occupancy,
                        occupancy_valid=occupancy_valid,
                        occupancy_status=occupancy_status,
                        serial=atom.serial,
                        x=float(atom.pos.x),
                        y=float(atom.pos.y),
                        z=float(atom.pos.z),
                        is_water=bool(residue.is_water()),
                        is_hydrogen=(element_known and element in ("H", "D")),
                        gemmi_atom=atom,
                        coordinate_chain_id=coordinate_chain_id,
                        coordinate_residue_number=coordinate_number,
                        coordinate_insertion_code=coordinate_insertion,
                        coordinate_resnum=coordinate_resnum,
                        source_polymer_position=source_identity.polymer_position,
                        source_chain_index=source_identity.chain_index,
                        source_residue_index=source_identity.residue_index,
                    )
                )
                gemmi_order += 1

    # Occupancy provenance is counted before malformed exact duplicates are
    # collapsed, so an invalid deposited record is never silently hidden.
    missing_count = sum(
        atom.occupancy_status == OccupancyStatus.MISSING for atom in all_sites
    )
    invalid_count = sum(
        not atom.occupancy_valid and atom.occupancy_status != OccupancyStatus.MISSING
        for atom in all_sites
    )
    zero_count = sum(
        atom.occupancy_valid and atom.occupancy == 0.0 for atom in all_sites
    )

    dedup: Dict[Tuple[object, ...], AtomSite] = {}
    duplicate_count = 0
    coordinate_conflicts = 0
    for site in all_sites:
        current = dedup.get(site.exact_identity)
        if current is None:
            dedup[site.exact_identity] = site
            continue
        duplicate_count += 1
        if position_distance(site.xyz, current.xyz) > DUPLICATE_ATOM_POSITION_TOLERANCE:
            coordinate_conflicts += 1
        if _site_is_better(site, current):
            dedup[site.exact_identity] = site
    source_atoms = tuple(sorted(dedup.values(), key=lambda site: site.source_order))

    residue_groups: Dict[Tuple[int, int, int], List[AtomSite]] = defaultdict(list)
    for site in source_atoms:
        residue_groups[site.residue_key].append(site)
    residues = tuple(
        _select_residue(residue_groups[key]) for key in sorted(residue_groups)
    )
    contact_atoms = tuple(
        atom for residue in residues for atom in residue.contact_atoms
    )
    malformed_names = sum(
        residue.malformed_duplicate_atom_name_count
        + residue.selected_over_blank_duplicate_count
        for residue in residues
    )
    unknown_count = sum(not atom.element_known for atom in source_atoms)
    (
        symmetry_available,
        symmetry_reason,
        crystallographic_operation_count,
        strict_ncs_operation_ids,
    ) = _symmetry_metadata(structure)

    warnings: List[str] = []
    if input_model_count > 1:
        warnings.append(WarningCode.MULTI_MODEL_STRUCTURE)
    if duplicate_count:
        warnings.append(WarningCode.DUPLICATE_ATOM_RECORDS)
    if coordinate_conflicts:
        warnings.append(WarningCode.DUPLICATE_ATOM_COORDINATE_CONFLICT)
    if malformed_names:
        warnings.append(WarningCode.MALFORMED_DUPLICATE_ATOM_NAMES)
    if any(residue.altloc_selection_fallback for residue in residues):
        warnings.append(WarningCode.ALTLOC_SELECTION_FALLBACK)
    if unknown_count:
        warnings.append(WarningCode.UNKNOWN_ELEMENTS)
    if zero_count:
        warnings.append(WarningCode.ZERO_OCCUPANCY_ATOMS)
    if mapping_failed:
        warnings.append(WarningCode.RAW_OCCUPANCY_MAPPING_FAILED)
    if legacy_identifiers_packed:
        warnings.append(WarningCode.LEGACY_PDB_IDENTIFIERS_PACKED)

    atom_by_indices = {
        (atom.chain_index, atom.residue_index, atom.atom_index): atom
        for atom in contact_atoms
    }
    residue_by_key = {residue.key: residue for residue in residues}
    by_author_lists: Dict[Tuple[str, str, str], List[ResidueSelection]] = defaultdict(
        list
    )
    by_source_author_lists: Dict[Tuple[str, str, str], List[ResidueSelection]] = (
        defaultdict(list)
    )
    by_coordinate_author_lists: Dict[Tuple[str, str, str], List[ResidueSelection]] = (
        defaultdict(list)
    )
    for selection in residues:
        by_source_author_lists[selection.author_key].append(selection)
        by_coordinate_author_lists[selection.coordinate_author_key].append(selection)
        by_author_lists[selection.author_key].append(selection)
        if selection.coordinate_author_key != selection.author_key:
            by_author_lists[selection.coordinate_author_key].append(selection)
    residues_by_author = {key: tuple(value) for key, value in by_author_lists.items()}
    residues_by_source_author = {
        key: tuple(value) for key, value in by_source_author_lists.items()
    }
    residues_by_coordinate_author = {
        key: tuple(value) for key, value in by_coordinate_author_lists.items()
    }

    occupancy_failed = bool(missing_count or invalid_count or mapping_failed)
    return StructureContext(
        pdb_id=pdb_id,
        path=path,
        structure=structure,
        model=model,
        model_index=0,
        model_policy="first",
        input_model_count=input_model_count,
        model_analyzed=1,
        analyzed_model_id=model_id,
        multi_model_structure=input_model_count > 1,
        source_atoms=source_atoms,
        contact_atoms=contact_atoms,
        residues=residues,
        occupancy_validation_failed=occupancy_failed,
        missing_occupancy_count=missing_count,
        invalid_occupancy_count=invalid_count,
        zero_occupancy_atom_count=zero_count,
        duplicate_atom_records_present=duplicate_count > 0,
        duplicate_atom_record_count=duplicate_count,
        duplicate_coordinate_conflict_count=coordinate_conflicts,
        malformed_duplicate_atom_name_count=malformed_names,
        unknown_element_atom_count=unknown_count,
        element_validation_warning=("unknown_element_atoms" if unknown_count else ""),
        raw_occupancy_mapping_failed=mapping_failed,
        raw_occupancy_mapping_failure_reason=mapping_reason,
        symmetry_search_available=symmetry_available,
        symmetry_search_failure_reason=symmetry_reason,
        crystallographic_operation_count=crystallographic_operation_count,
        strict_ncs_operation_ids=strict_ncs_operation_ids,
        analysis_coordinate_format=analysis_format,
        warning_codes=tuple(warnings),
        _atom_by_indices=atom_by_indices,
        _residue_by_key=residue_by_key,
        _residues_by_author=residues_by_author,
        _residues_by_source_author=residues_by_source_author,
        _residues_by_coordinate_author=residues_by_coordinate_author,
    )


def count_deposited_ni(context: StructureContext) -> float:
    """Occupancy-weighted non-H/D count in deposited first-model records."""
    if context.occupancy_validation_failed or context.unknown_element_atom_count:
        return NAN
    total = 0.0
    for atom in context.source_atoms:
        if atom.is_hydrogen:
            continue
        if not atom.occupancy_valid:
            return NAN
        total += atom.occupancy
    return total


def count_ni(context: StructureContext) -> float:
    """Occupancy-weighted non-H/D count for the complete asymmetric unit.

    Gemmi does not add strict-NCS copies to ``model``. Each non-given NCS
    operation represents one additional full copy in the asymmetric unit, so
    DPI must count those copies even though their atoms are not deposited.
    """
    deposited = count_deposited_ni(context)
    if not math.isfinite(deposited):
        return NAN
    return deposited * context.dpi_atom_count_multiplier
