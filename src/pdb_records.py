"""Raw PDB atom-record fields and their match to Gemmi atoms.

Gemmi normalizes occupancy and element while parsing. Alchemy re-reads the
deposited columns so that missing, malformed, and out-of-range values stay
visible, then joins each record to its Gemmi atom by author identity rather
than by traversal order.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import gemmi

from codes import ElementStatus, OccupancyStatus

# Blank PDB columns, Gemmi's NUL altloc, and the two mmCIF null tokens all mean
# "no value here".
MISSING_VALUE_TOKENS = ("", " ", "\x00", ".", "?")
PDB_HYBRID36_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#: Width of the PDB resSeq field; wider identifiers are hybrid-36 encoded.
PDB_RESSEQ_WIDTH = 4

# Fixed PDB coordinate-record columns as zero-based slices.
RECORD_NAME_COLUMNS = slice(0, 6)
SERIAL_COLUMNS = slice(6, 11)
ATOM_NAME_COLUMNS = slice(12, 16)
ALTLOC_COLUMN = slice(16, 17)
RESIDUE_NAME_COLUMNS = slice(17, 20)
#: Both chain-id columns; Gemmi reads column 21 too when it is not blank.
CHAIN_ID_COLUMNS = slice(20, 22)
RESSEQ_COLUMNS = slice(22, 26)
INSERTION_CODE_COLUMN = slice(26, 27)
OCCUPANCY_COLUMNS = slice(54, 60)
ELEMENT_COLUMNS = slice(76, 78)

MODEL_RECORD_NAME = "MODEL"
COORDINATE_RECORD_NAMES = ("ATOM", "HETATM")

#: Values of ``StructureContext.analysis_coordinate_format``.
PDB_FORMAT = "pdb"
MMCIF_FORMAT = "mmcif"
MMCIF_EXTENSIONS = (".cif", ".mmcif")

#: Gemmi's placeholder element names, which carry no chemistry.
UNKNOWN_ELEMENT_NAMES = ("", "X")

#: ``(chain, residue name, residue number, insertion code, atom name, altloc)``.
AtomIdentityKey = tuple[str, str, str, str, str, str]


def blank_if_missing(value: str) -> str:
    """Normalize a coordinate-format missing token to an empty string."""
    return "" if value in MISSING_VALUE_TOKENS else str(value)


def decode_pdb_resseq(value: str) -> int:
    """Decode a four-column decimal or hybrid-36 PDB resSeq value.

    Matches the integer Gemmi exposes as ``Residue.seqid.num``, which raw PDB
    and EDSTATS identifiers must agree with to join. Gemmi treats PDB resSeq
    letter case as equivalent, hence the upper-casing.
    """
    text = str(value)
    if len(text) > PDB_RESSEQ_WIDTH:
        raise ValueError(f"PDB residue sequence field is wider than 4: {value!r}")
    text = text.rjust(PDB_RESSEQ_WIDTH)
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

    EDSTATS separates an insertion code with ``:``; the compact form is
    accepted too.
    """
    text = str(value).strip()
    insertion = ""
    if ":" in text:
        number_text, insertion = text.split(":", 1)
    elif len(text) > PDB_RESSEQ_WIDTH:
        number_text, insertion = text[:PDB_RESSEQ_WIDTH], text[PDB_RESSEQ_WIDTH:]
    elif text and text[-1].isalpha() and not text[0].isalpha() and text[:-1]:
        number_text, insertion = text[:-1], text[-1]
    else:
        number_text = text
    if len(insertion) > 1:
        raise ValueError(f"invalid PDB insertion code in residue id: {value!r}")
    return f"{decode_pdb_resseq(number_text)}{insertion}"


def author_residue_number(residue: gemmi.Residue) -> int:
    """Return a Gemmi residue's author sequence number, which must be present."""
    number = residue.seqid.num
    if number is None:
        raise ValueError(f"residue {residue.name!r} has no author sequence number")
    return number


def valid_occupancy(value: object) -> bool:
    """Return whether a value is a finite occupancy in the physical range."""
    if not isinstance(value, (int, float, str)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def parse_pdb_element(value: str) -> tuple[str, str]:
    """Parse a deposited PDB element field and return its validation status."""
    deposited = value.strip()
    if not deposited:
        return "", ElementStatus.MISSING
    try:
        element = gemmi.Element(deposited)
    except (RuntimeError, ValueError):
        return "", ElementStatus.INVALID
    canonical = str(element.name).upper()
    if int(element.atomic_number) <= 0 or canonical in UNKNOWN_ELEMENT_NAMES:
        return "", ElementStatus.INVALID
    return canonical, ElementStatus.VALID


def analysis_format_for_path(path: str) -> str:
    """Return the coordinate format Alchemy analyzes ``path`` as."""
    extension = os.path.splitext(path)[1].lower()
    return MMCIF_FORMAT if extension in MMCIF_EXTENSIONS else PDB_FORMAT


@dataclass(frozen=True, slots=True)
class RawOccupancy:
    """Preserve raw PDB occupancy, element, and atom identity fields."""

    value: float | None
    status: str
    element: str = ""
    element_status: str = ElementStatus.MISSING
    atom_name: str = ""
    altloc: str = ""
    chain_id: str = ""
    residue_name: str = ""
    residue_number: str = ""
    insertion_code: str = ""
    serial: int | None = None
    source_order: int = 0

    @property
    def valid(self) -> bool:
        """Return whether the deposited occupancy is usable."""
        return self.status == OccupancyStatus.VALID

    @property
    def stable_identity(self) -> AtomIdentityKey:
        """Return the source identifiers used to match this atom reliably."""
        return (
            self.chain_id,
            self.residue_name,
            self.residue_number,
            self.insertion_code,
            self.atom_name,
            self.altloc,
        )


def _occupancy_field(text: str) -> tuple[float | None, str]:
    """Parse the deposited occupancy column into a value and its status."""
    stripped = text.strip()
    if not stripped:
        return None, OccupancyStatus.MISSING
    try:
        value = float(stripped)
    except (TypeError, ValueError, OverflowError):
        return None, OccupancyStatus.INVALID_NON_NUMERIC
    if not math.isfinite(value):
        return value, OccupancyStatus.INVALID_NON_FINITE
    if value < 0.0 or value > 1.0:
        return value, OccupancyStatus.INVALID_RANGE
    return value, OccupancyStatus.VALID


def _serial_field(text: str) -> int | None:
    serial_text = text.strip()
    try:
        return int(serial_text)
    except (TypeError, ValueError, OverflowError):
        return None


def raw_pdb_occupancies(path: str) -> tuple[list[list[RawOccupancy]], str]:
    """Read PDB occupancy and element fields without losing provenance.

    Returns one record list per MODEL block, in file order, and the error text
    if the file could not be read or a record's resSeq could not be decoded.
    """
    records: list[list[RawOccupancy]] = [[]]
    model_index = 0
    saw_model = False
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = line[RECORD_NAME_COLUMNS].strip().upper()
                if record == MODEL_RECORD_NAME:
                    if saw_model:
                        model_index += 1
                        records.append([])
                    else:
                        saw_model = True
                    continue
                if record not in COORDINATE_RECORD_NAMES:
                    continue
                element, element_status = parse_pdb_element(line[ELEMENT_COLUMNS])
                value, status = _occupancy_field(line[OCCUPANCY_COLUMNS])
                records[model_index].append(
                    RawOccupancy(
                        value=value,
                        status=status,
                        element=element,
                        element_status=element_status,
                        atom_name=line[ATOM_NAME_COLUMNS].strip(),
                        altloc=blank_if_missing(line[ALTLOC_COLUMN]),
                        chain_id=line[CHAIN_ID_COLUMNS].strip(),
                        residue_name=line[RESIDUE_NAME_COLUMNS].strip(),
                        residue_number=str(decode_pdb_resseq(line[RESSEQ_COLUMNS])),
                        insertion_code=blank_if_missing(line[INSERTION_CODE_COLUMN]),
                        serial=_serial_field(line[SERIAL_COLUMNS]),
                        source_order=len(records[model_index]),
                    )
                )
    except (OSError, ValueError) as exc:
        return [], str(exc)
    return records, ""


def _gemmi_atom_identity(
    chain: gemmi.Chain, residue: gemmi.Residue, atom: gemmi.Atom
) -> AtomIdentityKey:
    return (
        str(chain.name),
        str(residue.name),
        str(author_residue_number(residue)),
        blank_if_missing(residue.seqid.icode),
        str(atom.name).strip(),
        blank_if_missing(atom.altloc),
    )


def match_raw_occupancies(
    model: gemmi.Model,
    raw_records: Sequence[RawOccupancy],
) -> tuple[list[RawOccupancy | None], int, int]:
    """Match PDB records to Gemmi atoms without relying on traversal order.

    Gemmi may merge repeated chain segments when reading a PDB file, so records
    are located by author identity and, for malformed duplicates, atom serial.
    Returns one match per Gemmi atom in traversal order, the number of Gemmi
    atoms without a record, and the number of records without a Gemmi atom.
    """
    raw_by_identity: dict[AtomIdentityKey, list[RawOccupancy]] = defaultdict(list)
    for raw in raw_records:
        raw_by_identity[raw.stable_identity].append(raw)

    matches: list[RawOccupancy | None] = []
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
                serial = cast("int | None", atom.serial)
                if serial is not None:
                    for index, candidate in enumerate(candidates):
                        if candidate.serial == serial:
                            match_index = index
                            break
                matches.append(candidates.pop(match_index))

    unmatched_raw = sum(len(records) for records in raw_by_identity.values())
    return matches, unmatched_gemmi, unmatched_raw


def occupancy_for_atom(
    atom: gemmi.Atom, raw: RawOccupancy | None
) -> tuple[float, bool, str]:
    """Return an atom's occupancy, whether it is usable, and its status code.

    A matched raw record is authoritative; without one, Gemmi's parsed value
    is validated directly.
    """
    value = float(atom.occ)
    if raw is not None:
        if raw.valid and raw.value is not None:
            return raw.value, True, OccupancyStatus.VALID
        return value, False, raw.status
    valid = valid_occupancy(value)
    return (
        value,
        valid,
        OccupancyStatus.VALID if valid else OccupancyStatus.INVALID_VALUE,
    )


def element_for_atom(
    atom: gemmi.Atom, analysis_format: str, raw: RawOccupancy | None
) -> tuple[str, bool]:
    """Trust the deposited element field; Gemmi guesses one from the atom name."""
    if analysis_format == PDB_FORMAT:
        if raw is not None and raw.element_status == ElementStatus.VALID:
            return raw.element, True
        return "", False
    element = str(atom.element.name).upper()
    return element, element not in UNKNOWN_ELEMENT_NAMES
