"""Write and read the ``REMARK 950 ALCHEMY`` provenance records of converted PDBs.

Converting mmCIF to legacy PDB loses information the analysis must restore:
component names longer than three characters, author identifiers renamed or
packed into the PDB namespace, each residue's polymer-boundary status, and
occupancies the mmCIF dictionary defaulted. ``coordinate_conversion`` prepends
these records to the converted file and ``structure_loading`` reads them back;
both use the record layouts declared here, so the format is described exactly
once.

Every record is a whitespace-separated line: the prefix names the record type
and the remaining tokens follow that type's :class:`RemarkLayout`. In the
chain-id and insertion-code fields ``_`` stands in for an empty value, which
would otherwise vanish when the line is split, and a value made only of
underscores gains one more underscore so it stays distinct from the
placeholder. Model numbers are one-based on disk and zero-based in the parsed
mappings. Records may appear in any order.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field as dataclass_field, replace

REMARK_PREFIX = "REMARK 950 ALCHEMY"

# ``-`` non-polymer, ``?`` indeterminate, ``M`` internal, ``N``/``C``/``NC`` terminal.
POLYMER_POSITIONS = ("-", "?", "M", "N", "C", "NC")
_EMPTY_FIELD_PLACEHOLDER = "_"
_INTEGER_TOKEN = re.compile(r"-?[0-9]+")


def _encode_placeholder(value: str) -> str:
    """Write an empty value as ``_`` and keep real underscore values distinct."""
    if value.strip(_EMPTY_FIELD_PLACEHOLDER) == "":
        return value + _EMPTY_FIELD_PLACEHOLDER
    return value


def _decode_placeholder(token: str) -> str:
    """Invert :func:`_encode_placeholder` when reading a record."""
    if token.strip(_EMPTY_FIELD_PLACEHOLDER) == "":
        return token[:-1]
    return token


@dataclass(frozen=True, slots=True)
class RemarkLayout:
    """The fields of one record type, in the order they follow its prefix.

    ``placeholder_fields`` names the fields whose value may be empty; the
    writer and reader apply the placeholder encoding to exactly those.
    """

    prefix: str
    fields: tuple[str, ...]
    placeholder_fields: frozenset[str] = frozenset()
    prefix_tokens: tuple[str, ...] = dataclass_field(init=False)
    _offsets: dict[str, int] = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the placeholder fields and cache the token positions."""
        unknown = self.placeholder_fields - set(self.fields)
        if unknown:
            raise ValueError(f"unknown placeholder fields {sorted(unknown)}")
        prefix_tokens = tuple(self.prefix.split())
        object.__setattr__(self, "prefix_tokens", prefix_tokens)
        object.__setattr__(
            self,
            "_offsets",
            {
                name: len(prefix_tokens) + index
                for index, name in enumerate(self.fields)
            },
        )

    @property
    def token_count(self) -> int:
        """Number of whitespace-separated tokens in a well-formed record."""
        return len(self.prefix_tokens) + len(self.fields)

    def matches(self, tokens: Sequence[str]) -> bool:
        """Whether a split line starts with this layout's prefix."""
        return tuple(tokens[: len(self.prefix_tokens)]) == self.prefix_tokens

    def field(self, tokens: Sequence[str], name: str) -> str:
        """Return the decoded value of ``name`` in a split, length-checked record."""
        token = tokens[self._offsets[name]]
        if name in self.placeholder_fields:
            return _decode_placeholder(token)
        return token

    def format(self, *values: object) -> str:
        """Render one record line from ``values`` given in layout order.

        Every value must render as exactly one token, so the line splits back
        into ``token_count`` tokens; only ``placeholder_fields`` may be empty.
        """
        if len(values) != len(self.fields):
            raise ValueError(
                f"{self.prefix!r} takes {len(self.fields)} fields, got {len(values)}"
            )
        tokens = []
        for name, value in zip(self.fields, values, strict=True):
            token = str(value)
            if name in self.placeholder_fields:
                token = _encode_placeholder(token)
            if token.split() != [token]:
                raise ValueError(
                    f"{self.prefix!r} field {name!r} must be one non-empty token, "
                    f"got {value!r}"
                )
            tokens.append(token)
        return " ".join((self.prefix, *tokens)) + "\n"


# Record layouts. The first four fields of the residue-keyed types identify
# the residue as it appears in the converted PDB.
_RESIDUE_KEY_FIELDS = ("model", "chain", "resnum", "written_name")
RESNAME_LAYOUT = RemarkLayout(
    f"{REMARK_PREFIX} RESNAME",
    (*_RESIDUE_KEY_FIELDS, "source_name"),
    frozenset({"chain"}),
)
RESIDUE_LAYOUT = RemarkLayout(
    f"{REMARK_PREFIX} RESIDUE",
    (
        *_RESIDUE_KEY_FIELDS,
        "source_chain",
        "source_number",
        "source_insertion",
        "source_name",
        "source_chain_index",
        "source_residue_index",
        "source_polymer_position",
    ),
    frozenset({"chain", "source_chain", "source_insertion"}),
)
POLYMER_LAYOUT = RemarkLayout(
    f"{REMARK_PREFIX} POLYMER",
    (*_RESIDUE_KEY_FIELDS, "polymer_position"),
    frozenset({"chain"}),
)
OCCUPANCY_DEFAULT_LAYOUT = RemarkLayout(
    f"{REMARK_PREFIX} OCCUPANCY DEFAULTED", ("model", "count")
)
LAYOUTS = (RESNAME_LAYOUT, RESIDUE_LAYOUT, POLYMER_LAYOUT, OCCUPANCY_DEFAULT_LAYOUT)

# Writer inputs: one tuple per record, in the matching layout's field order,
# with one-based model numbers.
ResnameRecord = tuple[int, str, str, str, str]
ResidueIdentityRecord = tuple[int, str, str, str, str, int, str, str, int, int, str]
PolymerPositionRecord = tuple[int, str, str, str, str]

# (zero-based model index, written residue name, chain id, resnum with insertion code)
ResidueMappingKey = tuple[int, str, str, str]


@dataclass(frozen=True, slots=True)
class SourceResidueIdentity:
    """Source-mmCIF identity for one residue in an analysis PDB."""

    residue_name: str
    chain_id: str
    residue_number: int | None = None
    insertion_code: str = ""
    polymer_position: str = ""
    chain_index: int | None = None
    residue_index: int | None = None


@dataclass(frozen=True, slots=True)
class ConversionProvenance:
    """Everything the provenance records of one converted PDB say."""

    residue_mapping: dict[ResidueMappingKey, SourceResidueIdentity]
    #: Keyed by zero-based model index; models without a record are absent.
    defaulted_occupancy_counts: dict[int, int]


def _checked_polymer_position(token: str, context: str) -> str:
    """Return ``token`` if it is a known polymer position code."""
    if token not in POLYMER_POSITIONS:
        raise ValueError(f"invalid polymer position {token!r} in {context}")
    return token


def conversion_provenance_remarks(
    resname_records: Sequence[ResnameRecord],
    identity_records: Sequence[ResidueIdentityRecord] = (),
    polymer_records: Sequence[PolymerPositionRecord] = (),
    defaulted_occupancy_counts: Sequence[int] = (),
) -> list[str]:
    """Render the provenance lines of one conversion, in their on-disk order.

    ``defaulted_occupancy_counts`` is indexed by zero-based model; models whose
    count is zero produce no record. Values the reader would reject are
    rejected here, so a bad conversion fails at conversion time.
    """
    remarks = [RESNAME_LAYOUT.format(*record) for record in resname_records]
    for model_number, count in enumerate(defaulted_occupancy_counts, start=1):
        if count < 0:
            raise ValueError(
                f"negative defaulted occupancy count {count} for model {model_number}"
            )
        if count:
            remarks.append(OCCUPANCY_DEFAULT_LAYOUT.format(model_number, count))
    for identity_record in identity_records:
        _checked_polymer_position(
            identity_record[-1], f"residue identity record {identity_record!r}"
        )
        remarks.append(RESIDUE_LAYOUT.format(*identity_record))
    for polymer_record in polymer_records:
        _checked_polymer_position(
            polymer_record[-1], f"polymer position record {polymer_record!r}"
        )
        remarks.append(POLYMER_LAYOUT.format(*polymer_record))
    return remarks


def _provenance_records(path: str) -> Iterator[tuple[RemarkLayout, list[str], str]]:
    """Yield ``(layout, tokens, line)`` for every provenance record in a PDB."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(REMARK_PREFIX):
                continue
            tokens = line.split()
            layout = next(
                (layout for layout in LAYOUTS if layout.matches(tokens)), None
            )
            if layout is None:
                continue
            if len(tokens) != layout.token_count:
                raise ValueError(
                    f"malformed Alchemy provenance record: {line.rstrip()}"
                )
            yield layout, tokens, line


def _int_field(
    layout: RemarkLayout, tokens: Sequence[str], name: str, line: str
) -> int:
    """Read an integer field, accepting only plain decimal digits."""
    token = layout.field(tokens, name)
    if not _INTEGER_TOKEN.fullmatch(token):
        raise ValueError(
            f"invalid {name.replace('_', ' ')} in Alchemy provenance record: "
            f"{line.rstrip()}"
        )
    return int(token)


def _model_index_field(layout: RemarkLayout, tokens: Sequence[str], line: str) -> int:
    """Convert the one-based model field to the zero-based mapping index."""
    model_number = _int_field(layout, tokens, "model", line)
    if model_number < 1:
        raise ValueError(f"Alchemy provenance model must be positive: {line.rstrip()}")
    return model_number - 1


def _residue_mapping_key(
    layout: RemarkLayout, tokens: Sequence[str], line: str
) -> ResidueMappingKey:
    """Read the converted-PDB residue identity shared by every mapping record."""
    return (
        _model_index_field(layout, tokens, line),
        layout.field(tokens, "written_name"),
        layout.field(tokens, "chain"),
        layout.field(tokens, "resnum"),
    )


def _residue_record_identity(tokens: Sequence[str], line: str) -> SourceResidueIdentity:
    """Build the full source identity a ``RESIDUE`` record carries."""
    return SourceResidueIdentity(
        residue_name=RESIDUE_LAYOUT.field(tokens, "source_name"),
        chain_id=RESIDUE_LAYOUT.field(tokens, "source_chain"),
        residue_number=_int_field(RESIDUE_LAYOUT, tokens, "source_number", line),
        insertion_code=RESIDUE_LAYOUT.field(tokens, "source_insertion"),
        chain_index=_int_field(RESIDUE_LAYOUT, tokens, "source_chain_index", line),
        residue_index=_int_field(RESIDUE_LAYOUT, tokens, "source_residue_index", line),
        polymer_position=_checked_polymer_position(
            RESIDUE_LAYOUT.field(tokens, "source_polymer_position"), line.rstrip()
        ),
    )


def _conflict_message(kind: str, key: ResidueMappingKey) -> str:
    """Describe two records that disagree about the same converted residue."""
    model_index, written_name, chain, resnum = key
    return (
        f"conflicting Alchemy {kind} mappings for model {model_index + 1} "
        f"residue {written_name}/{chain or _EMPTY_FIELD_PLACEHOLDER}/{resnum}"
    )


def _merged_identity(
    previous: SourceResidueIdentity,
    incoming: SourceResidueIdentity,
    key: ResidueMappingKey,
) -> SourceResidueIdentity:
    """Combine two identity records for one converted residue, in either order.

    A ``RESNAME`` record carries only the source name; a ``RESIDUE`` record
    for the same residue carries the full identity and supersedes it. Two
    records of the same kind must agree.
    """
    if previous == incoming:
        return previous
    full = [
        identity
        for identity in (previous, incoming)
        if identity.residue_number is not None
    ]
    if len(full) == 1 and previous.residue_name == incoming.residue_name:
        return full[0]
    raise ValueError(_conflict_message("residue", key))


def read_conversion_provenance(path: str) -> ConversionProvenance:
    """Read the provenance records embedded in a converted PDB in one pass.

    ``RESNAME`` records carry the source component name alone; ``RESIDUE``
    records, written when a residue's chain, number, or name changed in
    conversion, also carry the source chain, sequence number and insertion
    code. ``POLYMER`` records independently preserve whether the deposited
    sequence identifies a residue as terminal, internal, non-polymer, or
    indeterminate. ``OCCUPANCY DEFAULTED`` records count, per model, the atoms
    whose occupancy came from the mmCIF dictionary default. Records that
    disagree about one residue or model are rejected.
    """
    mapping: dict[ResidueMappingKey, SourceResidueIdentity] = {}
    polymer_positions: dict[ResidueMappingKey, str] = {}
    counts: dict[int, int] = {}
    for layout, tokens, line in _provenance_records(path):
        if layout is OCCUPANCY_DEFAULT_LAYOUT:
            model_index = _model_index_field(layout, tokens, line)
            count = _int_field(layout, tokens, "count", line)
            if count <= 0:
                raise ValueError(
                    f"Alchemy occupancy provenance count must be positive: "
                    f"{line.rstrip()}"
                )
            if model_index in counts:
                raise ValueError(
                    "duplicate Alchemy occupancy provenance for model "
                    f"{model_index + 1}"
                )
            counts[model_index] = count
            continue
        key = _residue_mapping_key(layout, tokens, line)
        if layout is POLYMER_LAYOUT:
            position = _checked_polymer_position(
                layout.field(tokens, "polymer_position"), line.rstrip()
            )
            if polymer_positions.get(key, position) != position:
                raise ValueError(_conflict_message("polymer", key))
            polymer_positions[key] = position
            continue
        if layout is RESNAME_LAYOUT:
            _model_index, _written_name, chain, _resnum = key
            source = SourceResidueIdentity(
                residue_name=layout.field(tokens, "source_name"), chain_id=chain
            )
        else:
            source = _residue_record_identity(tokens, line)
        previous = mapping.get(key)
        mapping[key] = (
            source if previous is None else _merged_identity(previous, source, key)
        )
    for key, position in polymer_positions.items():
        current = mapping.get(key)
        if current is None:
            _model_index, written_name, chain, _resnum = key
            mapping[key] = SourceResidueIdentity(
                residue_name=written_name, chain_id=chain, polymer_position=position
            )
        elif current.polymer_position not in ("", position):
            raise ValueError(_conflict_message("polymer", key))
        else:
            mapping[key] = replace(current, polymer_position=position)
    return ConversionProvenance(mapping, counts)
