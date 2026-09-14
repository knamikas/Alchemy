"""Write and read the ``REMARK 950 ALCHEMY`` provenance records of converted PDBs.

Converting mmCIF to legacy PDB loses information the analysis must restore:
component names longer than three characters, author identifiers packed into
the PDB namespace, each residue's polymer-boundary status, and occupancies the
mmCIF dictionary defaulted. ``coordinate_conversion`` prepends these records to
the converted file and ``structure_analysis`` reads them back; both use the
record layouts declared here, so the format is described exactly once.

Every record is a whitespace-separated line: the prefix names the record type
and the remaining tokens follow that type's :class:`RemarkLayout`. ``_`` stands
in for an empty chain id or insertion code, which would otherwise vanish when
the line is split. Model numbers are one-based on disk and zero-based in the
parsed mappings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

RESNAME_REMARK_PREFIX = "REMARK 950 ALCHEMY RESNAME"
RESIDUE_REMARK_PREFIX = "REMARK 950 ALCHEMY RESIDUE"
POLYMER_REMARK_PREFIX = "REMARK 950 ALCHEMY POLYMER"
OCCUPANCY_DEFAULT_REMARK_PREFIX = "REMARK 950 ALCHEMY OCCUPANCY DEFAULTED"

# ``-`` non-polymer, ``?`` indeterminate, ``M`` internal, ``N``/``C``/``NC`` terminal.
POLYMER_POSITIONS = ("-", "?", "M", "N", "C", "NC")
_EMPTY_FIELD_PLACEHOLDER = "_"


@dataclass(frozen=True, slots=True)
class RemarkLayout:
    """The fields of one record type, in the order they follow its prefix."""

    prefix: str
    fields: tuple[str, ...]

    @property
    def prefix_tokens(self) -> tuple[str, ...]:
        """The prefix split the way a record line is split."""
        return tuple(self.prefix.split())

    @property
    def token_count(self) -> int:
        """Number of whitespace-separated tokens in a well-formed record."""
        return len(self.prefix_tokens) + len(self.fields)

    def matches(self, tokens: Sequence[str]) -> bool:
        """Whether a split line starts with this layout's prefix."""
        return tuple(tokens[: len(self.prefix_tokens)]) == self.prefix_tokens

    def field(self, tokens: Sequence[str], name: str) -> str:
        """Return the token holding ``name`` in a split, length-checked record."""
        return tokens[len(self.prefix_tokens) + self.fields.index(name)]

    def format(self, *values: object) -> str:
        """Render one record line from ``values`` given in layout order."""
        if len(values) != len(self.fields):
            raise ValueError(
                f"{self.prefix!r} takes {len(self.fields)} fields, got {len(values)}"
            )
        return " ".join((self.prefix, *(str(value) for value in values))) + "\n"


# Record layouts. The token count in the trailing comment is the whole line,
# prefix included; the first four fields are shared by the residue-keyed types
# and identify the residue as it appears in the converted PDB.
RESNAME_LAYOUT = RemarkLayout(
    RESNAME_REMARK_PREFIX,
    ("model", "chain", "resnum", "written_name", "source_name"),
)  # 9 tokens
RESIDUE_LAYOUT = RemarkLayout(
    RESIDUE_REMARK_PREFIX,
    (
        "model",
        "chain",
        "resnum",
        "written_name",
        "source_chain",
        "source_number",
        "source_insertion",
        "source_name",
        "source_chain_index",
        "source_residue_index",
        "source_polymer_position",
    ),
)  # 15 tokens
POLYMER_LAYOUT = RemarkLayout(
    POLYMER_REMARK_PREFIX,
    ("model", "chain", "resnum", "written_name", "polymer_position"),
)  # 9 tokens
OCCUPANCY_DEFAULT_LAYOUT = RemarkLayout(
    OCCUPANCY_DEFAULT_REMARK_PREFIX, ("model", "count")
)  # 7 tokens

_RESIDUE_MAPPING_LAYOUTS = (RESNAME_LAYOUT, RESIDUE_LAYOUT, POLYMER_LAYOUT)

# Writer inputs, one tuple per record in layout order after the model index.
# (model, chain, resnum, written name, source name)
ResnameRecord = tuple[int, str, str, str, str]
# (model, chain, resnum, written name, source chain, source number, source
#  insertion, source name, source chain index, source residue index, source
#  polymer position)
ResidueIdentityRecord = tuple[int, str, str, str, str, int, str, str, int, int, str]
# (model, chain, resnum, written name, polymer position)
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


def _placeholder_if_empty(value: str) -> str:
    """Write an empty chain id or insertion code so it survives splitting."""
    return value or _EMPTY_FIELD_PLACEHOLDER


def _empty_if_placeholder(token: str) -> str:
    """Invert :func:`_placeholder_if_empty` when reading a record."""
    return "" if token == _EMPTY_FIELD_PLACEHOLDER else token


def conversion_provenance_remarks(
    resname_records: Sequence[ResnameRecord],
    identity_records: Sequence[ResidueIdentityRecord] = (),
    polymer_records: Sequence[PolymerPositionRecord] = (),
    defaulted_occupancy_counts: Sequence[int] = (),
) -> list[str]:
    """Render the provenance lines of one conversion, in their on-disk order.

    ``defaulted_occupancy_counts`` is indexed by zero-based model; models whose
    count is zero produce no record.
    """
    remarks = [
        RESNAME_LAYOUT.format(
            model_index,
            _placeholder_if_empty(chain),
            resnum,
            written_name,
            source_name,
        )
        for model_index, chain, resnum, written_name, source_name in resname_records
    ]
    remarks.extend(
        OCCUPANCY_DEFAULT_LAYOUT.format(model_index, count)
        for model_index, count in enumerate(defaulted_occupancy_counts, start=1)
        if count
    )
    remarks.extend(
        RESIDUE_LAYOUT.format(
            model_index,
            _placeholder_if_empty(chain),
            resnum,
            written_name,
            _placeholder_if_empty(source_chain),
            source_number,
            _placeholder_if_empty(source_insertion),
            source_name,
            source_chain_index,
            source_residue_index,
            source_polymer_position,
        )
        for (
            model_index,
            chain,
            resnum,
            written_name,
            source_chain,
            source_number,
            source_insertion,
            source_name,
            source_chain_index,
            source_residue_index,
            source_polymer_position,
        ) in identity_records
    )
    remarks.extend(
        POLYMER_LAYOUT.format(
            model_index,
            _placeholder_if_empty(chain),
            resnum,
            written_name,
            polymer_position,
        )
        for model_index, chain, resnum, written_name, polymer_position in (
            polymer_records
        )
    )
    return remarks


def write_conversion_provenance(
    dst: str,
    resname_records: Sequence[ResnameRecord],
    identity_records: Sequence[ResidueIdentityRecord] = (),
    polymer_records: Sequence[PolymerPositionRecord] = (),
    defaulted_occupancy_counts: Sequence[int] = (),
) -> None:
    """Prepend the provenance remarks to the converted PDB file at ``dst``."""
    remarks = conversion_provenance_remarks(
        resname_records, identity_records, polymer_records, defaulted_occupancy_counts
    )
    with open(dst, encoding="utf-8", errors="strict", newline="") as handle:
        body = handle.read()
    with open(dst, "w", encoding="utf-8", newline="") as handle:
        handle.writelines(remarks)
        handle.write(body)


def _residue_mapping_layout(tokens: Sequence[str]) -> RemarkLayout | None:
    """Return the residue-keyed layout a split line belongs to, if any."""
    for layout in _RESIDUE_MAPPING_LAYOUTS:
        if layout.matches(tokens):
            return layout
    return None


def _parse_model_index(token: str) -> int:
    """Convert the one-based model field to the zero-based mapping index."""
    try:
        model_index = int(token) - 1
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid model in residue mapping: {token!r}") from exc
    if model_index < 0:
        raise ValueError("residue mapping model must be positive")
    return model_index


def _residue_mapping_key(
    tokens: Sequence[str], layout: RemarkLayout
) -> ResidueMappingKey:
    """Read the converted-PDB residue identity shared by every mapping record."""
    return (
        _parse_model_index(layout.field(tokens, "model")),
        layout.field(tokens, "written_name"),
        _empty_if_placeholder(layout.field(tokens, "chain")),
        layout.field(tokens, "resnum"),
    )


def _checked_polymer_position(token: str) -> str:
    """Return ``token`` if it is a known polymer position code."""
    if token not in POLYMER_POSITIONS:
        raise ValueError(
            f"invalid source polymer position in residue mapping: {token!r}"
        )
    return token


def _source_identity_from_residue_record(
    tokens: Sequence[str], line: str
) -> SourceResidueIdentity:
    """Build the full source identity a ``RESIDUE`` record carries."""
    try:
        source_number = int(RESIDUE_LAYOUT.field(tokens, "source_number"))
        source_chain_index = int(RESIDUE_LAYOUT.field(tokens, "source_chain_index"))
        source_residue_index = int(RESIDUE_LAYOUT.field(tokens, "source_residue_index"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"invalid source numeric field in residue mapping: {line.rstrip()}"
        ) from exc
    return SourceResidueIdentity(
        residue_name=RESIDUE_LAYOUT.field(tokens, "source_name"),
        chain_id=_empty_if_placeholder(RESIDUE_LAYOUT.field(tokens, "source_chain")),
        residue_number=source_number,
        insertion_code=_empty_if_placeholder(
            RESIDUE_LAYOUT.field(tokens, "source_insertion")
        ),
        chain_index=source_chain_index,
        residue_index=source_residue_index,
        polymer_position=_checked_polymer_position(
            RESIDUE_LAYOUT.field(tokens, "source_polymer_position")
        ),
    )


def _conflict_message(kind: str, key: ResidueMappingKey) -> str:
    """Describe two records that disagree about the same converted residue."""
    _model_index, written_name, chain, resnum = key
    return f"conflicting Alchemy {kind} mappings for {written_name}/{chain}/{resnum}"


def read_residue_mapping(
    path: str,
) -> dict[ResidueMappingKey, SourceResidueIdentity]:
    """Read reversible mmCIF residue mappings embedded during conversion.

    ``RESNAME`` records carry the source component name alone; ``RESIDUE``
    records, written when residues had to be packed into a PDB-safe namespace,
    also carry the source chain, sequence number and insertion code. ``POLYMER``
    records independently preserve whether the deposited sequence identifies a
    residue as terminal, internal, non-polymer, or indeterminate.
    """
    mapping: dict[ResidueMappingKey, SourceResidueIdentity] = {}
    polymer_positions: dict[ResidueMappingKey, str] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            layout = _residue_mapping_layout(tokens)
            if layout is None:
                continue
            if len(tokens) != layout.token_count:
                raise ValueError(f"malformed Alchemy residue mapping: {line.rstrip()}")
            key = _residue_mapping_key(tokens, layout)
            _model_index, written_name, chain, _resnum = key
            if layout is POLYMER_LAYOUT:
                polymer_position = _checked_polymer_position(
                    layout.field(tokens, "polymer_position")
                )
                previous_position = polymer_positions.get(key)
                if (
                    previous_position is not None
                    and previous_position != polymer_position
                ):
                    raise ValueError(_conflict_message("polymer", key))
                polymer_positions[key] = polymer_position
                continue
            if layout is RESNAME_LAYOUT:
                source = SourceResidueIdentity(
                    residue_name=layout.field(tokens, "source_name"),
                    chain_id=chain,
                )
            else:
                source = _source_identity_from_residue_record(tokens, line)
            previous = mapping.get(key)
            if previous is not None:
                compatible_legacy_upgrade = (
                    layout is RESIDUE_LAYOUT
                    and previous.residue_number is None
                    and previous.residue_name == source.residue_name
                )
                if previous != source and not compatible_legacy_upgrade:
                    raise ValueError(_conflict_message("residue", key))
            mapping[key] = source
    for key, polymer_position in polymer_positions.items():
        current = mapping.get(key)
        if current is None:
            _model_index, written_name, chain, _resnum = key
            mapping[key] = SourceResidueIdentity(
                residue_name=written_name,
                chain_id=chain,
                polymer_position=polymer_position,
            )
        else:
            mapping[key] = replace(current, polymer_position=polymer_position)
    return mapping


def read_defaulted_occupancy_counts(path: str) -> dict[int, int]:
    """Read per-model counts of atoms whose occupancy came from the mmCIF default.

    The result is keyed by zero-based model index and omits models without a
    record.
    """
    counts: dict[int, int] = {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = line.split()
            if not OCCUPANCY_DEFAULT_LAYOUT.matches(tokens):
                continue
            if len(tokens) != OCCUPANCY_DEFAULT_LAYOUT.token_count:
                raise ValueError(
                    f"malformed Alchemy occupancy provenance: {line.rstrip()}"
                )
            try:
                model_index = int(OCCUPANCY_DEFAULT_LAYOUT.field(tokens, "model")) - 1
                count = int(OCCUPANCY_DEFAULT_LAYOUT.field(tokens, "count"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"invalid Alchemy occupancy provenance: {line.rstrip()}"
                ) from exc
            if model_index < 0 or count <= 0:
                raise ValueError(
                    "Alchemy occupancy provenance requires a positive model and count"
                )
            if model_index in counts:
                raise ValueError(
                    "duplicate Alchemy occupancy provenance for model "
                    f"{model_index + 1}"
                )
            counts[model_index] = count
    return counts
