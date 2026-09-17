"""Entry-level diagnostics ``coordination.analysis`` records for one entry.

The manifest status detail is capped, so entry-wide limitations must be
recorded before per-site enumerations and the enumerations themselves must
stay short. Facts that hold for the whole entry are reported once, however
many metal sites contributed them.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import helpers
import pytest
from helpers import StructureBuilder

from coordination import site_environment

NON_FINITE_PREFIX = "geometry unavailable for selected metal site(s)"
UNSUPPORTED_PREFIX = "first-sphere reference unavailable for"
SPECIAL_POSITION_PREFIX = "special position evaluation failed"
ENTRY_PREFIXES = ("DPI unavailable:", "symmetry search unavailable")


def _messages_starting_with(messages: list[str], prefix: str) -> list[str]:
    """Every recorded message that begins with ``prefix``."""
    return [message for message in messages if message.startswith(prefix)]


def _only_message(messages: list[str], prefix: str) -> str:
    """The single recorded message beginning with ``prefix``."""
    matches = _messages_starting_with(messages, prefix)
    assert len(matches) == 1, matches
    return matches[0]


def _site(
    builder: StructureBuilder,
    *,
    metal: str,
    seqid: int,
    resname: str,
    atom_name: str,
    origin_x: float,
    distance: float = 2.5,
) -> None:
    """Add a metal at ``(origin_x, 0, 0)`` with one donor atom beside it.

    The rest of the donor residue sits far away, so only ``atom_name`` is
    within the discovery radius of any metal.
    """
    builder.add_metal(metal, seqid, chain="B", pos=(origin_x, 0.0, 0.0))
    builder.add_amino_acid(
        resname,
        10 + seqid,
        chain="A",
        positions={atom_name: (origin_x + distance, 0.0, 0.0)},
        origin=(origin_x, 40.0, 40.0),
    )


def test_unsupported_pairs_of_every_metal_are_reported_once(tmp_path: Path) -> None:
    """The reference gap is an entry-level fact, not one message per metal.

    Neither K-S nor Na-N is in the literature table, and both sites report
    through a single code and a single message naming both pairs.
    """
    builder = StructureBuilder()
    _site(builder, metal="K", seqid=1, resname="CYS", atom_name="SG", origin_x=0.0)
    _site(builder, metal="NA", seqid=2, resname="HIS", atom_name="NE2", origin_x=30.0)
    analysis = helpers.analyze_bonds(builder.write_cif(tmp_path / "pairs.cif"))
    metadata = analysis.metadata

    assert metadata.partial_reason_codes.count("missing_first_sphere_reference") == 1
    assert _only_message(metadata.messages, UNSUPPORTED_PREFIX) == (
        f"{UNSUPPORTED_PREFIX} K-S, NA-N"
    )


def test_a_pair_shared_by_several_metals_is_named_once(tmp_path: Path) -> None:
    """Three sites hitting the same gap leave one code and one undoubled pair."""
    builder = StructureBuilder()
    for index in range(3):
        _site(
            builder,
            metal="K",
            seqid=index + 1,
            resname="CYS",
            atom_name="SG",
            origin_x=30.0 * index,
        )
    analysis = helpers.analyze_bonds(builder.write_cif(tmp_path / "shared.cif"))
    metadata = analysis.metadata

    assert metadata.partial_reason_codes.count("missing_first_sphere_reference") == 1
    assert _only_message(metadata.messages, UNSUPPORTED_PREFIX) == (
        f"{UNSUPPORTED_PREFIX} K-S"
    )
    assert len(set(metadata.messages)) == len(metadata.messages)


def test_entry_limitations_precede_capped_per_site_enumerations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DPI and symmetry come first, and six broken sites list five plus a count.

    ``worker.lifecycle`` truncates the joined messages for the manifest, so an
    unbounded per-site enumeration must not push the entry-level reasons out.
    """
    builder = StructureBuilder(cell=None)
    for seqid in range(1, 7):
        builder.add_metal("ZN", seqid, chain="B", pos=(10.0 * seqid, 0.0, 0.0))
    path = builder.write_cif(tmp_path / "broken.cif")
    parsed = gemmi.read_structure(path)
    for chain in parsed[0]:
        for residue in chain:
            for atom in residue:
                if str(atom.element.name).upper() == "ZN":
                    atom.pos = gemmi.Position(float("nan"), 0.0, 0.0)

    def read_parsed_structure(_path: str, **_options: object) -> gemmi.Structure:
        return parsed

    monkeypatch.setattr(gemmi, "read_structure", read_parsed_structure)
    metadata = helpers.analyze_bonds(path).metadata

    assert "symmetry_search_unavailable" in metadata.partial_reason_codes
    assert "non_finite_metal_coordinates" in metadata.partial_reason_codes
    dpi_index = metadata.messages.index("DPI unavailable: missing_dpi_metadata_source")
    symmetry_index = next(
        index
        for index, message in enumerate(metadata.messages)
        if message.startswith("symmetry search unavailable")
    )
    non_finite = _only_message(metadata.messages, NON_FINITE_PREFIX)
    assert dpi_index < metadata.messages.index(non_finite)
    assert symmetry_index < metadata.messages.index(non_finite)
    assert non_finite.count("ZN/B/") == 5
    assert non_finite.endswith("(and 1 more)")


def test_a_failed_special_position_evaluation_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising symmetry evaluation becomes a message and a partial reason code.

    Without this wiring an entry whose site symmetry could not be evaluated is
    indistinguishable from one that simply has no special position: every
    ``metal_special_position`` column is blank either way.
    """
    builder = StructureBuilder()
    _site(builder, metal="ZN", seqid=1, resname="HIS", atom_name="NE2", origin_x=0.0)
    path = builder.write_cif(tmp_path / "special.cif")

    def exploding_spacegroup(_structure: gemmi.Structure) -> gemmi.SpaceGroup | None:
        raise RuntimeError("cell images unavailable")

    monkeypatch.setattr(site_environment, "spacegroup_or_none", exploding_spacegroup)
    metadata = helpers.analyze_bonds(path).metadata

    assert metadata.partial_reason_codes.count("symmetry_search_unavailable") == 1
    assert _only_message(metadata.messages, SPECIAL_POSITION_PREFIX) == (
        f"{SPECIAL_POSITION_PREFIX}: RuntimeError: cell images unavailable"
    )
    failure_index = metadata.messages.index(
        _only_message(metadata.messages, SPECIAL_POSITION_PREFIX)
    )
    entry_indexes = [
        index
        for index, message in enumerate(metadata.messages)
        if message.startswith(ENTRY_PREFIXES)
    ]
    assert entry_indexes
    assert max(entry_indexes) < failure_index
