"""Review fixes for ``coordination.declared_connections``.

Covers which warning codes a discarded declaration may emit, the conformer a
declaration that names none is answered with, and the failure paths around
reading and replaying the source connection file.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import gemmi
import helpers
import pytest
from helpers import AtomSpec, ResidueSpec, StructureBuilder, approx

import coordinate_conversion as conversion
from codes import CandidateSource
from coordination import declared_connections
from coordination.contact_record import Candidate
from metal_elements import METAL_ELEMENTS
from structure_analysis import StructureContext, load_structure


def _write_source_and_analysis(
    builder: StructureBuilder, directory: Path, fmt: str
) -> tuple[str, str]:
    """Write ``builder`` as the source file and return ``(source, analysis)``."""
    if fmt == "cif":
        source = builder.write_cif(os.path.join(str(directory), "source.cif"))
        return source, conversion.cif_to_pdb(
            source, os.path.join(str(directory), "analysis.pdb")
        )
    source = builder.write_pdb(os.path.join(str(directory), "source.pdb"))
    return source, source


def _declared_codes(warning_codes: Sequence[str]) -> list[str]:
    """The entry warning codes this module's subject is responsible for."""
    return [code for code in warning_codes if code.startswith("declared_")]


def _zinc_histidine_site(
    donor_positions: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[StructureBuilder, ResidueSpec, ResidueSpec]:
    """A one-histidine zinc site: returns ``(builder, his, zinc)``."""
    builder = StructureBuilder()
    his = builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions=donor_positions or {"NE2": (2.03, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    return builder, his, zinc


def _cra(
    chain: str,
    resname: str,
    seqid: int,
    atom_name: str,
    element: str,
    altloc: str = "",
    icode: str = " ",
) -> gemmi.CRA:
    """A stand-in for the ``CRA`` gemmi returns for a declared partner."""
    return cast(
        gemmi.CRA,
        SimpleNamespace(
            chain=SimpleNamespace(name=chain),
            residue=SimpleNamespace(
                name=resname, seqid=SimpleNamespace(num=seqid, icode=icode)
            ),
            atom=SimpleNamespace(
                name=atom_name, altloc=altloc, element=SimpleNamespace(name=element)
            ),
        ),
    )


class _PreparedModel:
    """A source model answering the declaration's partners with fixed CRAs."""

    def __init__(self, cras: Sequence[gemmi.CRA]) -> None:
        self._cras = list(cras)

    def find_cra(
        self, address: gemmi.AtomAddress, /, ignore_segment: bool = False
    ) -> gemmi.CRA:
        return self._cras.pop(0)


def _resolve_in_memory(
    context: StructureContext,
    builder: StructureBuilder,
    index: int = 0,
) -> tuple[gemmi.Connection, Candidate | None, list[str], list[str]]:
    """Resolve declaration ``index`` of ``builder`` against ``context``.

    The source model is the in-memory structure, which keeps the ``asu`` and
    reported distance of a declaration that a written file would normalize.
    """
    source_structure = builder.to_gemmi()
    connection = source_structure.connections[index]
    metals = context.metal_atoms(METAL_ELEMENTS, canonical=True)
    resolved = declared_connections.resolve_declared_partners(
        context, source_structure[0], connection, {}
    )
    declared, issues, warnings = declared_connections.declared_candidate_for_connection(
        context,
        connection,
        str(connection.name),
        CandidateSource.LINK,
        resolved,
        {metal.source_key for metal in metals},
    )
    candidate = None if declared is None else declared.candidate
    return connection, candidate, issues, warnings


def test_discarded_intra_cofactor_declaration_warns_about_nothing(
    tmp_path: Path,
) -> None:
    """A cluster's own ``struct_conn`` records leave no entry-level warning.

    Fe-S and heme depositions wire up the cofactor's internal chemistry, which
    is discarded before it can become a candidate, so a warning code naming a
    donor class or an occupancy would point at no published row.
    """
    builder = StructureBuilder()
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (2.30, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    cluster = builder.add_hetero_residue(
        "FES",
        1,
        [
            AtomSpec("FE1", "FE", (0.0, 0.0, 0.0)),
            AtomSpec("FE2", "FE", (2.70, 0.0, 0.0)),
            AtomSpec("S1", "S", (1.35, 1.15, 0.0)),
            AtomSpec("S2", "S", (1.35, -1.15, 0.0)),
        ],
        chain="B",
    )
    builder.add_connection(
        cluster.ref("FE1"), cluster.ref("S1"), name="internal", reported_distance=1.77
    )
    source, analysis_pdb = _write_source_and_analysis(builder, tmp_path, "pdb")

    result = helpers.analyze_bonds(analysis_pdb, connection_path=source)

    assert result.declared_rows == []
    assert _declared_codes(result.metadata.warning_codes) == []
    assert [
        message for message in result.metadata.messages if "internal" in message
    ] == []


def test_discarded_non_donor_declaration_reports_only_the_element_code(
    tmp_path: Path,
) -> None:
    """A declared metal-carbon contact reports the element, nothing else.

    The element code describes the deposited declaration, which no candidate
    can carry; the occupancy of a partner that never became a candidate is not
    the entry's business.
    """
    builder, his, zinc = _zinc_histidine_site(
        {"NE2": (2.03, 0.0, 0.0), "CE1": (0.0, 2.90, 0.0)}
    )
    his.atoms = [replace(atom, occupancy=0.0) for atom in his.atoms]
    builder.add_connection(
        zinc.ref("ZN"), his.ref("CE1"), name="c1", reported_distance=2.90
    )
    source, analysis_pdb = _write_source_and_analysis(builder, tmp_path, "pdb")

    result = helpers.analyze_bonds(analysis_pdb, connection_path=source)

    assert _declared_codes(result.metadata.warning_codes) == [
        "declared_donor_element_unsupported"
    ]
    assert [row for row in result.candidate_rows if row["neighbor_atom"] == "CE1"] == []


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_blank_altloc_declaration_answers_with_the_selected_conformer(
    tmp_path: Path, order: tuple[str, str]
) -> None:
    """Naming no conformer resolves to the selected one in either deposition order.

    Answering with the deposition-order-first alternate would report a
    substitution for a declaration that never named an alternate at all.
    """
    occupancies = {"A": 0.35, "B": 0.65}
    positions = {"A": (2.03, 0.0, 0.0), "B": (2.20, 0.0, 0.0)}
    builder, his, _ = _zinc_histidine_site()
    builder.add_conformers(
        his,
        [(altloc, occupancies[altloc], {"NE2": positions[altloc]}) for altloc in order],
        atom_names=["NE2"],
    )
    context = load_structure("test", builder.write_pdb(tmp_path / "conformers.pdb"))
    residue = context.residues_for_author("HIS", "A", "10")[0]
    assert residue.selected_altloc == "B"

    resolved = declared_connections.analysis_atom_for_partner(
        context, _cra("A", "HIS", 10, "NE2", "N"), {}
    )

    assert resolved is not None
    assert resolved.altloc == "B"
    assert resolved.xyz == approx((2.20, 0.0, 0.0))


@pytest.mark.parametrize("order", [("A", "B"), ("B", "A")])
def test_blank_altloc_declaration_is_not_a_conformer_substitution(
    tmp_path: Path, order: tuple[str, str]
) -> None:
    """A declaration naming no conformer never reports a substitution."""
    occupancies = {"A": 0.35, "B": 0.65}
    positions = {"A": (2.03, 0.0, 0.0), "B": (2.20, 0.0, 0.0)}
    builder, his, zinc = _zinc_histidine_site()
    builder.add_conformers(
        his,
        [(altloc, occupancies[altloc], {"NE2": positions[altloc]}) for altloc in order],
        atom_names=["NE2"],
    )
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.20
    )
    context = load_structure("test", builder.write_pdb(tmp_path / "conformers.pdb"))
    connection = builder.to_gemmi().connections[0]
    # A deposited declaration that leaves the alternate id blank; gemmi fills
    # the id in from the model when it writes the file itself.
    model = _PreparedModel(
        [_cra("B", "ZN", 1, "ZN", "Zn"), _cra("A", "HIS", 10, "NE2", "N")]
    )

    resolved = declared_connections.resolve_declared_partners(
        context, model, connection, {}
    )

    assert resolved.conformer_substituted is False
    assert resolved.conformer_deselected is False
    metals = context.metal_atoms(METAL_ELEMENTS, canonical=True)
    declared, issues, warnings = declared_connections.declared_candidate_for_connection(
        context,
        connection,
        "m1",
        CandidateSource.LINK,
        resolved,
        {metal.source_key for metal in metals},
    )
    assert declared is not None
    assert declared.candidate.neighbor is not None
    assert declared.candidate.neighbor.altloc == "B"
    assert (issues, warnings) == ([], [])


def test_a_failed_chain_name_replay_leaves_the_entry_analyzable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaying the conversion is guarded, so its failure is only an issue.

    Escaping here would fail the whole bond stage of an entry whose
    coordinates are perfectly analyzable.
    """
    builder, his, zinc = _zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    source, analysis_pdb = _write_source_and_analysis(builder, tmp_path, "cif")

    def _raise(path: str, structure: gemmi.Structure | None = None) -> dict[str, str]:
        raise RuntimeError("replay exploded")

    monkeypatch.setattr(declared_connections, "analysis_chain_names", _raise)

    result = helpers.analyze_bonds(analysis_pdb, connection_path=source)

    assert result.declared_rows == []
    assert len(result.rows_for("NE2")) == 1
    assert "declared_connection_resolution_incomplete" in (
        result.metadata.partial_reason_codes
    )
    assert "struct_conn parse failed: RuntimeError: replay exploded" in (
        result.metadata.messages
    )


def test_a_connection_file_without_a_model_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    """A source file holding no coordinate model yields one issue."""
    builder, _, _ = _zinc_histidine_site()
    context = load_structure("test", builder.write_pdb(tmp_path / "analysis.pdb"))
    empty = tmp_path / "empty.cif"
    empty.write_text("data_empty\n_entry.id EMPTY\n", encoding="utf-8")

    candidates, issues, warnings = declared_connections.collect_declared_candidates(
        context, str(empty), context.metal_atoms(METAL_ELEMENTS, canonical=True)
    )

    assert (candidates, warnings) == ([], [])
    assert issues == ["struct_conn contains no coordinate model"]


def _unit_cell_free_site(
    tmp_path: Path,
) -> tuple[StructureBuilder, ResidueSpec, ResidueSpec, StructureContext]:
    """A zinc and a non-standard oxygen donor in a file with no unit cell."""
    builder = StructureBuilder(cell=None, spacegroup=None)
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    ligand = builder.add_hetero_residue(
        "LIG", 2, [AtomSpec("O1", "O", (2.10, 0.0, 0.0))], chain="B"
    )
    context = load_structure("test", builder.write_pdb(tmp_path / "no_cell.pdb"))
    assert context.symmetry.search_available is False
    return builder, zinc, ligand, context


@pytest.mark.parametrize(
    "asu, expected_scope",
    [(gemmi.Asu.Same, "explicit"), (gemmi.Asu.Any, "explicit")],
)
def test_geometry_falls_back_to_the_explicit_image_without_symmetry(
    tmp_path: Path, asu: gemmi.Asu, expected_scope: str
) -> None:
    """``Same`` and an unsatisfiable ``Any`` are both measured as deposited."""
    *_, context = _unit_cell_free_site(tmp_path)
    metal = next(atom for atom in context.contact_atoms if atom.element == "ZN")
    neighbor = next(atom for atom in context.contact_atoms if atom.atom_name == "O1")
    connection = gemmi.Connection()
    connection.asu = asu

    image = declared_connections.declared_candidate_geometry(
        context, metal, neighbor, connection
    )

    assert image.scope == expected_scope
    assert image.symmetry_contact is False
    assert image.distance == approx(2.10, abs=1e-6)


def test_geometry_requiring_symmetry_raises_the_recorded_reason(
    tmp_path: Path,
) -> None:
    """``Different`` cannot be measured at all without symmetry metadata."""
    *_, context = _unit_cell_free_site(tmp_path)
    metal = next(atom for atom in context.contact_atoms if atom.element == "ZN")
    neighbor = next(atom for atom in context.contact_atoms if atom.atom_name == "O1")
    connection = gemmi.Connection()
    connection.asu = gemmi.Asu.Different

    with pytest.raises(ValueError, match="missing_or_invalid_unit_cell"):
        declared_connections.declared_candidate_geometry(
            context, metal, neighbor, connection
        )


def test_unresolved_geometry_is_an_issue_without_candidate_warnings(
    tmp_path: Path,
) -> None:
    """The donor class of a contact that was never measured is not reported.

    ``LIG`` is outside the scored residue classes, which the old ordering
    reported although the declaration produced no candidate to attach it to.
    """
    builder, zinc, ligand, context = _unit_cell_free_site(tmp_path)
    builder.add_connection(
        zinc.ref("ZN"), ligand.ref("O1"), name="sym", asu="different"
    )

    _, candidate, issues, warnings = _resolve_in_memory(context, builder)

    assert candidate is None
    assert warnings == []
    assert len(issues) == 1
    assert issues[0].startswith("LINK sym geometry unresolved: ValueError:")
    assert "missing_or_invalid_unit_cell" in issues[0]


@pytest.mark.parametrize(
    "reported",
    [-1.0, 0.0, float("inf"), float("-inf"), float("nan")],
)
def test_an_unusable_reported_distance_is_recorded_as_blank(
    tmp_path: Path, reported: float
) -> None:
    """A negative, zero, infinite or absent declared distance is not a value."""
    builder, his, zinc = _zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=reported
    )
    context = load_structure("test", builder.write_pdb(tmp_path / "analysis.pdb"))

    _, candidate, issues, warnings = _resolve_in_memory(context, builder)

    assert candidate is not None
    assert (issues, warnings) == ([], [])
    (record,) = candidate.declared_connections
    assert math.isnan(record.connection_reported_distance)


def test_a_usable_reported_distance_is_kept(tmp_path: Path) -> None:
    """A positive finite declared distance is retained as deposited."""
    builder, his, zinc = _zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    context = load_structure("test", builder.write_pdb(tmp_path / "analysis.pdb"))

    _, candidate, _, _ = _resolve_in_memory(context, builder)

    assert candidate is not None
    (record,) = candidate.declared_connections
    assert record.connection_reported_distance == approx(2.03)


def test_repeated_connection_ids_are_disambiguated(tmp_path: Path) -> None:
    """Two declarations sharing an id keep distinct provenance records.

    Candidate merging keys declaration records on ``(source, connection_id)``,
    so a repeated id would silently drop one deposited record.
    """
    builder, his, zinc = _zinc_histidine_site(
        {"NE2": (2.03, 0.0, 0.0), "ND1": (0.0, 2.10, 0.0)}
    )
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="dup", reported_distance=2.03
    )
    builder.add_connection(
        zinc.ref("ZN"), his.ref("ND1"), name="dup", reported_distance=2.10
    )
    source, analysis_pdb = _write_source_and_analysis(builder, tmp_path, "cif")
    context = load_structure("test", analysis_pdb)

    candidates, issues, _ = declared_connections.collect_declared_candidates(
        context, source, context.metal_atoms(METAL_ELEMENTS, canonical=True)
    )

    assert issues == []
    # Each resolved candidate is returned beside the metal site it was resolved
    # against; ``coordination.analysis`` groups the declarations by that key.
    assert {declared.metal.atom_name for declared in candidates} == {"ZN"}
    neighbors = [declared.candidate.neighbor.atom_name for declared in candidates]
    identifiers = [
        record.connection_id
        for declared in candidates
        for record in declared.candidate.declared_connections
    ]
    assert neighbors == ["NE2", "ND1"]
    assert identifiers == ["dup", "dup_2"]
