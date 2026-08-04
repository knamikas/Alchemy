"""Declared ``_struct_conn`` / ``LINK`` handling in ``bond_analysis``.

Alchemy reads coordination declarations from the authoritative source file --
``_struct_conn`` in an mmCIF, ``LINK`` records in a legacy PDB -- and resolves
each partner onto the *analysis* model, which for an mmCIF entry is a converted
PDB. This module protects that resolution and everything that hangs off it:

* format dispatch (``_connection_source``);
* the chain-name and component-name ambiguities the conversion introduces
  (``_analysis_chain_names`` plus the ``REMARK 950 ALCHEMY RESNAME`` provenance);
* author-identity resolution and its ``None`` results
  (``_analysis_atom_for_partner``);
* alternate-conformer re-pointing (``_selected_conformer_atom``);
* how a declared contact reaches the bond rows: merged with a coincident
  proximity candidate, retained outside the 4 A search radius and outside
  first-sphere eligibility, and overriding the typical-donor table;
* the auditable codes ``declared_connection_resolution_incomplete`` and
  ``declared_connection_conformer_substituted``.

Two fixed bugs are covered by dedicated regression tests, named
``test_regression_*``.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace
from typing import NamedTuple, Optional, Sequence

import pytest
import gemmi

import helpers
from helpers import AtomRef, AtomSpec, StructureBuilder

import bond_analysis
import coordinate_conversion as conversion
from structure_analysis import StructureContext, load_structure
import reference_data


# --------------------------------------------------------------------------- #
# Local scaffolding (private to this module by the suite's ownership rules)
# --------------------------------------------------------------------------- #
class Analysis(NamedTuple):
    """Everything one ``run_bond_analysis`` call produced, plus its context."""

    context: StructureContext
    rows: list
    candidates: list
    summaries: dict
    metadata: dict

    def rows_for(self, atom_name: str, resnum: Optional[str] = None) -> list:
        return [
            row
            for row in self.rows
            if row["neighbor_atom"] == atom_name
            and (resnum is None or row["neighbor_resnum"] == resnum)
        ]

    @property
    def declared_rows(self) -> list:
        return [row for row in self.rows if row["declared_connection"]]


def write_source_and_analysis(builder: StructureBuilder, directory, fmt: str) -> tuple:
    """Write ``builder`` as the source file and return ``(source, analysis)``.

    ``fmt="pdb"`` mirrors a PDB-sourced entry: Alchemy analyzes the deposited
    coordinates themselves, so both paths are the same file. ``fmt="cif"``
    mirrors an mmCIF-sourced entry: the analyzed coordinates are produced by the
    real ``conversion._cif_to_pdb`` conversion, which is where every identifier
    ambiguity this module cares about is introduced.
    """
    directory = str(directory)
    if fmt == "cif":
        source = builder.write_cif(os.path.join(directory, "source.cif"))
        return source, conversion._cif_to_pdb(
            source, os.path.join(directory, "analysis.pdb")
        )
    if fmt == "pdb":
        source = builder.write_pdb(os.path.join(directory, "source.pdb"))
        return source, source
    raise ValueError(f"unknown source format {fmt!r}")


def blank_struct_conn_atom_names(path: str, *partners: int) -> None:
    """Remove selected partner atom names from every source declaration."""
    document = gemmi.cif.read(path)
    block = document.sole_block()
    for partner in partners:
        column = block.find_values(f"_struct_conn.ptnr{partner}_label_atom_id")
        assert len(column) > 0
        for index in range(len(column)):
            column[index] = "?"
    document.write_file(path)


def analyze(
    analysis_pdb: str,
    connection_path: Optional[str] = None,
    dpi_inputs=None,
    pdb_id: str = "test",
) -> Analysis:
    """Run the real bond analysis over an already-written analysis PDB."""
    context = load_structure(pdb_id, analysis_pdb)
    rows, candidates, summaries, metadata = bond_analysis.run_bond_analysis(
        pdb_id,
        analysis_pdb,
        [],
        list(helpers.EDSTATS_HEADER),
        dpi_inputs if dpi_inputs is not None else helpers.dpi_inputs(),
        structure=context,
        connection_path=connection_path,
    )
    return Analysis(context, rows, candidates, summaries, metadata)


def zinc_histidine_site(
    donor_pos=(2.03, 0.0, 0.0),
    *,
    chain: str = "A",
    hetero_chain: str = "B",
    seqid: int = 10,
    icode: str = "",
) -> tuple:
    """A one-histidine zinc site: returns ``(builder, his, zinc)``.

    The polymer residue is added first so gemmi's PDB writer emits its ``TER``
    *before* the hetero metal -- the arrangement that made serial-based partner
    resolution wrong.
    """
    builder = StructureBuilder()
    his = builder.add_amino_acid(
        "HIS",
        seqid,
        chain=chain,
        icode=icode,
        positions={"NE2": tuple(float(value) for value in donor_pos)},
        origin=(20.0, 20.0, 20.0),
    )
    zinc = builder.add_metal("ZN", 1, chain=hetero_chain, pos=(0.0, 0.0, 0.0))
    return builder, his, zinc


def partner_address(
    chain: str,
    resname: str,
    seqid: int,
    atom_name: str,
    icode: str = " ",
    altloc: str = "",
) -> SimpleNamespace:
    """A stand-in for the ``CRA`` gemmi returns for a declared partner."""
    return SimpleNamespace(
        chain=SimpleNamespace(name=chain),
        residue=SimpleNamespace(
            name=resname, seqid=SimpleNamespace(num=seqid, icode=icode)
        ),
        atom=SimpleNamespace(name=atom_name, altloc=altloc),
    )


class AmbiguousStructure:
    """A structure whose author lookup deliberately reports ``n`` matches."""

    def __init__(self, context, matches: Sequence) -> None:
        self._context = context
        self._matches = tuple(matches)

    def residues_for_author(self, residue_name, chain_id, resnum):
        return self._matches

    def __getattr__(self, name):
        return getattr(self._context, name)


SOURCE_FORMATS = ("pdb", "cif")


# --------------------------------------------------------------------------- #
# _connection_source: which parser the coordinate file selects
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path, expected",
    [
        ("/data/1abc.cif", "struct_conn"),
        ("/data/1abc.mmcif", "struct_conn"),
        ("/data/1abc.cif.gz", "struct_conn"),
        ("/data/1ABC.CIF", "struct_conn"),
        ("/data/1abc.pdb", "LINK"),
        ("/data/1abc.pdb.gz", "LINK"),
        ("/data/1abc.ent", "LINK"),
        ("/data/pdb1abc", "LINK"),
        ("", "LINK"),
        (None, "LINK"),
    ],
)
def test_connection_source_dispatches_on_coordinate_format(path, expected):
    """Only mmCIF names select the ``_struct_conn`` reader; everything else is LINK.

    The label is not cosmetic: it becomes ``coordination_source`` on every
    declared bond row and prefixes the resolution messages, so an entry's
    provenance must follow the file it was actually read from -- including
    through a ``.gz`` suffix and regardless of case.
    """
    assert bond_analysis._connection_source(path) == expected


def test_connection_source_labels_reach_the_bond_row(tmp_path):
    """A declared row records the format the declaration was read from."""
    labels = {}
    for fmt in SOURCE_FORMATS:
        directory = tmp_path / fmt
        directory.mkdir()
        builder, his, zinc = zinc_histidine_site()
        builder.add_connection(
            zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
        )
        source, analysis_pdb = write_source_and_analysis(builder, directory, fmt)
        result = analyze(analysis_pdb, connection_path=source)
        (row,) = result.declared_rows
        labels[fmt] = row["coordination_source"]
    assert labels == {"pdb": "LINK", "cif": "struct_conn"}


# --------------------------------------------------------------------------- #
# _analysis_chain_names: reversing the conversion's chain shortening
# --------------------------------------------------------------------------- #
def test_analysis_chain_names_reverses_conversion_shortening(tmp_path):
    """Long mmCIF chain names map onto the single-character analysis names.

    The mapping must agree with what ``conversion._cif_to_pdb`` really wrote, so the
    expectation is taken from the converted file rather than assumed.
    """
    builder = StructureBuilder()
    builder.add_amino_acid(
        "HIS",
        10,
        chain="AAA",
        positions={"NE2": (2.03, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    builder.add_metal("ZN", 1, chain="BBB", pos=(0.0, 0.0, 0.0))
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    mapping = bond_analysis._analysis_chain_names(source)
    context = load_structure("test", analysis_pdb)
    converted_chains = {residue.chain_id for residue in context.residues}

    assert mapping == {"AAA": "A", "BBB": "B"}
    assert converted_chains == {"A", "B"}
    assert set(mapping.values()) <= converted_chains


def test_analysis_chain_names_is_empty_for_a_pdb_source(tmp_path):
    """A PDB source is analyzed as deposited, so no chain renaming is replayed.

    Returning a mapping here would rename chains that were never renamed.
    """
    builder, his, zinc = zinc_histidine_site()
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")
    assert bond_analysis._analysis_chain_names(source) == {}


def test_analysis_chain_names_omits_chains_conversion_leaves_alone(tmp_path):
    """Short mmCIF chain names survive conversion, so the mapping stays empty."""
    builder, his, zinc = zinc_histidine_site()
    source, _ = write_source_and_analysis(builder, tmp_path, "cif")
    assert bond_analysis._analysis_chain_names(source) == {}


def test_analysis_chain_names_tolerates_a_model_free_file(tmp_path):
    """A coordinate file with no model yields an empty mapping, not an error."""
    empty = tmp_path / "empty.cif"
    empty.write_text("data_empty\n_entry.id EMPTY\n", encoding="utf-8")
    assert bond_analysis._analysis_chain_names(str(empty)) == {}


# --------------------------------------------------------------------------- #
# _analysis_atom_for_partner: author-identity resolution and its None results
# --------------------------------------------------------------------------- #
def test_analysis_atom_for_partner_resolves_author_identity(tmp_path):
    """Chain, sequence id, insertion code, component and atom name select one atom."""
    builder = StructureBuilder()
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (2.60, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        icode="A",
        positions={"NE2": (0.0, 2.03, 0.0)},
        origin=(40.0, 20.0, 20.0),
    )
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    context = load_structure("test", builder.write_pdb(tmp_path / "s.pdb"))

    plain = bond_analysis._analysis_atom_for_partner(
        context, partner_address("A", "HIS", 10, "NE2"), {}
    )
    with_icode = bond_analysis._analysis_atom_for_partner(
        context, partner_address("A", "HIS", 10, "NE2", icode="A"), {}
    )

    assert plain is not None and with_icode is not None
    assert plain.resnum == "10" and with_icode.resnum == "10A"
    assert plain.xyz == (2.60, 0.0, 0.0)
    assert with_icode.xyz == (0.0, 2.03, 0.0)


def test_analysis_atom_for_partner_applies_the_chain_name_mapping(tmp_path):
    """A source chain name is translated before the author lookup, not after."""
    builder = StructureBuilder()
    builder.add_amino_acid(
        "HIS",
        10,
        chain="AAA",
        positions={"NE2": (2.03, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    builder.add_metal("ZN", 1, chain="BBB", pos=(0.0, 0.0, 0.0))
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")
    context = load_structure("test", analysis_pdb)
    address = partner_address("AAA", "HIS", 10, "NE2")

    assert bond_analysis._analysis_atom_for_partner(context, address, {}) is None
    resolved = bond_analysis._analysis_atom_for_partner(
        context, address, bond_analysis._analysis_chain_names(source)
    )
    assert resolved is not None
    assert (resolved.chain_id, resolved.atom_name) == ("A", "NE2")


@pytest.mark.parametrize(
    "address, reason",
    [
        (partner_address("A", "HIS", 999, "NE2"), "no residue with that number"),
        (partner_address("Z", "HIS", 10, "NE2"), "no residue in that chain"),
        (partner_address("A", "GLU", 10, "NE2"), "component does not match"),
        (
            partner_address("A", "HIS", 10, "NE2", icode="B"),
            "insertion code does not match",
        ),
        (partner_address("A", "HIS", 10, "OXT"), "residue holds no such atom"),
        (
            partner_address("A", "HIS", 10, "NE2", altloc="C"),
            "residue holds no such conformer",
        ),
    ],
)
def test_analysis_atom_for_partner_returns_none_for_unmatched_identity(
    tmp_path, address, reason
):
    """An identity that names no atom of the analysis model resolves to None.

    Returning a near miss here is what produced contacts at tens of Angstrom,
    so every mismatched component of the identity must reject rather than
    approximate.
    """
    builder, his, zinc = zinc_histidine_site()
    context = load_structure("test", builder.write_pdb(tmp_path / "s.pdb"))
    assert bond_analysis._analysis_atom_for_partner(context, address, {}) is None, (
        reason
    )


def test_analysis_atom_for_partner_refuses_an_ambiguous_residue(tmp_path):
    """Two residues sharing one author identity resolve to None, not to the first.

    Picking either would silently attach the declaration to an arbitrary atom.
    """
    builder, his, zinc = zinc_histidine_site()
    context = load_structure("test", builder.write_pdb(tmp_path / "s.pdb"))
    address = partner_address("A", "HIS", 10, "NE2")
    single = bond_analysis._analysis_atom_for_partner(context, address, {})
    assert single is not None

    duplicated = AmbiguousStructure(context, [context.residues[0], context.residues[0]])
    assert bond_analysis._analysis_atom_for_partner(duplicated, address, {}) is None
    assert (
        bond_analysis._analysis_atom_for_partner(
            AmbiguousStructure(context, []), address, {}
        )
        is None
    )


@pytest.mark.parametrize(
    "cra",
    [
        SimpleNamespace(chain=None, residue=None, atom=None),
        SimpleNamespace(chain=SimpleNamespace(name="A"), residue=None, atom=None),
        SimpleNamespace(),
    ],
)
def test_analysis_atom_for_partner_tolerates_an_unresolvable_address(tmp_path, cra):
    """A partner gemmi could not locate at all resolves to None without raising."""
    builder, his, zinc = zinc_histidine_site()
    context = load_structure("test", builder.write_pdb(tmp_path / "s.pdb"))
    assert bond_analysis._analysis_atom_for_partner(context, cra, {}) is None


# --------------------------------------------------------------------------- #
# _selected_conformer_atom: re-pointing onto the analyzed conformer
# --------------------------------------------------------------------------- #
def conformer_context(
    tmp_path,
    *,
    occupancies=(0.35, 0.65),
    positions=((2.03, 0.0, 0.0), (2.20, 0.0, 0.0)),
):
    """Load a HIS whose NE2 is split into conformers A and B."""
    builder, his, zinc = zinc_histidine_site()
    builder.add_conformers(
        his,
        [
            ("A", occupancies[0], {"NE2": positions[0]}),
            ("B", occupancies[1], {"NE2": positions[1]}),
        ],
        atom_names=["NE2"],
    )
    return builder, load_structure(
        "test", builder.write_pdb(tmp_path / "conformers.pdb")
    )


def test_selected_conformer_atom_repoints_a_deselected_record(tmp_path):
    """A declaration naming conformer A is answered with the selected conformer B.

    Contacts are measured on the selected conformer only; keeping the A record
    would add a second measurement of one chemical site.
    """
    _, context = conformer_context(tmp_path)
    residue = context.residues_for_author("HIS", "A", "10")[0]
    assert residue.selected_altloc == "B"
    deselected = next(
        atom
        for atom in residue.source_atoms
        if atom.atom_name == "NE2" and atom.altloc == "A"
    )

    selected = bond_analysis._selected_conformer_atom(context, deselected)

    assert selected is not None
    assert (selected.atom_name, selected.altloc) == ("NE2", "B")
    assert selected.xyz == (2.20, 0.0, 0.0)


def test_selected_conformer_atom_returns_a_selected_record_unchanged(tmp_path):
    """An atom already on the selected conformer is returned as itself."""
    _, context = conformer_context(tmp_path)
    residue = context.residues_for_author("HIS", "A", "10")[0]
    chosen = next(atom for atom in residue.contact_atoms if atom.atom_name == "NE2")
    assert bond_analysis._selected_conformer_atom(context, chosen) is chosen


def test_selected_conformer_atom_is_none_when_the_conformer_lacks_the_atom(tmp_path):
    """No same-named atom on the selected conformer means no analyzable counterpart."""
    builder, his, zinc = zinc_histidine_site()
    his.atoms = [atom for atom in his.atoms if atom.name not in ("NE2", "ND1")] + [
        AtomSpec("NE2", "N", (2.03, 0.0, 0.0), occupancy=0.35, altloc="A"),
        AtomSpec("ND1", "N", (2.05, 0.0, 0.0), occupancy=0.65, altloc="B"),
    ]
    context = load_structure("test", builder.write_pdb(tmp_path / "s.pdb"))
    residue = context.residues_for_author("HIS", "A", "10")[0]
    assert residue.selected_altloc == "B"
    orphan = next(atom for atom in residue.source_atoms if atom.atom_name == "NE2")

    assert bond_analysis._selected_conformer_atom(context, orphan) is None
    assert bond_analysis._selected_conformer_atom(context, None) is None


# --------------------------------------------------------------------------- #
# REGRESSION (bug 1, 733d8ec): resolution must not ride on atom serials
# --------------------------------------------------------------------------- #
def test_regression_declared_partner_survives_the_ter_serial_shift(tmp_path):
    """A declaration in an mmCIF still binds the right atoms after PDB conversion.

    gemmi's PDB writer emits a ``TER`` after the polymer chain and that ``TER``
    consumes a serial number, so every analysis serial past it runs ahead of the
    source ``_atom_site.id``. Mapping source serials into analysis serial space
    therefore attached the declaration to a neighbouring atom -- or to nothing at
    all -- for every metal written after the first ``TER``.

    The oracle is the declaration's own reported distance: a correctly bound
    partner pair measures exactly what the source claimed. A shifted partner
    either measures something else entirely or leaves no declared row at all.
    """
    builder = StructureBuilder()
    his = builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (2.03, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    builder.add_amino_acid(
        "GLU",
        11,
        chain="A",
        positions={"OE1": (0.0, -2.05, 0.0)},
        origin=(32.0, 20.0, 20.0),
    )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (0.0, 0.0, 2.09), chain="B")
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    # The premise: the conversion really did insert a TER ahead of the metal,
    # so the metal's analysis serial differs from its source _atom_site.id.
    lines = [line for line in open(analysis_pdb, encoding="utf-8")]
    ter_index = next(
        index for index, line in enumerate(lines) if line.startswith("TER")
    )
    metal_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("HETATM") and line[12:16].strip() == "ZN"
    )
    assert ter_index < metal_index
    metal_serial = int(lines[metal_index][6:11])
    import gemmi

    source_serial = next(
        atom.serial
        for chain in gemmi.read_structure(source)[0]
        for residue in chain
        for atom in residue
        if atom.name == "ZN"
    )
    assert metal_serial == source_serial + 1

    result = analyze(analysis_pdb, connection_path=source)

    declared = result.declared_rows
    assert len(declared) == 1, [
        (row["neighbor_resname"], row["neighbor_atom"], row["distance"])
        for row in declared
    ]
    row = declared[0]
    # Bound to the declared atoms, not to their neighbours in serial order.
    assert (row["metal_element"], row["metal_resnum"]) == ("ZN", "1")
    assert (
        row["neighbor_resname"],
        row["neighbor_atom"],
        row["neighbor_chain"],
        row["neighbor_resnum"],
    ) == ("HIS", "NE2", "A", "10")
    # The strong oracle: measured geometry agrees with the declaration itself.
    assert row["distance"] == pytest.approx(
        float(row["connection_reported_distance"]), abs=1e-3
    )
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)
    assert (
        "declared_connection_resolution_incomplete"
        not in (result.metadata["partial_reason_codes"])
    )


@pytest.mark.parametrize("fmt", SOURCE_FORMATS)
def test_regression_declared_distance_matches_the_declaration_in_both_formats(
    tmp_path, fmt
):
    """Both source formats bind the same atoms and measure the same distance.

    The mmCIF path goes through the conversion; the PDB path does not. A join
    that only holds for one representation is a silent, format-dependent
    corruption of every declared row.
    """
    builder = StructureBuilder()
    for index, (resname, atom_name, offset) in enumerate(
        (
            ("HIS", "NE2", (2.03, 0.0, 0.0)),
            ("GLU", "OE1", (0.0, -2.05, 0.0)),
            ("CYS", "SG", (0.0, 0.0, -2.31)),
        )
    ):
        builder.add_amino_acid(
            resname,
            10 + index,
            chain="A",
            positions={atom_name: offset},
            origin=(20.0 + 12.0 * index, 20.0, 20.0),
        )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    for residue, atom_name, distance in (
        (builder.residues[0], "NE2", 2.03),
        (builder.residues[1], "OE1", 2.05),
        (builder.residues[2], "SG", 2.31),
    ):
        builder.add_connection(
            zinc.ref("ZN"),
            residue.ref(atom_name),
            name=f"m{residue.seqid}",
            reported_distance=distance,
        )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, fmt)

    result = analyze(analysis_pdb, connection_path=source)

    declared = {
        (row["neighbor_resname"], row["neighbor_atom"]): row
        for row in result.declared_rows
    }
    assert set(declared) == {("HIS", "NE2"), ("GLU", "OE1"), ("CYS", "SG")}
    for row in declared.values():
        assert row["distance"] == pytest.approx(
            float(row["connection_reported_distance"]), abs=1e-3
        )


@pytest.mark.parametrize("fmt", SOURCE_FORMATS)
def test_regression_metal_may_be_the_second_declared_partner(tmp_path, fmt):
    """Partner order is immaterial for both LINK and ``_struct_conn`` records."""
    builder, histidine, zinc = zinc_histidine_site()
    builder.add_connection(
        histidine.ref("NE2"), zinc.ref("ZN"), name="donor-first", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, fmt)

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.declared_rows
    assert (row["metal_element"], row["metal_atom"]) == ("ZN", "ZN")
    assert (row["neighbor_resname"], row["neighbor_atom"]) == ("HIS", "NE2")
    assert row["coordination_source"] == ("LINK" if fmt == "pdb" else "struct_conn")
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)
    assert (
        "declared_connection_resolution_incomplete"
        not in (result.metadata["partial_reason_codes"])
    )


def test_declared_partner_resolves_through_shortened_and_renamed_components(tmp_path):
    """Conversion-time renaming of chains and components stays reversible.

    ``AAA``/``BBB`` do not fit the legacy chain field and ``A1LU6`` does not fit
    the legacy residue field, so the analysis PDB holds ``A``/``B`` and ``A1L``
    plus a ``REMARK 950 ALCHEMY RESNAME`` mapping. The declaration still names
    the source identifiers and must resolve through both.
    """
    builder = StructureBuilder()
    his = builder.add_amino_acid(
        "HIS",
        10,
        chain="AAA",
        positions={"NE2": (2.03, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    cofactor = builder.add_hetero_residue(
        "A1LU6", 1, [AtomSpec("ZN", "ZN", (0.0, 0.0, 0.0))], chain="BBB"
    )
    builder.add_connection(
        cofactor.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    remarks = [
        line
        for line in open(analysis_pdb, encoding="utf-8")
        if line.startswith("REMARK 950 ALCHEMY RESNAME")
    ]
    assert any("A1L A1LU6" in line for line in remarks)

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.declared_rows
    assert (row["metal_resname"], row["metal_chain"]) == ("A1LU6", "B")
    assert (row["neighbor_chain"], row["neighbor_resnum"], row["neighbor_atom"]) == (
        "A",
        "10",
        "NE2",
    )
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)


@pytest.mark.parametrize("fmt", SOURCE_FORMATS)
def test_declared_partner_is_distinguished_by_insertion_code(tmp_path, fmt):
    """``10`` and ``10A`` are different residues to a declaration.

    Both carry the same component and sequence number, so dropping the
    insertion code from the identity would bind the wrong histidine -- and its
    decoy NE2 sits at a different distance, which the measurement exposes.
    """
    builder = StructureBuilder()
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (2.60, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    tagged = builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        icode="A",
        positions={"NE2": (0.0, 2.03, 0.0)},
        origin=(40.0, 20.0, 20.0),
    )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_connection(
        zinc.ref("ZN"), tagged.ref("NE2"), name="ic", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, fmt)

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.declared_rows
    assert (row["neighbor_resnum"], row["neighbor_icode"]) == ("10A", "A")
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)
    # The decoy is still reported, but only as a proximity inference.
    (decoy,) = [
        candidate for candidate in result.rows if candidate["neighbor_resnum"] == "10"
    ]
    assert decoy["coordination_status"] == "inferred"
    assert decoy["distance"] == pytest.approx(2.60, abs=1e-6)


# --------------------------------------------------------------------------- #
# REGRESSION (bug 3, dd63940): declarations obey alternate-conformer selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", SOURCE_FORMATS)
def test_regression_declaration_on_a_deselected_conformer_yields_one_row(tmp_path, fmt):
    """A LINK naming conformer A produces exactly one bond, on selected conformer B.

    Occupancy selection keeps conformer B (0.65) and the proximity search
    already reports its NE2. Admitting the declaration's own A record as well
    produced two bond rows for one chemical atom: the coordination number was
    inflated, the two rows formed a fabricated multi-donor group, and the
    confidence denominator counted the site twice.
    """
    builder, his, zinc = zinc_histidine_site()
    builder.add_conformers(
        his,
        [
            ("A", 0.35, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.65, {"NE2": (2.20, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2", altloc="A"), name="m1", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, fmt)

    result = analyze(analysis_pdb, connection_path=source)

    assert (
        result.context.residues_for_author("HIS", "A", "10")[0].selected_altloc == "B"
    )
    ne2_rows = result.rows_for("NE2")
    assert len(ne2_rows) == 1, [
        (row["neighbor_altloc"], row["distance"]) for row in ne2_rows
    ]
    (row,) = ne2_rows
    assert row["neighbor_altloc"] == "B"
    assert row["distance"] == pytest.approx(2.20, abs=1e-6)
    assert row["declared_connection"] is True
    assert row["multi_donor_detected"] is False
    assert row["multi_donor_contact_count"] == 1
    # The substitution is auditable rather than silent.
    assert (
        "declared_connection_conformer_substituted"
        in (result.metadata["warning_codes"])
    )
    # One candidate record too: the declaration did not add a second image.
    assert (
        len(
            [
                candidate
                for candidate in result.candidates
                if candidate["neighbor_atom"] == "NE2"
            ]
        )
        == 1
    )


def test_declaration_on_the_selected_conformer_needs_no_substitution(tmp_path):
    """Naming the selected conformer resolves without the substitution warning.

    The warning must mark a real re-pointing, otherwise it is noise on every
    entry that models alternates.
    """
    builder, his, zinc = zinc_histidine_site()
    builder.add_conformers(
        his,
        [
            ("A", 0.35, {"NE2": (2.03, 0.0, 0.0)}),
            ("B", 0.65, {"NE2": (2.20, 0.0, 0.0)}),
        ],
        atom_names=["NE2"],
    )
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2", altloc="B"), name="m1", reported_distance=2.20
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.declared_rows
    assert row["neighbor_altloc"] == "B"
    assert row["distance"] == pytest.approx(2.20, abs=1e-6)
    assert (
        "declared_connection_conformer_substituted"
        not in (result.metadata["warning_codes"])
    )


def test_declaration_is_dropped_when_the_selected_conformer_lacks_the_atom(tmp_path):
    """A declared atom absent from the analyzed conformer is reported, not guessed.

    Conformer A models NE2 and conformer B models ND1; B is selected, so the
    declared NE2 has no counterpart in the analyzed model. The declaration is
    dropped and the entry is marked partially resolved.
    """
    builder, his, zinc = zinc_histidine_site()
    his.atoms = [atom for atom in his.atoms if atom.name not in ("NE2", "ND1")] + [
        AtomSpec("NE2", "N", (2.03, 0.0, 0.0), occupancy=0.35, altloc="A"),
        AtomSpec("ND1", "N", (2.05, 0.0, 0.0), occupancy=0.65, altloc="B"),
    ]
    builder.add_connection(
        zinc.ref("ZN"),
        AtomRef("A", "HIS", 10, "", "NE2", "A"),
        name="m1",
        reported_distance=2.03,
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")

    result = analyze(analysis_pdb, connection_path=source)

    assert result.declared_rows == []
    assert (
        "declared_connection_resolution_incomplete"
        in (result.metadata["partial_reason_codes"])
    )
    assert any("conformer" in message for message in result.metadata["messages"])
    # The selected conformer's own donor is still found by proximity.
    (row,) = result.rows_for("ND1")
    assert row["coordination_status"] == "inferred"


# --------------------------------------------------------------------------- #
# Resolution failures that must be reported rather than swallowed
# --------------------------------------------------------------------------- #
def test_broken_amino_acid_connection_is_outside_metal_analysis(tmp_path):
    """An unresolved amino-acid link is neither a partial nor a warning."""
    builder = StructureBuilder()
    asparagine = builder.add_amino_acid("ASN", 1, chain="A")
    serine = builder.add_amino_acid("SER", 2, chain="A")
    builder.add_connection(
        asparagine.ref("ND2"), serine.ref("OG"), name="broken-aa", type="covale"
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")
    blank_struct_conn_atom_names(source, 1, 2)

    result = analyze(analysis_pdb, connection_path=source)

    assert (
        "declared_connection_resolution_incomplete"
        not in (result.metadata["partial_reason_codes"])
    )
    assert (
        "declared_connection_partner_unresolved"
        not in (result.metadata["warning_codes"])
    )
    assert result.metadata["messages"] == []


def test_broken_connection_that_names_a_metal_is_partial(tmp_path):
    """A missing donor name remains material when the other partner is Zn."""
    builder, histidine, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), histidine.ref("NE2"), name="broken-metal", type="covale"
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")
    blank_struct_conn_atom_names(source, 2)

    result = analyze(analysis_pdb, connection_path=source)

    assert (
        "declared_connection_resolution_incomplete"
        in (result.metadata["partial_reason_codes"])
    )
    assert any(
        "broken-metal" in message and "unresolved" in message
        for message in result.metadata["messages"]
    )


def test_partner_absent_from_the_analyzed_model_is_reported(tmp_path):
    """A declaration Alchemy cannot bind marks the entry partially resolved.

    The source declares a metal-GLU bond, but the analyzed coordinates do not
    contain that residue. Dropping the declaration silently would understate the
    coordination number with no audit trail.
    """

    def build(with_glutamate):
        builder, his, zinc = zinc_histidine_site()
        glutamate = None
        if with_glutamate:
            glutamate = builder.add_amino_acid(
                "GLU",
                11,
                chain="A",
                positions={"OE1": (0.0, 2.05, 0.0)},
                origin=(40.0, 20.0, 20.0),
            )
        return builder, zinc, glutamate

    source_builder, source_zinc, glutamate = build(True)
    assert glutamate is not None
    source_builder.add_connection(
        source_zinc.ref("ZN"), glutamate.ref("OE1"), name="gone", reported_distance=2.05
    )
    source = source_builder.write_cif(tmp_path / "source.cif")
    analysis_builder, _, _ = build(False)
    analysis_pdb = analysis_builder.write_pdb(tmp_path / "analysis.pdb")

    result = analyze(analysis_pdb, connection_path=source)

    assert result.declared_rows == []
    assert (
        "declared_connection_resolution_incomplete"
        in (result.metadata["partial_reason_codes"])
    )
    assert any(
        "gone" in message and "unresolved" in message
        for message in result.metadata["messages"]
    )


def test_unparsable_connection_file_is_reported_and_not_fatal(tmp_path):
    """A source file gemmi cannot read degrades to a partial result.

    Bond analysis still emits its proximity-derived rows; only the declaration
    evidence is missing, and that fact is recorded.
    """
    builder, his, zinc = zinc_histidine_site()
    analysis_pdb = builder.write_pdb(tmp_path / "analysis.pdb")
    broken = tmp_path / "broken.cif"
    broken.write_text("this is not a coordinate file\n", encoding="utf-8")

    result = analyze(analysis_pdb, connection_path=str(broken))

    assert (
        "declared_connection_resolution_incomplete"
        in (result.metadata["partial_reason_codes"])
    )
    assert any(
        "struct_conn" in message and "parse failed" in message
        for message in result.metadata["messages"]
    )
    (row,) = result.rows_for("NE2")
    assert row["coordination_status"] == "inferred"


# --------------------------------------------------------------------------- #
# Declarations that must be ignored without complaint
# --------------------------------------------------------------------------- #
def test_declaration_between_two_metals_is_not_a_coordination_row(tmp_path):
    """A metal-metal LINK describes a cluster, not a metal-donor contact."""
    builder, his, zinc = zinc_histidine_site()
    second = builder.add_metal("ZN", 2, chain="B", pos=(3.2, 0.0, 0.0))
    builder.add_connection(
        zinc.ref("ZN"), second.ref("ZN"), name="mm", reported_distance=3.2
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")

    result = analyze(analysis_pdb, connection_path=source)

    assert result.declared_rows == []
    assert (
        "declared_connection_resolution_incomplete"
        not in (result.metadata["partial_reason_codes"])
    )


def test_declaration_to_a_non_donor_element_is_ignored(tmp_path):
    """A declared metal-carbon contact is not admitted as a coordination bond."""
    builder = StructureBuilder()
    his = builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (2.03, 0.0, 0.0), "CE1": (0.0, 2.90, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_connection(
        zinc.ref("ZN"), his.ref("CE1"), name="c1", reported_distance=2.90
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")

    result = analyze(analysis_pdb, connection_path=source)

    assert result.rows_for("CE1") == []
    assert result.declared_rows == []
    assert (
        "declared_connection_resolution_incomplete"
        not in (result.metadata["partial_reason_codes"])
    )


def test_declaration_inside_a_cofactor_is_not_an_external_contact(tmp_path):
    """An intra-cofactor LINK does not become a metal-donor bond row.

    Deposited ``_struct_conn`` records routinely wire up a cluster's own atoms;
    those describe the cofactor's internal chemistry, and the bond rows describe
    external coordination.
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
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")

    result = analyze(analysis_pdb, connection_path=source)

    assert result.rows_for("S1") == []
    assert result.declared_rows == []


# --------------------------------------------------------------------------- #
# How a declared contact reaches the bond rows
# --------------------------------------------------------------------------- #
def test_declaration_merges_with_a_coincident_proximity_candidate(tmp_path):
    """One atom image yields one row that carries both provenances.

    A donor found by the 4 A search and also declared must not be reported
    twice; the merged row keeps declared status while recording that the
    proximity search saw it too.
    """
    builder, his, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.rows_for("NE2")
    assert row["coordination_status"] == "declared"
    assert row["coordination_source"] == "struct_conn"
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)

    (candidate,) = [
        entry for entry in result.candidates if entry["neighbor_atom"] == "NE2"
    ]
    assert set(candidate["candidate_source"].split("|")) == {
        "proximity_4A",
        "struct_conn",
    }
    assert candidate["inferred_contact_eligible"] is True
    assert candidate["coordination_status"] == "declared"


def test_declared_contact_is_retained_beyond_the_search_radius(tmp_path):
    """A declaration outside the 4 A discovery radius is still a bond row.

    The proximity search never sees this atom, so the row exists only because
    the model declared it -- with its measured distance, not the reported one.
    """
    builder, his, zinc = zinc_histidine_site(donor_pos=(6.50, 0.0, 0.0))
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="far", reported_distance=6.50
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.rows_for("NE2")
    assert row["coordination_status"] == "declared"
    assert row["distance"] == pytest.approx(6.50, abs=1e-6)
    assert row["contact_scope"] == "explicit"

    (candidate,) = [
        entry for entry in result.candidates if entry["neighbor_atom"] == "NE2"
    ]
    assert candidate["candidate_source"] == "LINK"
    assert candidate["first_sphere_eligible"] is False
    assert candidate["eligibility_status"] == "outside_first_sphere"


def test_declared_contact_outside_first_sphere_keeps_measured_geometry(tmp_path):
    """A declared bond too long for the distance rule still gets Zbond and a flag.

    README: declared contacts remain bonds even when they fail first-sphere
    eligibility, and their Zbond is calculated whenever the literature reference
    and DPI exist. The expected z follows the documented formula from the
    bundled ZN-N reference.
    """
    builder, his, zinc = zinc_histidine_site(donor_pos=(3.90, 0.0, 0.0))
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="long", reported_distance=3.90
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")
    data_json = helpers.write_data_json(tmp_path / "data.json")

    result = analyze(
        analysis_pdb,
        connection_path=source,
        dpi_inputs=helpers.dpi_inputs(
            pdb_path=analysis_pdb,
            mtz_path=str(tmp_path / "absent.mtz"),
            data_json=data_json,
            resolution=1.50,
        ),
    )

    (row,) = result.rows_for("NE2")
    mu, sigma = reference_data.literature_distances()[("HIS", "N", "ZN")]
    assert (row["literature_distance"], row["literature_stdev"]) == (mu, sigma)
    assert math.isfinite(row["dpi"]) and row["dpi"] > 0
    expected = (3.90 - mu) / math.sqrt(row["dpi"] ** 2 + sigma**2)
    assert row["zscore"] == pytest.approx(expected, abs=5e-4)
    assert abs(expected) >= 6 and row["geometry_outlier"] is True
    assert row["coordination_status"] == "declared"

    (candidate,) = [
        entry for entry in result.candidates if entry["neighbor_atom"] == "NE2"
    ]
    assert candidate["first_sphere_eligible"] is False


def test_declaration_overrides_the_typical_donor_table(tmp_path):
    """A declared non-typical donor becomes a bond and says why.

    ARG NH1 can never be geometry-inferred. A declaration promotes it, and the
    row must carry the explicit override plus the interpretive warning instead
    of quietly looking like an ordinary inference.
    """
    builder = StructureBuilder()
    arginine = builder.add_amino_acid(
        "ARG",
        10,
        chain="A",
        positions={"NH1": (2.30, 0.0, 0.0)},
        origin=(20.0, 20.0, 20.0),
    )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_connection(
        zinc.ref("ZN"), arginine.ref("NH1"), name="n1", reported_distance=2.30
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.rows_for("NH1")
    assert row["coordination_status"] == "declared"
    assert row["inferred_donor_allowed"] is False
    assert row["donor_rule_override"] == "declared_connection"
    assert row["context_warning"] is True
    assert "declared_non_typical_donor" in row["context_warning_reasons"].split("|")

    summary = next(iter(result.summaries.values()))
    assert summary["declared_donor_override_contact_count"] == 1
    assert summary["context_warning"] is True


def test_declared_symmetry_contact_uses_the_nearest_image(tmp_path):
    """An ``asu=any`` declaration is measured against the nearest crystal image.

    The donor sits 1.0 A from the metal across a cell edge and 19.0 A away
    inside the asymmetric unit, so the recorded distance identifies which image
    the declaration was resolved against.
    """
    builder = StructureBuilder(
        cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), spacegroup="P 1"
    )
    builder.add_amino_acid(
        "HIS",
        10,
        chain="A",
        positions={"NE2": (10.0, 10.0, 10.0)},
        origin=(8.0, 8.0, 8.0),
    )
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    water = builder.add_water(101, (19.5, 0.5, 0.5), chain="B")
    builder.add_connection(
        zinc.ref("ZN"), water.ref("O"), name="sym", asu="any", reported_distance=1.0
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = [entry for entry in result.rows if entry["neighbor_resname"] == "HOH"]
    assert row["coordination_status"] == "declared"
    assert row["distance"] == pytest.approx(1.0, abs=1e-6)
    assert row["contact_scope"] == "crystallographic"
    assert row["crystallographic_contact"] is True
    assert row["symmetry_operation"] == "1_455"


@pytest.mark.parametrize(
    "donor_x,asu,expected_distance,expected_scope,expected_operation",
    [
        pytest.param(1.5, gemmi.Asu.Same, 1.0, "explicit", "1_555", id="near-same"),
        pytest.param(1.5, gemmi.Asu.Any, 1.0, "explicit", "1_555", id="near-any"),
        pytest.param(
            1.5,
            gemmi.Asu.Different,
            math.sqrt(5.0),
            "crystallographic",
            "2_555",
            id="near-different",
        ),
        pytest.param(
            19.5, gemmi.Asu.Same, 19.0, "explicit", "1_555", id="across-cell-same"
        ),
        pytest.param(
            19.5, gemmi.Asu.Any, 1.0, "crystallographic", "1_455", id="across-cell-any"
        ),
        pytest.param(
            19.5,
            gemmi.Asu.Different,
            1.0,
            "crystallographic",
            "1_455",
            id="across-cell-different",
        ),
    ],
)
def test_declared_geometry_honors_each_asu_constraint(
    tmp_path, donor_x, asu, expected_distance, expected_scope, expected_operation
):
    """Same, Any, and Different select observably different allowed images."""
    builder = StructureBuilder(
        cell=(20.0, 20.0, 20.0, 90.0, 90.0, 90.0), spacegroup="P 1 2 1"
    )
    builder.add_metal("ZN", 1, chain="B", pos=(0.5, 0.5, 0.5))
    builder.add_water(101, (donor_x, 0.5, 0.5), chain="B")
    context = load_structure("test", builder.write_pdb(tmp_path / "asu_geometry.pdb"))
    metal = next(atom for atom in context.contact_atoms if atom.element == "ZN")
    neighbor = next(
        atom for atom in context.contact_atoms if atom.residue_name == "HOH"
    )
    connection = SimpleNamespace(asu=asu)

    geometry = bond_analysis._declared_candidate_geometry(
        context, metal, neighbor, connection
    )

    assert geometry["distance_raw"] == pytest.approx(expected_distance, abs=1e-6)
    assert geometry["contact_scope"] == expected_scope
    assert geometry["symmetry_contact"] is (expected_scope != "explicit")
    assert geometry["crystallographic_contact"] is (expected_scope != "explicit")
    assert geometry["symmetry_operation"] == expected_operation


# --------------------------------------------------------------------------- #
# Declaration metadata carried onto the row
# --------------------------------------------------------------------------- #
def test_mmcif_declaration_carries_its_identity_type_and_link_id(tmp_path):
    """mmCIF preserves the declaration's own id, type and link id."""
    builder, his, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"),
        his.ref("NE2"),
        name="c1",
        type="covale",
        link_id="ZN-NE2",
        reported_distance=2.03,
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.declared_rows
    assert row["connection_id"] == "c1"
    assert row["connection_type"] == "covale"
    assert row["connection_link_id"] == "ZN-NE2"
    assert row["connection_asu"] == "same"
    assert float(row["connection_reported_distance"]) == pytest.approx(2.03)


def test_pdb_link_declaration_is_still_identified_without_an_id_field(tmp_path):
    """A legacy LINK carries no identifier, so the row gets a generated one.

    The record must still be auditable: a non-empty id and the declared status,
    even though the file could not name the connection.
    """
    builder, his, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"),
        his.ref("NE2"),
        name="c1",
        link_id="ZN-NE2",
        reported_distance=2.03,
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "pdb")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.declared_rows
    assert row["connection_id"]
    assert row["connection_link_id"] == ""
    assert float(row["connection_reported_distance"]) == pytest.approx(2.03)


def test_absent_reported_distance_is_blank_not_zero(tmp_path):
    """A LINK that states no distance reports nothing, not a measured zero.

    Deposited ``LINK`` records often stop before the distance columns. ``0.0``
    in that column would read as a coincident pair; the measured distance is
    unaffected either way.
    """
    builder, his, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    written = builder.write_pdb(tmp_path / "written.pdb")
    analysis_pdb = str(tmp_path / "analysis.pdb")
    with open(written, encoding="utf-8") as source_handle:
        lines = source_handle.readlines()
    assert any(line.startswith("LINK") for line in lines)
    with open(analysis_pdb, "w", encoding="utf-8") as handle:
        # Truncate every LINK before its distance columns (73-78).
        handle.writelines(
            line.rstrip("\n")[:72].rstrip() + "\n" if line.startswith("LINK") else line
            for line in lines
        )

    result = analyze(analysis_pdb, connection_path=analysis_pdb)

    (row,) = result.declared_rows
    assert row["connection_reported_distance"] == ""
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)


def test_multiple_declarations_for_one_atom_image_are_all_recorded(tmp_path):
    """Two declarations of the same contact merge into one row keeping both ids."""
    builder, his, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="first", reported_distance=2.03
    )
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="second", reported_distance=2.03
    )
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    (row,) = result.rows_for("NE2")
    assert row["connection_id"].split("|") == ["first", "second"]
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)


def test_connection_path_defaults_to_the_analyzed_coordinates(tmp_path):
    """Omitting ``connection_path`` reads declarations from the analyzed PDB."""
    builder, his, zinc = zinc_histidine_site()
    builder.add_connection(
        zinc.ref("ZN"), his.ref("NE2"), name="m1", reported_distance=2.03
    )
    analysis_pdb = builder.write_pdb(tmp_path / "analysis.pdb")

    result = analyze(analysis_pdb, connection_path=None)

    (row,) = result.declared_rows
    assert row["coordination_source"] == "LINK"
    assert row["distance"] == pytest.approx(2.03, abs=1e-6)


def test_a_file_without_declarations_produces_no_declared_rows(tmp_path):
    """With no ``_struct_conn``/``LINK`` records every contact stays inferred."""
    builder, his, zinc = zinc_histidine_site()
    source, analysis_pdb = write_source_and_analysis(builder, tmp_path, "cif")

    result = analyze(analysis_pdb, connection_path=source)

    assert result.declared_rows == []
    (row,) = result.rows_for("NE2")
    assert row["coordination_status"] == "inferred"
    assert row["coordination_source"] == "proximity_rule"
    assert result.metadata["partial_reason_codes"] == ["missing_dpi_metadata_source"]


# --------------------------------------------------------------------------- #
# Regressions: provenance and audit trail of resolved/unresolved declarations
# --------------------------------------------------------------------------- #
def _structure_with_one_ncs_operation(tmp_path):
    """A Zn and a water whose +10 Angstrom NCS image sits 2.09 A from the Zn.

    The water is 7.91 Angstrom away inside the asymmetric unit, so the only
    contact-range approach to the metal is through strict-NCS operation ``1``.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    water = builder.add_water(101, (-7.91, 0.0, 0.0), chain="W")
    structure = builder.to_gemmi()
    transform = gemmi.Transform()
    transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([10.0, 0.0, 0.0])
    structure.ncs.append(gemmi.NcsOp(transform, "1", False))
    path = str(tmp_path / "ncs.pdb")
    structure.write_pdb(path)
    return path, metal, water


def test_declared_contact_through_an_ncs_image_is_labelled_strict_ncs(tmp_path):
    """A declared contact must get the same provenance as an inferred one.

    ``StructureContext.image_provenance`` is the single source of truth for
    whether a Gemmi image is crystallographic, strict-NCS or both, and the
    proximity search already uses it.

    Regression: the declared path computed its own answer from ``same_asu()``
    and never reported strict NCS, so the same physical contact was labelled
    ``crystallographic`` when declared and ``strict_ncs`` when inferred --
    changing ``contact_scope`` and the per-site
    ``coordination_depends_on_strict_ncs`` summary.

    The connection is built in-process rather than round-tripped through a file
    because Gemmi normalises ``Connection.asu`` back to ``Same`` on write, which
    would never reach the symmetry branch.
    """
    path, _, _ = _structure_with_one_ncs_operation(tmp_path)
    context = load_structure("test", path)
    assert context.strict_ncs_operation_ids == ("1",)

    metal = context.metal_atoms(["ZN"])[0]
    neighbor = next(atom for atom in context.contact_atoms if atom.atom_name == "O")
    connection = SimpleNamespace(asu=gemmi.Asu.Any)

    geometry = bond_analysis._declared_candidate_geometry(
        context, metal, neighbor, connection
    )

    # The proximity path's verdict for the very same image is the oracle.
    crystallographic, strict_ncs, ncs_id, scope = context.image_provenance(
        geometry["symmetry_image_index"], tuple(geometry["translation"])
    )
    assert (strict_ncs, ncs_id, scope) == (True, "1", "strict_ncs")

    assert geometry["strict_ncs_contact"] == strict_ncs
    assert geometry["strict_ncs_operation_id"] == ncs_id
    assert geometry["crystallographic_contact"] == crystallographic
    assert geometry["contact_scope"] == scope


def test_declaration_with_two_unresolved_partners_leaves_an_audit_trace(tmp_path):
    """Failure to map either declared partner must make the result incomplete.

    Regression: metal membership was tested before unresolved partners, so when
    neither source identity existed in the analysis model neither looked like a
    selected metal and the declaration was dropped with no row, reason code or
    message -- an entry whose identifiers Alchemy cannot map reported complete
    success with no declared contacts at all.
    """
    source_builder = StructureBuilder()
    donor = source_builder.add_amino_acid(
        "HIS", 10, chain="A", positions={"NE2": (2.03, 0.0, 0.0)}
    )
    metal = source_builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    source_builder.add_connection(
        metal.ref("ZN"), donor.ref("NE2"), name="unmapped", reported_distance=2.03
    )
    source = source_builder.write_cif(tmp_path / "source.cif")

    # Bond analysis still has a selected metal site, but neither source author
    # identity exists in this deliberately unrelated analysis model.
    analysis_builder = StructureBuilder()
    analysis_builder.add_metal("ZN", 99, chain="Z", pos=(20.0, 20.0, 20.0))
    analysis = analysis_builder.write_pdb(tmp_path / "analysis.pdb")
    context = load_structure("test", analysis)
    rows, candidates, _, metadata = bond_analysis.run_bond_analysis(
        "test",
        analysis,
        [],
        list(helpers.EDSTATS_HEADER),
        helpers.dpi_inputs(),
        structure=context,
        connection_path=source,
    )

    trace = "declared_connection_resolution_incomplete" in metadata[
        "partial_reason_codes"
    ] or any("unmapped" in message for message in metadata["messages"])
    assert trace, (
        "the declaration produced no audit trace when both partners were "
        f"unresolved (rows={rows!r}, candidates={candidates!r})"
    )


def test_declared_donor_outside_the_standard_residues_is_kept_as_evidence(tmp_path):
    """An unscoreable declared donor stays visible instead of disappearing.

    Nucleic acids, modified residues and organic ligands are real metal donors
    and the depositor declared this one with its own distance, but no bundled
    literature reference covers them, so the contact can never be z-scored.

    Regression: the declaration used to hit a bare ``continue``, producing no
    bond row, no candidate row and no reason code -- indistinguishable from a
    metal with no coordination at all, which is why 300d's six Mn ions reported
    zero coordination evidence.

    The contract is deliberately asymmetric: the contact is *reported* with its
    measured distance and provenance, but is *not* promoted to a bond, because
    doing so would raise coordination counts and enlarge the confidence
    geometry-coverage denominator on evidence nothing can assess.
    """
    builder = StructureBuilder()
    metal = builder.add_metal("MN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    nucleotide = builder.add_hetero_residue(
        "DG",
        10,
        [
            AtomSpec("P", "P", (2.20, 1.20, 0.0)),
            AtomSpec("OP1", "O", (2.15, 0.0, 0.0)),
            AtomSpec("OP2", "O", (3.30, 2.00, 0.0)),
        ],
        chain="C",
    )
    builder.add_connection(
        metal.ref("MN"), nucleotide.ref("OP1"), name="metalc1", reported_distance=2.15
    )
    path = builder.write_cif(tmp_path / "nucleotide.cif")

    context = load_structure("test", path)
    rows, candidates, _, metadata = bond_analysis.run_bond_analysis(
        "test",
        path,
        [],
        list(helpers.EDSTATS_HEADER),
        helpers.dpi_inputs(),
        structure=context,
        connection_path=path,
    )

    # Not scored: no bond row, so no coordination count or QG denominator moves.
    assert rows == []

    declared = [row for row in candidates if row["neighbor_atom"] == "OP1"]
    assert len(declared) == 1, candidates
    row = declared[0]
    assert row["neighbor_resname"] == "DG"
    assert row["declared_connection"] is True
    assert row["connection_id"] == "metalc1"
    assert float(row["candidate_distance"]) == pytest.approx(2.15, abs=5e-3)
    assert row["inferred_donor_allowed"] is False
    # The declaration cannot override a donor rule for a class that has no
    # reference at all, so it must not be labelled as having done so.
    assert row["donor_rule_override"] == ""
    assert "declared_donor_outside_supported_classes" in metadata["warning_codes"]
