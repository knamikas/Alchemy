"""Shared test helpers: synthetic structures, EDSTATS tables, capability probes.

Nothing here touches the network, CCP4, or the PDB-REDO mirror unless a probe is
called explicitly.

Side-chain skeleton coordinates are schematic, not a real rotamer: a test that
cares about a distance must place that atom itself through ``positions=``.
"""

from __future__ import annotations

import math
import os
import shutil
import socket
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    cast,
)

import gemmi

if TYPE_CHECKING:
    # Annotations only, so that the deliberate per-function imports below stay
    # the only place this module reaches into ``src`` at run time.
    from output_rows import MetalStatsRow
    from structure_analysis import StructureContext


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")

Vec3 = tuple[float, float, float]

# Callers pass ``tmp_path / "name"`` as readily as a plain string; every writer
# below stringifies its argument.
_StrPath = str | os.PathLike[str]


class _PdbWritable(Protocol):
    """The fully typed positional overload of Gemmi's PDB writer."""

    def write_pdb(self, path: str, options: gemmi.PdbWriteOptions, /) -> None: ...


def write_pdb(structure: gemmi.Structure, path: _StrPath) -> str:
    """Write a structure through Gemmi's concretely typed overload."""
    rendered_path = str(path)
    cast(_PdbWritable, structure).write_pdb(rendered_path, gemmi.PdbWriteOptions())
    return rendered_path


# ``P 21 21 21`` gives four symmetry operations, so symmetry search is available
# and image contacts are exercised.
DEFAULT_CELL: tuple[float, float, float, float, float, float] = (
    60.0,
    70.0,
    80.0,
    90.0,
    90.0,
    90.0,
)
DEFAULT_SPACEGROUP = "P 21 21 21"

# Far enough from the origin that an unmodified residue contributes no contact
# to a metal sitting at ``(0, 0, 0)``.
DEFAULT_RESIDUE_ORIGIN: Vec3 = (20.0, 20.0, 20.0)

# Unit directions spreading donors around a metal, ordered octahedrally.
DONOR_DIRECTIONS: tuple[Vec3, ...] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)

STANDARD_AMINO_ACIDS = frozenset(
    (
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    )
)

WATER_NAMES = frozenset(("HOH", "WAT", "DOD", "H2O"))

_BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O")

_SIDE_CHAIN_ATOMS: dict[str, tuple[str, ...]] = {
    "ALA": ("CB",),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("CB", "CG", "OD1", "ND2"),
    "ASP": ("CB", "CG", "OD1", "OD2"),
    "CYS": ("CB", "SG"),
    "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"),
    "GLY": (),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "MET": ("CB", "CG", "SD", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("CB", "CG", "CD"),
    "SER": ("CB", "OG"),
    "THR": ("CB", "OG1", "CG2"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("CB", "CG1", "CG2"),
}

_BACKBONE_OFFSETS: dict[str, Vec3] = {
    "N": (0.000, 0.000, 0.000),
    "CA": (1.458, 0.000, 0.000),
    "C": (2.009, 1.420, 0.000),
    "O": (1.251, 2.390, 0.000),
    "OXT": (3.245, 1.600, 0.000),
    "OT1": (3.245, 1.600, 0.000),
    "OT2": (3.100, 1.600, 1.200),
}


def element_for_atom_name(atom_name: str) -> str:
    """Element symbol implied by a standard amino-acid heavy-atom name.

    Every heavy atom of the twenty standard residues begins with C, N, O or S,
    so the first character is unambiguous; anything else raises.
    """
    name = str(atom_name).strip().upper()
    if not name:
        raise ValueError("atom name is empty")
    if name[0] in ("C", "N", "O", "S"):
        return name[0]
    raise ValueError(
        f"cannot infer an element for atom name {atom_name!r}; pass an explicit element"
    )


def _side_chain_offset(index: int) -> Vec3:
    """Schematic position of the ``index``-th side-chain atom, from N."""
    return (2.010 + 1.220 * index, -0.773 - 0.500 * index, -1.199 - 0.700 * index)


def _translate(offset: Vec3, origin: Sequence[float]) -> Vec3:
    return (
        float(origin[0]) + offset[0],
        float(origin[1]) + offset[1],
        float(origin[2]) + offset[2],
    )


@dataclass(frozen=True)
class AtomSpec:
    """One deposited atom record."""

    name: str
    element: str
    pos: Vec3
    occupancy: float = 1.0
    altloc: str = ""
    b_iso: float = 20.0
    charge: int = 0

    def moved(self, pos: Vec3) -> AtomSpec:
        return replace(self, pos=(float(pos[0]), float(pos[1]), float(pos[2])))


@dataclass(frozen=True)
class AtomRef:
    """Address of one atom, used to declare a ``struct_conn``/``LINK`` record."""

    chain: str
    resname: str
    seqid: int
    icode: str
    atom_name: str
    altloc: str = ""


@dataclass
class ResidueSpec:
    """One residue and its atoms. Returned by every ``StructureBuilder.add_*``."""

    name: str
    seqid: int
    atoms: list[AtomSpec]
    het: bool = False
    icode: str = ""
    chain: str = ""

    @property
    def resnum(self) -> str:
        """Author residue identifier as Alchemy formats it (number + icode)."""
        return f"{self.seqid}{self.icode}"

    def ref(self, atom_name: str, altloc: str = "") -> AtomRef:
        """Address of ``atom_name`` in this residue, for ``add_connection``."""
        if not any(atom.name == atom_name for atom in self.atoms):
            raise KeyError(
                f"residue {self.name}/{self.chain}/{self.resnum} has no atom "
                f"{atom_name!r}"
            )
        return AtomRef(
            chain=self.chain,
            resname=self.name,
            seqid=self.seqid,
            icode=self.icode,
            atom_name=atom_name,
            altloc=altloc,
        )

    def atom(self, atom_name: str, altloc: str = "") -> AtomSpec:
        for spec in self.atoms:
            if spec.name == atom_name and spec.altloc == altloc:
                return spec
        raise KeyError(
            f"residue {self.name}/{self.chain}/{self.resnum} has no atom "
            f"{atom_name!r} with altloc {altloc!r}"
        )


@dataclass
class ConnectionSpec:
    """A declared metal-ligand connection (mmCIF ``_struct_conn`` / PDB LINK)."""

    partner1: AtomRef
    partner2: AtomRef
    name: str = ""
    type: str = "metalc"
    asu: str = "same"
    reported_distance: float = 0.0
    link_id: str = ""


_CONNECTION_TYPES = {
    "covale": gemmi.ConnectionType.Covale,
    "disulf": gemmi.ConnectionType.Disulf,
    "hydrog": gemmi.ConnectionType.Hydrog,
    "metalc": gemmi.ConnectionType.MetalC,
    "none": gemmi.ConnectionType.Unknown,
    "unknown": gemmi.ConnectionType.Unknown,
}

_ASU_VALUES = {
    "any": gemmi.Asu.Any,
    "same": gemmi.Asu.Same,
    "different": gemmi.Asu.Different,
}


class StructureBuilder:
    """Compose a gemmi structure and write it as PDB and/or mmCIF.

    Residues are emitted in insertion order, except that within a chain every
    non-het residue is written before every het one: gemmi's ``setup_entities``
    classifies a polymer residue following a het residue as non-polymer, which
    would silently disable Alchemy's polymer-terminal donor rules. Pass
    ``group_het=False`` for strict insertion order.
    """

    def __init__(
        self,
        name: str = "TEST",
        cell: Sequence[float] | None = DEFAULT_CELL,
        spacegroup: str | None = DEFAULT_SPACEGROUP,
        resolution: float = 1.50,
        group_het: bool = True,
        setup_entities: bool = True,
    ) -> None:
        self.name = name
        self.cell = tuple(cell) if cell is not None else None
        self.spacegroup = spacegroup
        self.resolution = resolution
        self.group_het = group_het
        self.setup_entities = setup_entities
        self._chain_order: list[str] = []
        self._residues: list[ResidueSpec] = []
        self.connections: list[ConnectionSpec] = []

    def add_residue(self, residue: ResidueSpec, chain: str = "A") -> ResidueSpec:
        """Append an already-built :class:`ResidueSpec` to ``chain``."""
        residue.chain = chain
        if chain not in self._chain_order:
            self._chain_order.append(chain)
        self._residues.append(residue)
        return residue

    @property
    def residues(self) -> tuple[ResidueSpec, ...]:
        return tuple(self._residues)

    def add_amino_acid(
        self,
        resname: str,
        seqid: int,
        *,
        chain: str = "A",
        positions: Mapping[str, Sequence[float]] | None = None,
        origin: Sequence[float] = DEFAULT_RESIDUE_ORIGIN,
        occupancy: float = 1.0,
        altloc: str = "",
        icode: str = "",
        b_iso: float = 20.0,
        atoms: Sequence[AtomSpec] | None = None,
    ) -> ResidueSpec:
        """Add a standard amino-acid residue.

        ``positions`` maps atom name to an absolute ``(x, y, z)``; listed atoms
        override the schematic skeleton and unknown names are appended with an
        element derived from the name. Every other atom is placed relative to
        ``origin``. ``atoms`` replaces the skeleton entirely and is mutually
        exclusive with ``positions``.
        """
        resname = str(resname).upper()
        if atoms is not None and positions:
            raise ValueError("pass either atoms= or positions=, not both")
        if atoms is None:
            atoms = self._amino_acid_skeleton(resname, origin, occupancy, altloc, b_iso)
            atoms = _apply_positions(atoms, positions, occupancy, altloc, b_iso)
        return self.add_residue(
            ResidueSpec(
                name=resname,
                seqid=int(seqid),
                atoms=list(atoms),
                het=False,
                icode=str(icode),
            ),
            chain=chain,
        )

    def add_metal(
        self,
        element: str = "ZN",
        seqid: int = 1,
        *,
        chain: str = "B",
        pos: Sequence[float] = (0.0, 0.0, 0.0),
        occupancy: float = 1.0,
        resname: str | None = None,
        atom_name: str | None = None,
        altloc: str = "",
        icode: str = "",
        b_iso: float = 20.0,
    ) -> ResidueSpec:
        """Add a single-atom hetero metal ion.

        ``resname`` and ``atom_name`` default to the upper-cased element, which
        is what the PDB uses for monatomic ions and what makes
        ``metal_identification`` classify the residue as ``metal``.
        """
        element = str(element).upper()
        resname = element if resname is None else str(resname).upper()
        atom_name = element if atom_name is None else str(atom_name)
        atom = AtomSpec(
            name=atom_name,
            element=element,
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            occupancy=occupancy,
            altloc=altloc,
            b_iso=b_iso,
        )
        return self.add_residue(
            ResidueSpec(
                name=resname, seqid=int(seqid), atoms=[atom], het=True, icode=str(icode)
            ),
            chain=chain,
        )

    def add_water(
        self,
        seqid: int,
        pos: Sequence[float],
        *,
        chain: str = "B",
        occupancy: float = 1.0,
        altloc: str = "",
        icode: str = "",
        resname: str = "HOH",
        b_iso: float = 20.0,
    ) -> ResidueSpec:
        """Add a single-oxygen water residue at ``pos``."""
        atom = AtomSpec(
            name="O",
            element="O",
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            occupancy=occupancy,
            altloc=altloc,
            b_iso=b_iso,
        )
        return self.add_residue(
            ResidueSpec(
                name=str(resname).upper(),
                seqid=int(seqid),
                atoms=[atom],
                het=True,
                icode=str(icode),
            ),
            chain=chain,
        )

    def add_hetero_residue(
        self,
        resname: str,
        seqid: int,
        atoms: Sequence[AtomSpec],
        *,
        chain: str = "B",
        icode: str = "",
    ) -> ResidueSpec:
        """Add an arbitrary hetero component, e.g. a multi-metal cofactor."""
        return self.add_residue(
            ResidueSpec(
                name=str(resname).upper(),
                seqid=int(seqid),
                atoms=list(atoms),
                het=True,
                icode=str(icode),
            ),
            chain=chain,
        )

    def add_conformers(
        self,
        residue: ResidueSpec,
        conformers: Sequence[tuple[str, float, Mapping[str, Sequence[float]]]],
        *,
        atom_names: Iterable[str] | None = None,
    ) -> ResidueSpec:
        """Split atoms of ``residue`` into alternate conformers.

        ``conformers`` is a sequence of ``(altloc, occupancy, positions)``, and
        ``positions`` may be empty. ``atom_names`` restricts the split to a
        subset, leaving the rest as blank-altloc records shared by every
        conformer, which is how deposited files model a partial side-chain
        alternative.
        """
        if not conformers:
            raise ValueError("at least one conformer is required")
        selected = (
            set(atom_names)
            if atom_names is not None
            else {atom.name for atom in residue.atoms if not atom.altloc}
        )
        kept = [atom for atom in residue.atoms if atom.name not in selected]
        originals = [atom for atom in residue.atoms if atom.name in selected]
        expanded: list[AtomSpec] = []
        for altloc, occupancy, positions in conformers:
            for atom in originals:
                pos = (positions or {}).get(atom.name, atom.pos)
                expanded.append(
                    replace(
                        atom,
                        altloc=str(altloc),
                        occupancy=float(occupancy),
                        pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                    )
                )
        residue.atoms = kept + expanded
        return residue

    def add_connection(
        self,
        partner1: AtomRef,
        partner2: AtomRef,
        *,
        name: str = "",
        type: str = "metalc",
        asu: str = "same",
        reported_distance: float = 0.0,
        link_id: str = "",
    ) -> ConnectionSpec:
        """Declare a connection between two atoms.

        Written as ``_struct_conn`` in mmCIF and as a ``LINK`` record in PDB.
        ``asu`` is one of ``same``, ``any`` or ``different``.
        """
        if str(type).lower() not in _CONNECTION_TYPES:
            raise ValueError(f"unknown connection type {type!r}")
        if str(asu).lower() not in _ASU_VALUES:
            raise ValueError(f"unknown asu {asu!r}")
        spec = ConnectionSpec(
            partner1=partner1,
            partner2=partner2,
            name=name or f"conn{len(self.connections) + 1}",
            type=str(type).lower(),
            asu=str(asu).lower(),
            reported_distance=float(reported_distance),
            link_id=link_id,
        )
        self.connections.append(spec)
        return spec

    def _ordered_residues(self, chain: str) -> list[ResidueSpec]:
        residues = [r for r in self._residues if r.chain == chain]
        if not self.group_het:
            return residues
        return [r for r in residues if not r.het] + [r for r in residues if r.het]

    def to_gemmi(self) -> gemmi.Structure:
        """Materialize a fresh gemmi structure."""
        structure = gemmi.Structure()
        structure.name = self.name
        if self.cell is not None:
            structure.cell = gemmi.UnitCell(*self.cell)
        if self.spacegroup is not None:
            structure.spacegroup_hm = self.spacegroup
        if self.resolution:
            structure.resolution = float(self.resolution)
        model = gemmi.Model(1)
        for chain_name in self._chain_order:
            chain = gemmi.Chain(chain_name)
            for residue in self._ordered_residues(chain_name):
                chain.add_residue(_gemmi_residue(residue))
            model.add_chain(chain)
        structure.add_model(model)
        if self.setup_entities:
            structure.setup_entities()
        for spec in self.connections:
            structure.connections.append(_gemmi_connection(spec))
        return structure

    def write_pdb(self, path: _StrPath) -> str:
        """Write legacy PDB (ATOM/HETATM + LINK); returns the path."""
        return write_pdb(self.to_gemmi(), path)

    def write_cif(self, path: _StrPath) -> str:
        """Write mmCIF (``_atom_site`` + ``_struct_conn``); returns the path."""
        path = str(path)
        structure = self.to_gemmi()
        structure.make_mmcif_document().write_file(path)
        return path

    def write(self, path: _StrPath, fmt: str | None = None) -> str:
        """Write PDB or mmCIF, choosing the format from ``path``'s suffix."""
        path = str(path)
        if fmt is None:
            fmt = (
                "cif"
                if os.path.splitext(path)[1].lower() in (".cif", ".mmcif")
                else "pdb"
            )
        if fmt == "cif":
            return self.write_cif(path)
        if fmt == "pdb":
            return self.write_pdb(path)
        raise ValueError(f"unknown coordinate format {fmt!r}")

    def _amino_acid_skeleton(
        self,
        resname: str,
        origin: Sequence[float],
        occupancy: float,
        altloc: str,
        b_iso: float,
    ) -> list[AtomSpec]:
        if resname not in _SIDE_CHAIN_ATOMS:
            raise ValueError(
                f"{resname!r} has no bundled skeleton; pass atoms= explicitly"
            )
        specs = [
            AtomSpec(
                name=name,
                element=element_for_atom_name(name),
                pos=_translate(_BACKBONE_OFFSETS[name], origin),
                occupancy=occupancy,
                altloc=altloc,
                b_iso=b_iso,
            )
            for name in _BACKBONE_ATOMS
        ]
        for index, name in enumerate(_SIDE_CHAIN_ATOMS[resname]):
            specs.append(
                AtomSpec(
                    name=name,
                    element=element_for_atom_name(name),
                    pos=_translate(_side_chain_offset(index), origin),
                    occupancy=occupancy,
                    altloc=altloc,
                    b_iso=b_iso,
                )
            )
        return specs


def _apply_positions(
    atoms: Sequence[AtomSpec],
    positions: Mapping[str, Sequence[float]] | None,
    occupancy: float,
    altloc: str,
    b_iso: float,
) -> list[AtomSpec]:
    result = list(atoms)
    for name, pos in (positions or {}).items():
        pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        for index, atom in enumerate(result):
            if atom.name == name:
                result[index] = atom.moved(pos)
                break
        else:
            result.append(
                AtomSpec(
                    name=name,
                    element=element_for_atom_name(name),
                    pos=pos,
                    occupancy=occupancy,
                    altloc=altloc,
                    b_iso=b_iso,
                )
            )
    return result


def _gemmi_residue(spec: ResidueSpec) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = spec.name
    residue.seqid = gemmi.SeqId(int(spec.seqid), (spec.icode or " ")[0])
    residue.het_flag = "H" if spec.het else "A"
    for atom_spec in spec.atoms:
        atom = gemmi.Atom()
        atom.name = atom_spec.name
        atom.element = gemmi.Element(atom_spec.element)
        atom.pos = gemmi.Position(*atom_spec.pos)
        atom.occ = float(atom_spec.occupancy)
        atom.b_iso = float(atom_spec.b_iso)
        atom.charge = int(atom_spec.charge)
        if atom_spec.altloc:
            atom.altloc = atom_spec.altloc
        residue.add_atom(atom)
    return residue


def _gemmi_connection(spec: ConnectionSpec) -> gemmi.Connection:
    connection = gemmi.Connection()
    connection.name = spec.name
    connection.type = _CONNECTION_TYPES[spec.type]
    connection.asu = _ASU_VALUES[spec.asu]
    connection.link_id = spec.link_id
    connection.reported_distance = float(spec.reported_distance)
    connection.partner1 = _gemmi_address(spec.partner1)
    connection.partner2 = _gemmi_address(spec.partner2)
    return connection


def _gemmi_address(ref: AtomRef) -> gemmi.AtomAddress:
    return gemmi.AtomAddress(
        ref.chain,
        gemmi.SeqId(int(ref.seqid), (ref.icode or " ")[0]),
        ref.resname,
        ref.atom_name,
        ref.altloc or "*",
    )


def simple_metal_site(
    metal: str = "ZN",
    donors: Sequence[tuple[str, str, float]] = (
        ("HIS", "NE2", 2.03),
        ("ASP", "OD1", 1.99),
        ("HOH", "O", 2.09),
    ),
    *,
    metal_pos: Vec3 = (0.0, 0.0, 0.0),
    metal_occupancy: float = 1.0,
    protein_chain: str = "A",
    hetero_chain: str = "B",
    first_seqid: int = 10,
    directions: Sequence[Vec3] = DONOR_DIRECTIONS,
    **builder_kwargs: Any,
) -> StructureBuilder:
    """One metal ion surrounded by donors at exactly the requested distances.

    ``donors`` is a sequence of ``(resname, atom_name, distance)``. Each donor
    gets its own residue, numbered from ``first_seqid``, with its named atom
    ``distance`` Angstrom from the metal along a distinct direction, so
    donor-donor separations stay large and no other atom of the residue comes
    near the metal.
    """
    if len(donors) > len(directions):
        raise ValueError(f"{len(donors)} donors need at least as many directions")
    builder = StructureBuilder(**builder_kwargs)
    builder.add_metal(
        metal, seqid=1, chain=hetero_chain, pos=metal_pos, occupancy=metal_occupancy
    )
    seqid = int(first_seqid)
    water_seqid = 100
    for index, (resname, atom_name, distance) in enumerate(donors):
        direction = directions[index]
        norm = math.sqrt(sum(value * value for value in direction))
        pos = tuple(
            metal_pos[axis] + direction[axis] / norm * float(distance)
            for axis in range(3)
        )
        if resname.upper() in WATER_NAMES:
            builder.add_water(
                water_seqid, pos, chain=hetero_chain, resname=resname.upper()
            )
            water_seqid += 1
        else:
            builder.add_amino_acid(
                resname,
                seqid,
                chain=protein_chain,
                positions={atom_name: pos},
                origin=(20.0 + 12.0 * index, 20.0, 20.0),
            )
            seqid += 1
    return builder


def write_data_json(
    path: _StrPath,
    *,
    nrefcnt: object = 50000,
    rffin: object = 0.20,
    properties: Mapping[str, object] | None = None,
) -> str:
    """Write a PDB-REDO-style ``data.json`` for the DPI calculation.

    ``NREFCNT`` is the reflection count and ``RFFIN`` the final R-free. Both are
    typed ``object`` so a test can pass ``None`` to omit the key or a
    non-numeric value to reach the malformed-metadata branches.
    """
    import json

    props: dict[str, object] = {}
    if nrefcnt is not None:
        props["NREFCNT"] = nrefcnt
    if rffin is not None:
        props["RFFIN"] = rffin
    props.update(properties or {})
    path = str(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"properties": props}, handle)
    return path


def dpi_inputs(
    pdb_path: _StrPath | None = None,
    mtz_path: _StrPath | None = None,
    data_json: _StrPath | None = None,
    resolution: float = 1.50,
) -> dict[str, object]:
    """Build the ``dpi_inputs`` mapping ``run_bond_analysis`` expects.

    With no ``data_json`` the DPI is unavailable by construction and
    ``run_bond_analysis`` reports ``missing_dpi_metadata_source``.
    """
    return {
        "resolution": resolution,
        "data_json": None if data_json is None else str(data_json),
        "pdb_path": None if pdb_path is None else str(pdb_path),
        "mtz_path": None if mtz_path is None else str(mtz_path),
    }


# Literal EDSTATS 1.0.9 ``stats.out`` residue-table header, transcribed rather
# than imported from ``metal_identification`` so that a production reorder
# cannot update both the generated input and its own oracle at once.
EDSTATS_HEADER: tuple[str, ...] = (
    "RT",
    "CI",
    "RN",
    "BAm",
    "NPm",
    "Rm",
    "RGm",
    "SRGm",
    "CCSm",
    "CCPm",
    "ZCCPm",
    "ZOm",
    "ZDm",
    "ZD-m",
    "ZD+m",
    "BAs",
    "NPs",
    "Rs",
    "RGs",
    "SRGs",
    "CCSs",
    "CCPs",
    "ZCCPs",
    "ZOs",
    "ZDs",
    "ZD-s",
    "ZD+s",
    "BAa",
    "NPa",
    "Ra",
    "RGa",
    "SRGa",
    "CCSa",
    "CCPa",
    "ZCCPa",
    "ZOa",
    "ZDa",
    "ZD-a",
    "ZD+a",
    "MN",
    "CP",
    "NR",
)
EDSTATS_METRICS: tuple[str, ...] = EDSTATS_HEADER[3:39]
EDSTATS_NULL = "n/a"


def edstats_separator_row(model: int = 1, row_number: int = 1) -> list[str]:
    """The row EDSTATS emits between MODEL/ENDMDL blocks: 39 fields, not 42."""
    return [
        "_",
        *([EDSTATS_NULL] * len(EDSTATS_METRICS)),
        str(int(model)),
        str(int(row_number)),
    ]


def edstats_row(
    rt: str,
    ci: str,
    rn: object,
    *,
    mn: int = 1,
    cp: str | None = None,
    nr: int = 1,
    metrics: Mapping[str, object] | None = None,
    default: object = 0.0,
    omit_cp: bool = False,
) -> list[str]:
    """Build one 42-field EDSTATS residue row as a list of strings.

    ``rt``/``ci``/``rn`` are the residue name, EDSTATS chain group and residue
    number; ``cp`` is the deposited chain and defaults to ``ci``. ``metrics``
    overrides metric columns by name, and every other metric gets ``default``.

    ``omit_cp=True`` produces EDSTATS' blank-chain form, in which the trailing
    CP field is absent and the row has 41 fields. Use it with ``ci="_"`` (or
    ``ci="0"`` for waters), which is what EDSTATS emits for a residue deposited
    without a chain identifier.
    """
    unknown = set(metrics or {}) - set(EDSTATS_METRICS)
    if unknown:
        raise KeyError(f"not EDSTATS metric columns: {sorted(unknown)}")
    values = dict.fromkeys(EDSTATS_METRICS, default)
    values.update(metrics or {})
    fields = [str(rt), str(ci), str(rn)]
    fields.extend(str(values[name]) for name in EDSTATS_METRICS)
    fields.append(str(int(mn)))
    if not omit_cp:
        fields.append(str(ci if cp is None else cp))
    fields.append(str(int(nr)))
    return fields


def edstats_text(
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str] = EDSTATS_HEADER,
    trailing_newline: bool = True,
) -> str:
    """Render a header line plus rows as the whitespace table EDSTATS writes."""
    lines = [" ".join(str(name) for name in header)]
    lines.extend(" ".join(str(field) for field in row) for row in rows)
    return "\n".join(lines) + ("\n" if trailing_newline else "")


def write_edstats(path: _StrPath, rows: Sequence[Sequence[str]], **kwargs: Any) -> str:
    """Write an EDSTATS ``stats.out`` file and return its path."""
    path = str(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(edstats_text(rows, **kwargs))
    return path


def edstats_rows_for_structure(
    context: StructureContext,
    *,
    metrics: Mapping[str, object] | None = None,
    per_residue: Mapping[tuple[str, str, str], Mapping[str, object]] | None = None,
    default: object = 0.0,
    model: int = 1,
    blank_chain_form: bool = False,
) -> list[list[str]]:
    """One EDSTATS row per residue of a loaded :class:`StructureContext`.

    Rows use each residue's coordinate name, chain and author residue number,
    so the completeness check in ``extract_metal_statistics`` always passes.
    ``metrics`` applies to every row; ``per_residue`` is keyed by
    ``(coordinate_residue_name, chain_id, resnum)`` and overrides on top.

    ``NR`` restarts at 1 for each deposited chain, which is how EDSTATS numbers
    it. Numbering it across the whole model instead would make every row of a
    multi-chain structure unrepresentative of real output.
    """
    rows: list[list[str]] = []
    chain_ordinals: dict[str, int] = {}
    for residue in context.residues:
        rt, chain, resnum = residue.coordinate_author_key
        chain_ordinals[chain] = chain_ordinals.get(chain, 0) + 1
        row_metrics = dict(metrics or {})
        row_metrics.update((per_residue or {}).get((rt, chain, resnum), {}))
        blank = blank_chain_form and not chain
        ci = chain
        if blank:
            ci = "0" if residue.is_water else "_"
        rows.append(
            edstats_row(
                rt,
                ci,
                resnum,
                mn=model,
                cp=chain,
                nr=chain_ordinals[chain],
                metrics=row_metrics,
                default=default,
                omit_cp=blank,
            )
        )
    return rows


def write_edstats_for_structure(
    path: _StrPath, context: StructureContext, **kwargs: Any
) -> str:
    """Write a complete synthetic ``stats.out`` for a loaded structure."""
    return write_edstats(path, edstats_rows_for_structure(context, **kwargs))


def stats_rows_for_structure(
    context: StructureContext,
    path: _StrPath,
    *,
    pdb_id: str | None = None,
    metals: Iterable[str] | None = None,
    cofactors: Iterable[str] | None = None,
    **kwargs: Any,
) -> tuple[list[MetalStatsRow], list[str], str]:
    """Write a synthetic ``stats.out``, then parse it back with the real code.

    Returns ``(rows, header, stats_path)``, ready to hand to
    ``coordination.analysis.run_bond_analysis`` as its sigma join input.
    """
    from metal_elements import METAL_ELEMENTS
    from metal_identification import extract_metal_statistics

    stats_path = write_edstats_for_structure(path, context, **kwargs)
    rows, header = extract_metal_statistics(
        pdb_id if pdb_id is not None else context.pdb_id,
        stats_path,
        set(METAL_ELEMENTS) if metals is None else set(metals),
        set() if cofactors is None else set(cofactors),
        structure=context,
    )
    return rows, header, stats_path


CCP4_TOOLS = ("mtzfix", "fft", "mapmask", "edstats")
_NETWORK_CACHE: dict[tuple[str, int], bool] = {}


def ccp4_env() -> dict[str, str] | None:
    """An environment with the CCP4 tools on PATH, or ``None``."""
    from ccp4_setup import ccp4_tools_available, find_ccp4_setup, resolve_env

    if os.environ.get("ALCHEMY_TESTS_NO_CCP4"):
        return None
    env = os.environ.copy()
    if ccp4_tools_available(env):
        return env
    try:
        setup = find_ccp4_setup(env=env)
    except Exception:
        return None
    if not setup:
        return None
    try:
        env = resolve_env(setup)
    except Exception:  # a probe must never fail the run
        return None
    return env if ccp4_tools_available(env) else None


def ccp4_available() -> bool:
    """Whether all four required CCP4 programs can be resolved."""
    return ccp4_env() is not None


def network_available(
    host: str = "pdb-redo.eu", port: int = 443, timeout: float = 5.0
) -> bool:
    """Whether a TCP connection to ``host:port`` succeeds. Memoized per process.

    Set ``ALCHEMY_TESTS_NO_NETWORK=1`` to force this to ``False``.
    """
    if os.environ.get("ALCHEMY_TESTS_NO_NETWORK"):
        return False
    key = (host, int(port))
    if key not in _NETWORK_CACHE:
        try:
            with socket.create_connection(key, timeout=timeout):
                _NETWORK_CACHE[key] = True
        except OSError:
            _NETWORK_CACHE[key] = False
    return _NETWORK_CACHE[key]


def which(program: str, env: Mapping[str, str] | None = None) -> str | None:
    """``shutil.which`` against an explicit environment's PATH."""
    return shutil.which(program, path=(env or os.environ).get("PATH"))


# Warm PDB-REDO download cache variables, canonical name first.
CACHE_ENV_VARS = ("ALCHEMY_TESTS_CACHE", "ALCHEMY_TEST_CACHE")


def cache_dir_from_env(env: Mapping[str, str] | None = None) -> str | None:
    """The configured PDB-REDO cache directory, or ``None`` if unset.

    Blank and whitespace-only values count as unset: ``ALCHEMY_TESTS_CACHE=``
    in a wrapper script must not resolve the cache to the current directory.
    """
    source = os.environ if env is None else env
    for variable in CACHE_ENV_VARS:
        value = (source.get(variable) or "").strip()
        if value:
            return value
    return None


__all__ = [
    "CACHE_ENV_VARS",
    "cache_dir_from_env",
    "AtomRef",
    "AtomSpec",
    "ConnectionSpec",
    "ResidueSpec",
    "StructureBuilder",
    "CCP4_TOOLS",
    "DEFAULT_CELL",
    "DEFAULT_RESIDUE_ORIGIN",
    "DEFAULT_SPACEGROUP",
    "DONOR_DIRECTIONS",
    "EDSTATS_HEADER",
    "EDSTATS_METRICS",
    "EDSTATS_NULL",
    "REPO_ROOT",
    "SRC_DIR",
    "STANDARD_AMINO_ACIDS",
    "TESTS_DIR",
    "WATER_NAMES",
    "ccp4_available",
    "ccp4_env",
    "dpi_inputs",
    "edstats_row",
    "edstats_rows_for_structure",
    "edstats_separator_row",
    "edstats_text",
    "element_for_atom_name",
    "network_available",
    "simple_metal_site",
    "stats_rows_for_structure",
    "which",
    "write_data_json",
    "write_edstats",
    "write_edstats_for_structure",
]
