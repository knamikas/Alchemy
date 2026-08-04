"""Shared test helpers: synthetic structures, EDSTATS tables, capability probes.

Nothing here touches the network, CCP4, or the PDB-REDO mirror unless a probe is
called explicitly. Everything is built in memory with gemmi and written to a
caller-supplied temporary path.

The three things most tests need:

* :class:`StructureBuilder` -- compose a coordinate model out of amino acids,
  metals and waters, add alternate conformers and ``struct_conn``/``LINK``
  declarations, then write it as legacy PDB or mmCIF.
* :func:`simple_metal_site` -- a one-liner builder for "one metal surrounded by
  donors at chosen distances", which is what most bond-analysis tests want.
* :func:`write_edstats_for_structure` -- a synthetic EDSTATS ``stats.out`` whose
  rows are guaranteed to satisfy the completeness check in
  ``metal_identification.extract_metal_statistics``.

Geometry note: side-chain skeleton coordinates are *schematic*. They are chosen
so that atoms are distinct, elements are unambiguous and bond lengths are
plausible, not so that they reproduce a real rotamer. Tests that care about a
distance must place that atom explicitly through ``positions=``.
"""

from __future__ import annotations

import math
import os
import shutil
import socket
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gemmi


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
# ``src`` is placed on sys.path by conftest.py, which pytest always imports
# before this module. The standalone catalog builder in tools/ runs without
# pytest and keeps its own bootstrap; a single shared insertion point is not
# possible for both.

Vec3 = Tuple[float, float, float]

#: Default crystal metadata. ``P 21 21 21`` gives four symmetry operations, so
#: ``StructureContext.symmetry_search_available`` is true and image searches and
#: ``crystallographic_operation_count`` are exercised.
DEFAULT_CELL: Tuple[float, float, float, float, float, float] = (
    60.0,
    70.0,
    80.0,
    90.0,
    90.0,
    90.0,
)
DEFAULT_SPACEGROUP = "P 21 21 21"

#: Where a residue's schematic skeleton is placed when the caller does not say.
#: Far enough from the origin that an unmodified residue contributes no contact
#: to a metal sitting at ``(0, 0, 0)``.
DEFAULT_RESIDUE_ORIGIN: Vec3 = (20.0, 20.0, 20.0)

#: Unit directions used by :func:`simple_metal_site` to spread donors around a
#: metal, ordered so the first six are octahedral.
DONOR_DIRECTIONS: Tuple[Vec3, ...] = (
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

_BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")

_SIDE_CHAIN_ATOMS: Dict[str, Tuple[str, ...]] = {
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

# Schematic local coordinates for the four backbone atoms, in Angstrom.
_BACKBONE_OFFSETS: Dict[str, Vec3] = {
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
    so the first character is unambiguous. Anything else raises, because a
    silently wrong element would produce a structure the pipeline misreads.
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


# Specs
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

    def moved(self, pos: Vec3) -> "AtomSpec":
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
    atoms: List[AtomSpec]
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


# Builder
class StructureBuilder:
    """Compose a gemmi structure and write it as PDB and/or mmCIF.

    Residues are emitted in the order they were added, except that within one
    chain every non-het residue is written before every het residue. gemmi's
    ``setup_entities`` classifies a polymer residue that follows a het residue
    as non-polymer, which would silently disable Alchemy's polymer-terminal
    donor rules; grouping avoids that. Pass ``group_het=False`` to keep strict
    insertion order when a test needs it.
    """

    def __init__(
        self,
        name: str = "TEST",
        cell: Optional[Sequence[float]] = DEFAULT_CELL,
        spacegroup: Optional[str] = DEFAULT_SPACEGROUP,
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
        self._chain_order: List[str] = []
        self._residues: List[ResidueSpec] = []
        self.connections: List[ConnectionSpec] = []

    # -- low level ---------------------------------------------------------- #
    def add_residue(self, residue: ResidueSpec, chain: str = "A") -> ResidueSpec:
        """Append an already-built :class:`ResidueSpec` to ``chain``."""
        residue.chain = chain
        if chain not in self._chain_order:
            self._chain_order.append(chain)
        self._residues.append(residue)
        return residue

    @property
    def residues(self) -> Tuple[ResidueSpec, ...]:
        return tuple(self._residues)

    # -- residue constructors ----------------------------------------------- #
    def add_amino_acid(
        self,
        resname: str,
        seqid: int,
        *,
        chain: str = "A",
        positions: Optional[Mapping[str, Sequence[float]]] = None,
        origin: Sequence[float] = DEFAULT_RESIDUE_ORIGIN,
        occupancy: float = 1.0,
        altloc: str = "",
        icode: str = "",
        b_iso: float = 20.0,
        atoms: Optional[Sequence[AtomSpec]] = None,
    ) -> ResidueSpec:
        """Add a standard amino-acid residue.

        ``positions`` maps atom name to an absolute ``(x, y, z)``; listed atoms
        override the schematic skeleton and unknown names are appended with an
        element derived from the name (so ``{"OXT": ...}`` adds a C-terminal
        oxygen). Every other atom is placed relative to ``origin``, which
        defaults far enough away that it contributes no metal contact.

        ``atoms`` replaces the skeleton entirely and is mutually exclusive with
        ``positions``.
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
        resname: Optional[str] = None,
        atom_name: Optional[str] = None,
        altloc: str = "",
        icode: str = "",
        b_iso: float = 20.0,
    ) -> ResidueSpec:
        """Add a single-atom hetero metal ion.

        ``resname`` and ``atom_name`` both default to the upper-cased element,
        which is what the PDB uses for monatomic ions and what makes
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

    # -- conformers --------------------------------------------------------- #
    def add_conformers(
        self,
        residue: ResidueSpec,
        conformers: Sequence[Tuple[str, float, Mapping[str, Sequence[float]]]],
        *,
        atom_names: Optional[Iterable[str]] = None,
    ) -> ResidueSpec:
        """Split atoms of ``residue`` into alternate conformers.

        ``conformers`` is a sequence of ``(altloc, occupancy, positions)``.
        Every named atom is replicated once per conformer with that altloc label
        and occupancy; ``positions`` (may be empty) overrides coordinates for
        that conformer only. ``atom_names`` restricts the split to a subset,
        leaving the rest as blank-altloc records shared by both conformers,
        which is how deposited files usually model a partial side-chain
        alternative.

        Alchemy selects the conformer with the highest mean occupancy and
        breaks ties by the alphabetically first label.
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
        expanded: List[AtomSpec] = []
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

    # -- connections -------------------------------------------------------- #
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

        Written as ``_struct_conn`` in mmCIF and as a ``LINK`` record in PDB, so
        the same builder exercises both branches of
        ``declared_connections._collect_declared_candidates``. ``asu`` is one of
        ``same``, ``any`` or ``different``.
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

    # -- output ------------------------------------------------------------- #
    def _ordered_residues(self, chain: str) -> List[ResidueSpec]:
        residues = [r for r in self._residues if r.chain == chain]
        if not self.group_het:
            return residues
        return [r for r in residues if not r.het] + [r for r in residues if r.het]

    def to_gemmi(self) -> gemmi.Structure:
        """Materialize the gemmi structure. Called fresh by every writer."""
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

    def write_pdb(self, path) -> str:
        """Write legacy PDB (ATOM/HETATM + LINK) and return the path as ``str``."""
        path = str(path)
        self.to_gemmi().write_pdb(path)
        return path

    def write_cif(self, path) -> str:
        """Write mmCIF (``_atom_site`` + ``_struct_conn``); returns the path."""
        path = str(path)
        structure = self.to_gemmi()
        structure.make_mmcif_document().write_file(path)
        return path

    def write(self, path, fmt: Optional[str] = None) -> str:
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

    # -- convenience -------------------------------------------------------- #
    def _amino_acid_skeleton(
        self,
        resname: str,
        origin: Sequence[float],
        occupancy: float,
        altloc: str,
        b_iso: float,
    ) -> List[AtomSpec]:
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
    positions: Optional[Mapping[str, Sequence[float]]],
    occupancy: float,
    altloc: str,
    b_iso: float,
) -> List[AtomSpec]:
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


# High-level structure recipes
def simple_metal_site(
    metal: str = "ZN",
    donors: Sequence[Tuple[str, str, float]] = (
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
    **builder_kwargs,
) -> StructureBuilder:
    """One metal ion surrounded by donors at exactly the requested distances.

    ``donors`` is a sequence of ``(resname, atom_name, distance)``. Each donor
    gets its own residue; its named atom is placed ``distance`` Angstrom from
    the metal along a distinct direction from ``directions`` (octahedral by
    default), so donor-donor separations stay large and no other atom of the
    residue comes near the metal. Water names produce water residues, standard
    amino-acid names produce amino-acid residues.

    Residues are numbered from ``first_seqid`` upward in the order given.
    Returns the builder so the caller can add conformers or connections before
    writing.

    >>> builder = simple_metal_site("ZN", [("HIS", "NE2", 2.03)])
    >>> path = builder.write_pdb(tmp_path / "zn.pdb")           # doctest: +SKIP
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
    path,
    *,
    nrefcnt: object = 50000,
    rffin: object = 0.20,
    properties: Optional[Mapping[str, object]] = None,
) -> str:
    """Write a PDB-REDO-style ``data.json`` for the DPI calculation.

    ``NREFCNT`` is the reflection count and ``RFFIN`` the final R-free; pass
    ``None`` for either to omit it and exercise the missing-input branches of
    ``dpi._calculate_dpi_details``.

    Both are typed ``object`` deliberately: callers also pass non-numeric values
    such as ``"many"`` to exercise the malformed-metadata branches, which is the
    point of those tests rather than a type error.
    """
    import json

    props: Dict[str, object] = {}
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
    pdb_path=None, mtz_path=None, data_json=None, resolution: float = 1.50
) -> Dict[str, object]:
    """Build the ``dpi_inputs`` mapping ``run_bond_analysis`` expects.

    With no ``data_json`` the DPI is unavailable by construction and
    ``run_bond_analysis`` reports ``missing_dpi_metadata_source``; measured
    distances are still emitted.
    """
    return {
        "resolution": resolution,
        "data_json": None if data_json is None else str(data_json),
        "pdb_path": None if pdb_path is None else str(pdb_path),
        "mtz_path": None if mtz_path is None else str(mtz_path),
    }


#: Literal EDSTATS 1.0.9 ``stats.out`` residue-table header, intentionally
#: independent of ``metal_identification``. Keeping the fixture oracle separate
#: from the production constant means an accidental reorder in production cannot
#: update both the generated test input and its expected schema at once.
EDSTATS_HEADER: Tuple[str, ...] = (
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
EDSTATS_METRICS: Tuple[str, ...] = EDSTATS_HEADER[3:39]
EDSTATS_NULL = "n/a"


#: The synthetic row EDSTATS emits between MODEL/ENDMDL blocks: ``_`` followed
#: by 36 null metrics and the model and row numbers (39 whitespace fields).
def edstats_separator_row(model: int = 1, row_number: int = 1) -> List[str]:
    """Return EDSTATS' model-separator row as a field list."""
    return [
        "_",
        *([EDSTATS_NULL] * len(EDSTATS_METRICS)),
        str(int(model)),
        str(int(row_number)),
    ]


def edstats_row(
    rt: str,
    ci: str,
    rn,
    *,
    mn: int = 1,
    cp: Optional[str] = None,
    nr: int = 1,
    metrics: Optional[Mapping[str, object]] = None,
    default: object = 0.0,
    omit_cp: bool = False,
) -> List[str]:
    """Build one 42-field EDSTATS residue row as a list of strings.

    ``rt``/``ci``/``rn`` are the residue name, EDSTATS chain group and residue
    number. ``cp`` is the deposited chain and defaults to ``ci``. ``metrics``
    overrides individual metric columns by name (``"ZDm"``, ``"CCPa"``, ...);
    every other metric gets ``default``, which may be ``"n/a"``.

    ``omit_cp=True`` produces EDSTATS' legitimate blank-chain form: the trailing
    CP field is absent, so the row has only 41 whitespace fields and
    ``metal_identification`` has to restore it. Use it together with
    ``ci="_"`` (or ``ci="0"`` for waters), which is what EDSTATS emits for a
    residue deposited without a chain identifier.

    >>> row = edstats_row("ZN", "B", 1, metrics={"ZDm": 1.25})
    >>> len(row)
    42
    """
    unknown = set(metrics or {}) - set(EDSTATS_METRICS)
    if unknown:
        raise KeyError(f"not EDSTATS metric columns: {sorted(unknown)}")
    values = {name: default for name in EDSTATS_METRICS}
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


def write_edstats(path, rows: Sequence[Sequence[str]], **kwargs) -> str:
    """Write an EDSTATS ``stats.out`` file and return its path."""
    path = str(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(edstats_text(rows, **kwargs))
    return path


def edstats_rows_for_structure(
    context,
    *,
    metrics: Optional[Mapping[str, object]] = None,
    per_residue: Optional[Mapping[Tuple[str, str, str], Mapping[str, object]]] = None,
    default: object = 0.0,
    model: int = 1,
    blank_chain_form: bool = False,
) -> List[List[str]]:
    """One EDSTATS row per residue of a loaded :class:`StructureContext`.

    Rows use each residue's *coordinate* name, chain and author residue number,
    so the completeness check in ``extract_metal_statistics`` -- which demands a
    row for every metal and cofactor residue -- always passes.

    ``metrics`` applies to every row; ``per_residue`` is keyed by
    ``(coordinate_residue_name, chain_id, resnum)`` and overrides on top.
    ``blank_chain_form=True`` emits the 41-field no-CP shape for residues whose
    chain identifier is empty.
    """
    rows: List[List[str]] = []
    for index, residue in enumerate(context.residues, start=1):
        rt, chain, resnum = residue.coordinate_author_key
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
                nr=index,
                metrics=row_metrics,
                default=default,
                omit_cp=blank,
            )
        )
    return rows


def write_edstats_for_structure(path, context, **kwargs) -> str:
    """Write a complete synthetic ``stats.out`` for a loaded structure.

    >>> ctx = load_structure("test", pdb_path)                 # doctest: +SKIP
    >>> stats = write_edstats_for_structure(tmp_path / "s.out", ctx,
    ...                                     metrics={"ZDm": 2.0})  # doctest: +SKIP
    """
    return write_edstats(path, edstats_rows_for_structure(context, **kwargs))


def stats_rows_for_structure(
    context,
    path,
    *,
    pdb_id: Optional[str] = None,
    metals: Optional[Iterable[str]] = None,
    cofactors: Optional[Iterable[str]] = None,
    **kwargs,
):
    """Write a synthetic ``stats.out`` then parse it back with the real code.

    Returns ``(rows, header, stats_path)`` where ``rows``/``header`` are exactly
    what ``metal_identification.extract_metal_statistics`` produced, ready to
    hand to ``bond_analysis.run_bond_analysis`` as its sigma join input.
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


# Capability probes
CCP4_TOOLS = ("mtzfix", "fft", "mapmask", "edstats")
_NETWORK_CACHE: Dict[Tuple[str, int], bool] = {}


def ccp4_env() -> Optional[Dict[str, str]]:
    """Return an environment with the CCP4 tools on PATH, or ``None``.

    Tries the current environment first, then ``find_ccp4_setup``
    (``CCP4_SETUP``, the user config, and the usual install locations) sourced
    through ``resolve_env``.
    """
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


def which(program: str, env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``shutil.which`` against an explicit environment's PATH."""
    return shutil.which(program, path=(env or os.environ).get("PATH"))


#: Canonical name of the warm PDB-REDO download cache variable, first, followed
#: by the spellings kept working for compatibility. Every reader in the suite
#: goes through :func:`cache_dir_from_env` so one export is enough whichever
#: name a caller learned first.
CACHE_ENV_VARS = ("ALCHEMY_TESTS_CACHE", "ALCHEMY_TEST_CACHE")


def cache_dir_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The configured PDB-REDO cache directory, or ``None`` if unset.

    Reads :data:`CACHE_ENV_VARS` in order, so the canonical
    ``ALCHEMY_TESTS_CACHE`` wins when both are exported. Blank and
    whitespace-only values count as unset -- ``ALCHEMY_TESTS_CACHE=`` in a
    wrapper script must not resolve the cache to the current directory.
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
