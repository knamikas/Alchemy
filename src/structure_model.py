"""Alchemy's view of one analyzed coordinate model.

``AtomSite`` keeps every deduplicated deposited atom for occupancy-weighted
counts, ``ResidueSelection`` pairs those atoms with one conformer per residue
for contact searches, and ``StructureContext`` carries both together with the
symmetry, occupancy, and record audits that ``structure_analysis.load_structure``
fills in. ``ContactImage`` describes one neighbor image found around a metal.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, cast

import gemmi

from codes import ContactScope

NAN = float("nan")

#: The symmetry code Gemmi assigns to the identity image of the asymmetric unit.
IDENTITY_SYMMETRY_OPERATION = "1_555"
IDENTITY_TRANSLATION = (0, 0, 0)
#: Elements whose atoms never count toward the heavy-atom total.
HYDROGEN_ELEMENTS = ("H", "D")


def position_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Cartesian distance between two three-dimensional positions."""
    ax, ay, az = a
    bx, by, bz = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


class _PbcShiftImage(Protocol):
    """The typed part of Gemmi's nearest-image result that Alchemy consumes."""

    pbc_shift: tuple[int, int, int]


def pbc_translation(image: object) -> tuple[int, int, int]:
    """Return a nearest image's integer unit-cell translation.

    Gemmi's stub leaves the tuple element type unspecified, so this small
    boundary records the concrete three-integer contract exposed at runtime.
    """
    shift_a, shift_b, shift_c = cast(_PbcShiftImage, image).pbc_shift
    return int(shift_a), int(shift_b), int(shift_c)


def spacegroup_or_none(structure: gemmi.Structure) -> gemmi.SpaceGroup | None:
    """Return the structure's space group, or ``None`` if Gemmi cannot name it.

    The binding returns ``None`` for an unidentified space group although its
    bundled stub declares a plain ``SpaceGroup``; this is the one place that
    narrows it, so malformed entries get an explicit boundary.
    """
    return cast("gemmi.SpaceGroup | None", structure.find_spacegroup())


@dataclass(frozen=True, slots=True)
class ResidueIdentity:
    """Where one residue sits in the analyzed model and how its authors name it.

    ``chain_id``, ``residue_name`` and ``resnum`` report the source author
    identity. When conversion packed a very large mmCIF into a legacy-PDB
    namespace, the analysis coordinates (and EDSTATS) instead see the
    ``coordinate_*`` identity, and the ``source_*`` indices place the residue
    in the source model rather than the analysis model.
    """

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
    coordinate_chain_id: str = ""
    coordinate_resnum: str = ""
    source_polymer_position: str = ""
    source_chain_index: int | None = None
    source_residue_index: int | None = None

    @property
    def key(self) -> tuple[int, int, int]:
        """Return stable model, chain, and residue indices."""
        return self.model_index, self.chain_index, self.residue_index

    @property
    def author_key(self) -> tuple[str, str, str]:
        """Return the source author residue identity."""
        return self.residue_name, self.chain_id, self.resnum

    @property
    def coordinate_author_key(self) -> tuple[str, str, str]:
        """Return the author identity present in the analysis coordinates."""
        return (
            self.coordinate_residue_name,
            self.coordinate_chain_id or self.chain_id,
            self.coordinate_resnum or self.resnum,
        )

    @property
    def output_chain_index(self) -> int:
        """Return the source-facing chain index used in output identities."""
        return (
            self.chain_index
            if self.source_chain_index is None
            else self.source_chain_index
        )

    @property
    def output_residue_index(self) -> int:
        """Return the source-facing residue index used in output identities."""
        return (
            self.residue_index
            if self.source_residue_index is None
            else self.source_residue_index
        )


class _ResidueIdentityAccess:
    """Expose a composed ``ResidueIdentity`` as flat read-only attributes.

    Atoms and residues are addressed by their author fields throughout the
    analysis, so both record types present the identity as their own fields.
    """

    identity: ResidueIdentity

    @property
    def model_index(self) -> int:
        """Return the analyzed model's index within the coordinate file."""
        return self.identity.model_index

    @property
    def model_id(self) -> str:
        """Return the analyzed model's deposited number."""
        return self.identity.model_id

    @property
    def chain_index(self) -> int:
        """Return the chain's index within the analyzed model."""
        return self.identity.chain_index

    @property
    def chain_id(self) -> str:
        """Return the source author chain name."""
        return self.identity.chain_id

    @property
    def residue_index(self) -> int:
        """Return the residue's index within its chain."""
        return self.identity.residue_index

    @property
    def residue_name(self) -> str:
        """Return the source component name."""
        return self.identity.residue_name

    @property
    def coordinate_residue_name(self) -> str:
        """Return the component name present in the analysis coordinates."""
        return self.identity.coordinate_residue_name

    @property
    def residue_number(self) -> int:
        """Return the source author sequence number."""
        return self.identity.residue_number

    @property
    def insertion_code(self) -> str:
        """Return the source author insertion code, or an empty string."""
        return self.identity.insertion_code

    @property
    def resnum(self) -> str:
        """Return the source author sequence number with its insertion code."""
        return self.identity.resnum

    @property
    def coordinate_chain_id(self) -> str:
        """Return the chain name present in the analysis coordinates."""
        return self.identity.coordinate_chain_id

    @property
    def coordinate_resnum(self) -> str:
        """Return the residue number present in the analysis coordinates."""
        return self.identity.coordinate_resnum

    @property
    def source_polymer_position(self) -> str:
        """Return the deposited polymer position code, or an empty string."""
        return self.identity.source_polymer_position

    @property
    def source_chain_index(self) -> int | None:
        """Return the chain index in the source model, if it differs."""
        return self.identity.source_chain_index

    @property
    def source_residue_index(self) -> int | None:
        """Return the residue index in the source model, if it differs."""
        return self.identity.source_residue_index

    @property
    def output_chain_index(self) -> int:
        """Return the source-facing chain index used in output identities."""
        return self.identity.output_chain_index

    @property
    def output_residue_index(self) -> int:
        """Return the source-facing residue index used in output identities."""
        return self.identity.output_residue_index


@dataclass
class AtomSite(_ResidueIdentityAccess):
    """One deposited atom record with stable source-model indices."""

    identity: ResidueIdentity
    atom_index: int
    source_order: int
    atom_name: str
    altloc: str
    element: str
    element_known: bool
    occupancy: float
    occupancy_valid: bool
    occupancy_status: str
    x: float
    y: float
    z: float
    is_water: bool
    is_hydrogen: bool
    gemmi_atom: gemmi.Atom = field(repr=False, compare=False)

    @property
    def pos(self) -> gemmi.Position:
        """Return the Gemmi position backing this atom."""
        return self.gemmi_atom.pos

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Return Cartesian coordinates in angstroms."""
        return self.x, self.y, self.z

    @property
    def b_iso(self) -> float:
        """Coordinate-model isotropic or equivalent-isotropic B factor."""
        return float(self.gemmi_atom.b_iso)

    @property
    def coordinates_valid(self) -> bool:
        """Whether this atom can safely participate in spatial analysis."""
        return all(math.isfinite(value) for value in self.xyz)

    @property
    def residue_key(self) -> tuple[int, int, int]:
        """Return stable model, chain, and residue indices."""
        return self.identity.key

    @property
    def source_key(self) -> tuple[int, int, int, int]:
        """Return stable model-local indices for this atom."""
        return (*self.residue_key, self.atom_index)

    @property
    def exact_identity(self) -> tuple[object, ...]:
        """Return residue, atom, alternate-location, and element identity."""
        return (*self.residue_key, self.atom_name, self.altloc, self.element)

    @property
    def chemical_site_identity(self) -> tuple[object, ...]:
        """Return atom identity with alternate locations collapsed."""
        return (*self.residue_key, self.atom_name, self.element)


@dataclass
class ResidueSelection(_ResidueIdentityAccess):
    """All source sites and the selected canonical sites for one residue."""

    identity: ResidueIdentity
    is_water: bool
    source_atoms: tuple[AtomSite, ...]
    contact_atoms: tuple[AtomSite, ...]
    selected_altloc: str
    selected_conformer_mean_occupancy: float | None
    altloc_options: str
    alternative_conformers_present: bool
    altloc_selection_fallback: bool
    selected_over_blank_duplicate_count: int
    malformed_duplicate_atom_name_count: int
    chemical_atom_site_count: int

    @property
    def key(self) -> tuple[int, int, int]:
        """Return stable model, chain, and residue indices."""
        return self.identity.key

    @property
    def author_key(self) -> tuple[str, str, str]:
        """Return the source author residue identity."""
        return self.identity.author_key

    @property
    def coordinate_author_key(self) -> tuple[str, str, str]:
        """Return the author identity present in the analysis coordinates."""
        return self.identity.coordinate_author_key

    @property
    def elements(self) -> frozenset[str]:
        """Return the validated elements deposited in this residue."""
        return frozenset(
            atom.element for atom in self.source_atoms if atom.element_known
        )


#: ``(model_index, chain_index, residue_index)``, as ``ResidueIdentity.key``.
ResidueKey = tuple[int, int, int]
#: ``(chain_index, residue_index, atom_index)``, as a Gemmi neighbor mark reports.
AtomIndices = tuple[int, int, int]
#: ``(residue_name, chain_id, resnum)`` in either source or coordinate form.
AuthorResidueKey = tuple[str, str, str]
#: Residues sharing one author identity, in model order.
ResidueIndex = Mapping[AuthorResidueKey, tuple[ResidueSelection, ...]]


class ImageProvenance(NamedTuple):
    """How a Gemmi cell image relates to the explicit asymmetric unit."""

    crystallographic: bool
    strict_ncs: bool
    ncs_operation_id: str
    scope: ContactScope


@dataclass(frozen=True, slots=True)
class SymmetryMetadata:
    """Whether Gemmi image searches can run, and the operations they cover."""

    search_available: bool
    search_failure_reason: str
    crystallographic_operation_count: int
    strict_ncs_operation_ids: tuple[str, ...]

    @property
    def strict_ncs_operation_count(self) -> int:
        """Return the number of noncrystallographic symmetry operations."""
        return len(self.strict_ncs_operation_ids)

    @property
    def dpi_atom_count_multiplier(self) -> int:
        """Number of full coordinate copies represented in the asymmetric unit."""
        return 1 + self.strict_ncs_operation_count


@dataclass(frozen=True, slots=True)
class OccupancyValidation:
    """What the deposited occupancies allow, and what disqualified them.

    ``validation_failed`` disables every occupancy-weighted quantity, DPI
    included; the counts explain why and are published with each site.
    """

    validation_failed: bool
    missing_count: int
    invalid_count: int
    overfull_site_count: int
    overfull_excess: float
    overfull_site_keys: frozenset[tuple[object, ...]]
    defaulted_atom_count: int
    zero_atom_count: int
    raw_mapping_failed: bool
    raw_mapping_failure_reason: str


@dataclass(frozen=True, slots=True)
class AtomRecordAudit:
    """Defects found in the deposited atom records while building the model."""

    duplicate_records_present: bool
    duplicate_record_count: int
    coordinate_conflict_count: int
    malformed_duplicate_atom_name_count: int
    unknown_element_atom_count: int
    element_validation_warning: str
    non_finite_coordinate_atom_count: int


@dataclass
class StructureContext:
    """Gemmi structure plus deterministic first-model analysis metadata."""

    pdb_id: str
    structure: gemmi.Structure
    model: gemmi.Model
    model_index: int
    model_policy: str
    input_model_count: int
    model_analyzed: int
    analyzed_model_id: str
    multi_model_structure: bool
    source_atoms: tuple[AtomSite, ...]
    contact_atoms: tuple[AtomSite, ...]
    residues: tuple[ResidueSelection, ...]
    occupancy: OccupancyValidation
    records: AtomRecordAudit
    symmetry: SymmetryMetadata
    analysis_coordinate_format: str
    warning_codes: tuple[str, ...]
    #: Occupancy-weighted non-H/D count of the deposited first-model records,
    #: or NaN when the occupancies or elements leave it unknowable.
    deposited_ni: float
    _spatial_model: gemmi.Model = field(repr=False)
    _atom_by_indices: Mapping[AtomIndices, AtomSite] = field(
        repr=False, default_factory=dict[AtomIndices, AtomSite]
    )
    _residue_by_key: Mapping[ResidueKey, ResidueSelection] = field(
        repr=False, default_factory=dict[ResidueKey, ResidueSelection]
    )
    _residues_by_author: ResidueIndex = field(
        repr=False,
        default_factory=dict[AuthorResidueKey, tuple[ResidueSelection, ...]],
    )
    _residues_by_source_author: ResidueIndex = field(
        repr=False,
        default_factory=dict[AuthorResidueKey, tuple[ResidueSelection, ...]],
    )
    _residues_by_coordinate_author: ResidueIndex = field(
        repr=False,
        default_factory=dict[AuthorResidueKey, tuple[ResidueSelection, ...]],
    )

    def image_provenance(
        self,
        image_index: int,
        cell_translation: tuple[int, int, int],
    ) -> ImageProvenance:
        """Classify a Gemmi cell image as crystallographic and/or strict NCS.

        ``setup_cell_images()`` orders the identity first, then the remaining
        space-group operations, then one complete block of those operations per
        strict-NCS transform. See
        https://gemmi.readthedocs.io/en/stable/analysis.html.
        """
        if image_index < 0:
            raise ValueError("Gemmi image index cannot be negative")
        operation_count = self.symmetry.crystallographic_operation_count
        if operation_count <= 0:
            raise ValueError("crystallographic operation count is unavailable")

        strict_ncs = image_index >= operation_count
        crystallographic = (
            image_index % operation_count != 0
            or cell_translation != IDENTITY_TRANSLATION
        )
        ncs_operation_id = ""
        if strict_ncs:
            ncs_index = image_index // operation_count - 1
            operation_ids = self.symmetry.strict_ncs_operation_ids
            if 0 <= ncs_index < len(operation_ids):
                ncs_operation_id = operation_ids[ncs_index]
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
        return ImageProvenance(crystallographic, strict_ncs, ncs_operation_id, scope)

    def atom_for_indices(
        self, chain_index: int, residue_index: int, atom_index: int
    ) -> AtomSite | None:
        """Return the selected atom at model-local indices, if present."""
        return self._atom_by_indices.get((chain_index, residue_index, atom_index))

    def atom_for_mark(self, mark: gemmi.NeighborSearch.Mark) -> AtomSite | None:
        """Resolve a Gemmi neighbor-search mark to a selected atom."""
        return self.atom_for_indices(mark.chain_idx, mark.residue_idx, mark.atom_idx)

    def residue_for_atom(self, atom: AtomSite) -> ResidueSelection:
        """Return the selected residue containing an atom."""
        return self._residue_by_key[atom.residue_key]

    def residues_for_author(
        self, residue_name: str, chain_id: str, resnum: str
    ) -> tuple[ResidueSelection, ...]:
        """Return residues matching either source or coordinate author identity."""
        return self._residues_by_author.get(
            (str(residue_name), str(chain_id), str(resnum)), ()
        )

    def residues_for_source_author(
        self, residue_name: str, chain_id: str, resnum: str
    ) -> tuple[ResidueSelection, ...]:
        """Return residues matching a source-coordinate author identity."""
        return self._residues_by_source_author.get(
            (str(residue_name), str(chain_id), str(resnum)), ()
        )

    def residues_for_coordinate_author(
        self, residue_name: str, chain_id: str, resnum: str
    ) -> tuple[ResidueSelection, ...]:
        """Return residues matching the analysis-coordinate author identity."""
        return self._residues_by_coordinate_author.get(
            (str(residue_name), str(chain_id), str(resnum)), ()
        )

    def metal_atoms(
        self,
        elements: Iterable[str],
        canonical: bool = True,
        include_zero_occupancy: bool = False,
    ) -> list[AtomSite]:
        """Return selected metal atoms, excluding modeled absence by default.

        A valid occupancy of zero remains part of the deposited atom inventory
        and the ``Ni`` sum, but it is not an analyzable metal site. Invalid or
        missing occupancies are retained so coordinate-only analysis can still
        proceed while separate occupancy-validation flags disable scoring.
        """
        wanted = {str(element).upper() for element in elements}
        atoms = self.contact_atoms if canonical else self.source_atoms
        return [
            atom
            for atom in atoms
            if atom.element_known
            and atom.element in wanted
            and (
                include_zero_occupancy
                or not (atom.occupancy_valid and atom.occupancy == 0.0)
            )
        ]

    def make_neighbor_search(
        self,
        radius: float,
        include_symmetry: bool = True,
        positive_occupancy_only: bool = True,
    ) -> gemmi.NeighborSearch:
        """Build a neighbor search over finite selected atom positions."""
        if include_symmetry:
            if not self.symmetry.search_available:
                raise ValueError(
                    self.symmetry.search_failure_reason
                    or "symmetry image search is unavailable"
                )
            cell = self.structure.cell
        else:
            cell = gemmi.UnitCell()
        # An empty UnitCell makes Gemmi derive bounds from all model atoms,
        # so use a model with finite coordinates only.
        search = gemmi.NeighborSearch(self._spatial_model, cell, radius)
        for atom in self.contact_atoms:
            # Gemmi raises while binning a NaN position. Keep malformed atoms
            # in the source inventory, but never hand them to spatial code.
            if not atom.coordinates_valid:
                continue
            if positive_occupancy_only and not (
                atom.occupancy_valid and atom.occupancy > 0
            ):
                continue
            search.add_atom(
                atom.gemmi_atom, atom.chain_index, atom.residue_index, atom.atom_index
            )
        return search


@dataclass(frozen=True, slots=True)
class ContactImage:
    """Geometry and provenance of one neighbor image around a metal.

    ``position`` is the neighbor's coordinates in the image that touches the
    metal; ``distance`` is measured to that image. The identity image of the
    explicit asymmetric unit carries the fixed ``1_555`` operation, a zero
    translation, and no symmetry provenance.
    """

    distance: float
    position: tuple[float, float, float]
    crystallographic_contact: bool
    strict_ncs_contact: bool
    strict_ncs_operation_id: str
    scope: ContactScope
    image_index: int
    symmetry_operation: str
    translation: tuple[int, int, int]

    @property
    def symmetry_contact(self) -> bool:
        """Whether the image is generated by any symmetry operation."""
        return self.crystallographic_contact or self.strict_ncs_contact

    @classmethod
    def explicit(cls, metal: AtomSite, neighbor: AtomSite) -> ContactImage:
        """Return the deposited neighbor in the explicit asymmetric unit."""
        return cls(
            distance=position_distance(metal.xyz, neighbor.xyz),
            position=neighbor.xyz,
            crystallographic_contact=False,
            strict_ncs_contact=False,
            strict_ncs_operation_id="",
            scope=ContactScope.EXPLICIT,
            image_index=0,
            symmetry_operation=IDENTITY_SYMMETRY_OPERATION,
            translation=IDENTITY_TRANSLATION,
        )

    @classmethod
    def from_nearest_image(
        cls,
        structure: StructureContext,
        nearest: gemmi.NearestImage,
        position: gemmi.Position,
    ) -> ContactImage:
        """Return the image Gemmi selected, classified against ``structure``.

        ``position`` is the neighbor already transformed into that image;
        callers compute it because Gemmi offers two routes that differ in the
        last bits.
        """
        translation = pbc_translation(nearest)
        image_index = int(nearest.sym_idx)
        provenance = structure.image_provenance(image_index, translation)
        return cls(
            distance=float(nearest.dist()),
            position=(float(position.x), float(position.y), float(position.z)),
            crystallographic_contact=provenance.crystallographic,
            strict_ncs_contact=provenance.strict_ncs,
            strict_ncs_operation_id=provenance.ncs_operation_id,
            scope=provenance.scope,
            image_index=image_index,
            symmetry_operation=nearest.symmetry_code(),
            translation=translation,
        )
