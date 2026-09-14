"""Shared status, reason, and provenance values for serialized outputs."""

from enum import StrEnum


class EntryStatus(StrEnum):
    """Outcome recorded for one processed entry in the run manifest."""

    OK = "ok"
    PARTIAL = "partial"
    SKIP = "skip"
    ERROR = "error"


class RunMode(StrEnum):
    """How a run chose its entries, as recorded in the run report.

    Only ``DATABASE`` -- an uncapped pass over a PDB-REDO mirror -- can build
    a new confidence reference; every other mode scores against one.
    """

    MANUAL = "manual"
    SINGLE = "single"
    ID_FILE = "id_file"
    DATABASE = "database"
    CAPPED_DATABASE = "capped_database"


class GeometryStatus(StrEnum):
    """Site-level verdict on the geometry of one metal's coordination.

    Written to ``explicit_geometry_status`` and
    ``image_inclusive_geometry_status``.
    """

    # Preserve the space in this published CSV value.
    #: No contact could be scored -- no reference covered them, or the metal
    #: itself has zero occupancy.
    INSUFFICIENT_DATA = "insufficient data"
    #: At least one scored contact is a geometry outlier.
    SUSPECT = "suspect"
    #: Contacts were scored and none was an outlier.
    PLAUSIBLE = "plausible"


class MultiDonorStatus(StrEnum):
    """Verdict on one donor-residue image contacting one metal.

    Written to ``multi_donor_geometry_status``. Group status is contextual: it
    never weakens or excludes an individual bond's own z-score.
    """

    #: The residue contributes exactly one contact, so there is no group.
    SINGLE_DONOR = "single_donor"
    #: Every contact in the group was scored and none is an outlier.
    CONSISTENT = "consistent"
    #: At least one contact in the group is a geometry outlier.
    SUSPECT = "suspect"
    #: The group has contacts that could not be scored, so no verdict holds.
    INDETERMINATE = "indeterminate"


class ContactScope(StrEnum):
    """Which crystallographic operation produced a contact's donor image."""

    #: The donor is in the deposited asymmetric unit; no operation applied.
    EXPLICIT = "explicit"
    CRYSTALLOGRAPHIC = "crystallographic"
    STRICT_NCS = "strict_ncs"
    STRICT_NCS_AND_CRYSTALLOGRAPHIC = "strict_ncs_and_crystallographic"


class CandidateSource(StrEnum):
    """How a candidate contact came to Alchemy's attention.

    A merged candidate carries more than one, joined by ``|`` in the output.
    """

    PROXIMITY_4A = "proximity_4A"
    #: Declared by the source mmCIF's ``_struct_conn`` records.
    STRUCT_CONN = "struct_conn"
    #: Declared by a source PDB's ``LINK`` records.
    LINK = "LINK"


class OccupancyStatus(StrEnum):
    """Why one deposited atom's occupancy is or is not usable."""

    VALID = "valid"
    #: The column was blank: a missing measurement, not a malformed one.
    MISSING = "missing"
    INVALID_NON_NUMERIC = "invalid_non_numeric"
    INVALID_NON_FINITE = "invalid_non_finite"
    #: Numeric and finite, but outside the physical range 0.0 to 1.0.
    INVALID_RANGE = "invalid_range"
    #: The raw PDB record could not be joined back to the parsed atom.
    RAW_MAPPING_FAILED = "raw_mapping_failed"
    #: Unusable with no raw record to say how, which is all an mmCIF-sourced
    #: atom can report.
    INVALID_VALUE = "invalid_value"


class ElementStatus(StrEnum):
    """Whether an atom's deposited element symbol could be trusted."""

    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"


class ReasonCode(StrEnum):
    """Reasons for incomplete or excluded entries.

    Documented in docs/operations.md and used by resume to interpret prior results.
    """

    # Entry lifecycle: the entry could not be processed at all.
    #: The pool never received a result: the worker holding the entry died.
    WORKER_PROCESS_DIED = "worker_process_died"
    #: An input file named on the command line or in the mirror is absent.
    MISSING_INPUT = "missing_input"
    #: A CCP4 program was killed at ``--ccp4-timeout``. It reported nothing
    #: about the entry, so unlike a failure exit this is worth retrying.
    CCP4_TOOL_TIMEOUT = "ccp4_tool_timeout"
    MTZFIX_VALIDATION_FAILURE = "mtzfix_validation_failure"
    #: An unanticipated exception whose type leaves a retry meaningful.
    UNEXPECTED_PROCESSING_ERROR = "unexpected_processing_error"
    #: An exception expected to recur on identical inputs; resume may retry it.
    DETERMINISTIC_PROCESSING_ERROR = "deterministic_processing_error"
    # Cohort membership: the entry was processed but deliberately excluded.
    #: An atom's element could not be trusted, so metal absence cannot be
    #: established under the no-inference policy and no site is analysable.
    METAL_PRESENCE_INDETERMINATE = "metal_presence_indeterminate"
    #: The standard cohort excludes metal-dense assemblies whose correlated
    #: sites would dominate both the reference population and batch runtime.
    METAL_SITE_LIMIT_EXCEEDED = "metal_site_limit_exceeded"
    # Stage failures and density-to-coordinate joins.
    BOND_STAGE_FAILURE = "bond_stage_failure"
    #: An EDSTATS row for a catalog cofactor matched no coordinate residue.
    COFACTOR_COORDINATE_JOIN_FAILED = "cofactor_coordinate_join_failed"
    COFACTOR_WITHOUT_SELECTED_METAL = "cofactor_without_selected_metal"
    #: A selected coordinate metal site has no corresponding statistics row.
    METAL_SITE_WITHOUT_DENSITY = "metal_site_without_density"
    # Coordination analysis.
    DECLARED_CONNECTION_RESOLUTION_INCOMPLETE = (
        "declared_connection_resolution_incomplete"
    )
    SYMMETRY_SEARCH_UNAVAILABLE = "symmetry_search_unavailable"
    MISSING_FIRST_SPHERE_REFERENCE = "missing_first_sphere_reference"
    #: A selected metal has a NaN or infinite Cartesian coordinate, so no
    #: distance-based evidence can be collected for that site.
    NON_FINITE_METAL_COORDINATES = "non_finite_metal_coordinates"
    # DPI availability: each is recorded as the entry's partial reason.
    MISSING_DPI_METADATA_SOURCE = "missing_dpi_metadata_source"
    INVALID_DPI_METADATA = "invalid_dpi_metadata"
    INVALID_OCCUPANCY = "invalid_occupancy"
    MISSING_OR_INVALID_REFLECTION_COUNT = "missing_or_invalid_reflection_count"
    MISSING_OR_INVALID_RFREE = "missing_or_invalid_rfree"
    MISSING_OR_INVALID_ASU_VOLUME = "missing_or_invalid_asu_volume"
    INVALID_DPI_ATOM_COUNT = "invalid_dpi_atom_count"
    DPI_CALCULATION_FAILED = "dpi_calculation_failed"


class WarningCode(StrEnum):
    """Non-fatal observations about an entry, carried to the manifest."""

    #: Density was retained for a selected metal in a multi-atom component
    #: absent from the frozen cofactor catalog; no structural class is inferred.
    COFACTOR_CATALOG_FALLBACK = "cofactor_catalog_fallback"
    MULTI_MODEL_STRUCTURE = "multi_model_structure"
    DUPLICATE_ATOM_RECORDS = "duplicate_atom_records"
    DUPLICATE_ATOM_COORDINATE_CONFLICT = "duplicate_atom_coordinate_conflict"
    MALFORMED_DUPLICATE_ATOM_NAMES = "malformed_duplicate_atom_names"
    ALTLOC_SELECTION_FALLBACK = "altloc_selection_fallback"
    UNKNOWN_ELEMENTS = "unknown_elements"
    #: One or more atoms have NaN or infinite Cartesian coordinates. They stay
    #: in the deposited inventory for provenance but are excluded from every
    #: spatial search.
    NON_FINITE_COORDINATES = "non_finite_coordinates"
    ZERO_OCCUPANCY_ATOMS = "zero_occupancy_atoms"
    #: A metal was excluded for valid zero occupancy. Distinguish modeled
    #: absence from an entry containing no metal records.
    ZERO_OCCUPANCY_METAL_EXCLUDED = "zero_occupancy_metal_excluded"
    OVERFULL_ALTERNATE_OCCUPANCY = "overfull_alternate_occupancy"
    OCCUPANCY_DICTIONARY_DEFAULT_APPLIED = "occupancy_dictionary_default_applied"
    RAW_OCCUPANCY_MAPPING_FAILED = "raw_occupancy_mapping_failed"
    LEGACY_PDB_IDENTIFIERS_PACKED = "legacy_pdb_identifiers_packed"
    #: EDSTATS printed ``****`` because an NPm/NPs/NPa grid-point count did
    #: not fit its fixed-width field. The count is retained as unavailable;
    #: all other density metrics and the entry remain usable.
    EDSTATS_GRID_POINT_COUNT_OVERFLOW = "edstats_grid_point_count_overflow"
    #: Refmac's twinned map coefficients were rewritten for EDSTATS.
    TWIN_REFMAC_COEFFICIENTS_NORMALIZED = "twin_refmac_coefficients_normalized"
    #: A declaration named an alternate conformer that per-residue selection
    #: did not choose, so it was re-pointed onto the one that was chosen.
    DECLARED_CONNECTION_CONFORMER_SUBSTITUTED = (
        "declared_connection_conformer_substituted"
    )
    #: A source declaration resolves to an atom whose valid occupancy is zero.
    #: The declaration remains candidate evidence but cannot become a bond.
    DECLARED_CONNECTION_ZERO_OCCUPANCY_PARTNER = (
        "declared_connection_zero_occupancy_partner"
    )
    DECLARED_DONOR_ELEMENT_UNSUPPORTED = "declared_donor_element_unsupported"
    #: No bundled reference covers the donor's residue class, so its geometry
    #: can never be z-scored.
    DECLARED_DONOR_OUTSIDE_SUPPORTED_CLASSES = (
        "declared_donor_outside_supported_classes"
    )


class CoordinateMappingStatus(StrEnum):
    """How an EDSTATS residue row joined to the analyzed coordinates."""

    MATCHED = "matched"
    RESIDUE_NOT_FOUND = "coordinate_residue_not_found"


class SelectedSiteStatus(StrEnum):
    """Whether a density row belongs to a selected metal site."""

    SELECTED = "selected"
    #: A catalog cofactor row whose residue holds no selected metal.
    NO_SELECTED_METAL = "no_selected_metal"


class DensityMapScope(StrEnum):
    """Extent of the maps EDSTATS was given.

    Only ``MODEL_ENVELOPE`` and ``FULL`` can be requested; the fallbacks record
    why a requested envelope was replaced by the full map.
    """

    MODEL_ENVELOPE = "model-envelope"
    FULL = "full"
    #: The cropped map was no smaller than the full map.
    FULL_SIZE_FALLBACK = "full-size-fallback"
    #: The crop started beyond the unit cell's positive edge, which EDSTATS
    #: would wrap.
    FULL_EXTENT_FALLBACK = "full-extent-fallback"


class ConfidenceLevel(StrEnum):
    """Verdict on one site or one evidence component."""

    PASS = "PASS"
    REVIEW = "REVIEW"
    SUSPECT = "SUSPECT"
    #: The component (or every component) could not be assessed.
    INCOMPLETE = "INCOMPLETE"


class EvidenceBasis(StrEnum):
    """Which evidence components contributed to a site's confidence level."""

    DENSITY_AND_GEOMETRY = "density_and_geometry"
    DENSITY_ONLY = "density_only"
    GEOMETRY_ONLY = "geometry_only"
    NO_ASSESSABLE_EVIDENCE = "no_assessable_evidence"


class VerdictReason(StrEnum):
    """Which rule of the non-compensatory decision matrix fired."""

    NO_ASSESSABLE_EVIDENCE = "no_assessable_evidence"
    DENSITY_AND_GEOMETRY_SUSPECT = "density_and_geometry_suspect"
    DENSITY_SUSPECT = "density_suspect"
    GEOMETRY_SUSPECT = "geometry_suspect"
    #: Two REVIEW components together are treated as SUSPECT.
    REVIEW_PLUS_REVIEW = "review_plus_review"
    DENSITY_REVIEW = "density_review"
    GEOMETRY_REVIEW = "geometry_review"
    ALL_AVAILABLE_COMPONENTS_PASS = "all_available_components_pass"


class ConfidenceInputStatus(StrEnum):
    """Which evidence components a prepared confidence-input row carries."""

    COMPLETE = "complete"
    DENSITY_ONLY = "density_only"
    GEOMETRY_ONLY = "geometry_only"
    UNSCORABLE = "unscorable"


class EligibilityStatus(StrEnum):
    """Whether a candidate contact may become an assigned first-sphere bond."""

    FIRST_SPHERE_ELIGIBLE = "first_sphere_eligible"
    OUTSIDE_FIRST_SPHERE = "outside_first_sphere"
    #: Within the cutoff, but the donor chemistry forbids inference.
    NON_TYPICAL_DONOR = "non_typical_donor"
    #: No reference distance exists for the metal-donor pair.
    MISSING_ASSIGNMENT_REFERENCE = "missing_assignment_reference"
    ZERO_OCCUPANCY = "zero_occupancy"


class EligibilityReason(StrEnum):
    """Why a candidate received its eligibility status.

    The two distance reasons embed ``FIRST_SPHERE_TOLERANCE`` from
    ``coordination.analysis``, which asserts the values agree at import.
    """

    DISTANCE_WITHIN_TOLERANCE = "distance_within_target_plus_0.75"
    DISTANCE_EXCEEDS_TOLERANCE = "distance_exceeds_target_plus_0.75"
    ATOM_NOT_TYPICAL_DONOR = "atom_not_in_typical_inferred_donor_list"
    NO_ASSIGNMENT_REFERENCE = "no_metal_donor_assignment_reference"
    ZERO_OCCUPANCY_ATOM = "zero_occupancy_atom_is_not_contact_evidence"


class ReferenceKind(StrEnum):
    """Which literature entry supplied a candidate's assignment target."""

    EXACT = "exact"
    #: No residue-specific entry; the donor element's generic entry was used.
    ELEMENT_FALLBACK = "element_fallback"
    MISSING = "missing"


class ParentType(StrEnum):
    """Structural class of the component that carries a selected metal."""

    CLUSTER = "cluster"
    HEME = "heme"
    ION = "ion"
    OTHER = "other"


class CoordinationStatus(StrEnum):
    """How a bond row came to be assigned."""

    DECLARED = "declared"
    INFERRED = "inferred"
    UNASSIGNED = "unassigned"


class NeighborClass(StrEnum):
    """Coarse chemical class of a contact's donor residue."""

    WATER = "water"
    NUCLEOTIDE = "nucleotide"
    AMINO_ACID = "amino_acid"
    OTHER = "other"
