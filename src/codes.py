"""The status and provenance words Alchemy writes, defined once."""

from enum import StrEnum


class GeometryStatus(StrEnum):
    """Site-level verdict on the geometry of one metal's coordination.

    Written to ``explicit_geometry_status`` and
    ``image_inclusive_geometry_status``.
    """

    # INSUFFICIENT_DATA keeps its space: it is a published column value, and
    # changing it reclassifies every site in an existing output file.
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
    """Why an entry did not finish as a plain ``ok``, carried to the manifest.

    These are the most user-visible words Alchemy writes, and
    ``docs/operations.md`` documents them, so they belong here rather than as
    literals at the point of use. One of them is also read back:
    ``driver.resume`` decides whether a bond stage was applicable by testing for
    ``METAL_PRESENCE_INDETERMINATE``, and with the string spelled out at both
    ends a rename would silently reprocess those entries on every resume.
    """

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
    #: An unanticipated exception that will recur identically on the same
    #: inputs, so the entry is terminal rather than retried forever.
    DETERMINISTIC_PROCESSING_ERROR = "deterministic_processing_error"
    #: An atom's element could not be trusted, so metal absence cannot be
    #: established under the no-inference policy and no site is analysable.
    METAL_PRESENCE_INDETERMINATE = "metal_presence_indeterminate"
    BOND_STAGE_FAILURE = "bond_stage_failure"
    #: An EDSTATS row for a catalog cofactor matched no coordinate residue.
    COFACTOR_COORDINATE_JOIN_FAILED = "cofactor_coordinate_join_failed"
    AMBIGUOUS_COORDINATE_RESIDUE_JOIN = "ambiguous_coordinate_residue_join"
    COFACTOR_WITHOUT_SELECTED_METAL = "cofactor_without_selected_metal"
    #: Bond analysis measured a metal site that statistics did not report, so
    #: its contacts carry no density evidence and it is not counted as a site.
    METAL_SITE_WITHOUT_DENSITY = "metal_site_without_density"
    DECLARED_CONNECTION_RESOLUTION_INCOMPLETE = (
        "declared_connection_resolution_incomplete"
    )
    SYMMETRY_SEARCH_UNAVAILABLE = "symmetry_search_unavailable"
    MISSING_FIRST_SPHERE_REFERENCE = "missing_first_sphere_reference"
    #: A selected metal has a NaN or infinite Cartesian coordinate, so no
    #: distance-based evidence can be collected for that site.
    NON_FINITE_METAL_COORDINATES = "non_finite_metal_coordinates"
    #: The remaining members are the DPI's own reasons for being unavailable,
    #: each recorded as the entry's partial reason.
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
    #: A metal record was excluded from site selection because its occupancy is
    #: a valid zero. Zero occupancy is not evidence for a site, but without this
    #: an entry whose only metal is modeled absent reports ``no_metals`` -- an
    #: authoritative negative about a file that does contain a metal record.
    #: ``zero_occupancy_atoms`` alone cannot say the metal was the atom dropped.
    ZERO_OCCUPANCY_METAL_EXCLUDED = "zero_occupancy_metal_excluded"
    OVERFULL_ALTERNATE_OCCUPANCY = "overfull_alternate_occupancy"
    OCCUPANCY_DICTIONARY_DEFAULT_APPLIED = "occupancy_dictionary_default_applied"
    RAW_OCCUPANCY_MAPPING_FAILED = "raw_occupancy_mapping_failed"
    LEGACY_PDB_IDENTIFIERS_PACKED = "legacy_pdb_identifiers_packed"
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
