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


class WarningCode(StrEnum):
    """Non-fatal observations about an entry, carried to the manifest."""

    MULTI_MODEL_STRUCTURE = "multi_model_structure"
    DUPLICATE_ATOM_RECORDS = "duplicate_atom_records"
    DUPLICATE_ATOM_COORDINATE_CONFLICT = "duplicate_atom_coordinate_conflict"
    MALFORMED_DUPLICATE_ATOM_NAMES = "malformed_duplicate_atom_names"
    ALTLOC_SELECTION_FALLBACK = "altloc_selection_fallback"
    UNKNOWN_ELEMENTS = "unknown_elements"
    ZERO_OCCUPANCY_ATOMS = "zero_occupancy_atoms"
    RAW_OCCUPANCY_MAPPING_FAILED = "raw_occupancy_mapping_failed"
    LEGACY_PDB_IDENTIFIERS_PACKED = "legacy_pdb_identifiers_packed"
    #: Refmac's twinned map coefficients were rewritten for EDSTATS.
    TWIN_REFMAC_COEFFICIENTS_NORMALIZED = "twin_refmac_coefficients_normalized"
    #: A declaration named an alternate conformer that per-residue selection
    #: did not choose, so it was re-pointed onto the one that was chosen.
    DECLARED_CONNECTION_CONFORMER_SUBSTITUTED = (
        "declared_connection_conformer_substituted"
    )
    DECLARED_DONOR_ELEMENT_UNSUPPORTED = "declared_donor_element_unsupported"
    #: No bundled reference covers the donor's residue class, so its geometry
    #: can never be z-scored.
    DECLARED_DONOR_OUTSIDE_SUPPORTED_CLASSES = (
        "declared_donor_outside_supported_classes"
    )
