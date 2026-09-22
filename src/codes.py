"""Shared status, reason, and provenance values for serialized outputs.

Every vocabulary written to a CSV column lives here so producers, resume, and
the documentation tests read one definition. Because these are ``StrEnum``
classes, a member compares equal to its string, and therefore to a member of
another vocabulary that happens to spell the same value. Such overlaps are
allowed only where the two vocabularies genuinely mean the same thing, and each
one is declared in ``SHARED_VALUES`` so an accidental new overlap fails the
documentation tests rather than passing a comparison against the wrong enum.
"""

from enum import StrEnum


class EntryStatus(StrEnum):
    """Outcome recorded for one processed entry in the run manifest."""

    OK = "ok"
    PARTIAL = "partial"
    SKIP = "skip"
    ERROR = "error"


class RefinementState(StrEnum):
    """Which refinement the analyzed reflections and metadata describe.

    Written to the manifest's ``refinement_state``.
    """

    #: Coordinates and reflections were supplied on the command line.
    MANUAL = "manual"
    #: The entry's final PDB-REDO refinement.
    FINAL = "final"


class RunMode(StrEnum):
    """How a run chose its entries, as recorded in the run report.

    Only ``DATABASE`` -- an uncapped pass over a PDB-REDO mirror -- can build
    a new score reference; every other mode scores against one.
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
    """Which crystallographic operation produced a contact's donor image.

    The first four are per-contact scopes, written to ``contact_scope``; every
    contact carries one. ``NONE`` is the site-level summary's addition to the
    same vocabulary, and no contact ever holds it.
    """

    #: The donor is in the deposited asymmetric unit; no operation applied.
    EXPLICIT = "explicit"
    CRYSTALLOGRAPHIC = "crystallographic"
    STRICT_NCS = "strict_ncs"
    STRICT_NCS_AND_CRYSTALLOGRAPHIC = "strict_ncs_and_crystallographic"
    #: Site level only: the generated-image search ran and no generated
    #: contact came of it. ``generated_contact_scope`` is blank instead when
    #: the search could not run, which is a different claim.
    NONE = "none"


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
    #: An unanticipated exception whose type leaves a retry meaningful. This
    #: includes the exception types a code defect typically raises, so a
    #: regression can never become a terminal exclusion.
    UNEXPECTED_PROCESSING_ERROR = "unexpected_processing_error"
    #: An exception that describes the entry's data and will recur on identical
    #: inputs; terminal for database completion, though resume may retry it.
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
    """How an EDSTATS residue row joined to the analyzed coordinates.

    Also written to score-input rows, which add the case where a bonded
    site had no density row to join at all.
    """

    MATCHED = "matched"
    RESIDUE_NOT_FOUND = "coordinate_residue_not_found"
    #: A score-input row built from bond rows alone.
    DENSITY_ROW_UNAVAILABLE = "density_row_unavailable"


class SelectedSiteStatus(StrEnum):
    """Whether a density row belongs to a selected metal site.

    Also written to score-input rows, which add two cases for selected
    sites that could not be joined to a density row.
    """

    SELECTED = "selected"
    #: A catalog cofactor row whose residue holds no selected metal.
    NO_SELECTED_METAL = "no_selected_metal"
    #: A site with bond rows but no density row; identified from the bonds.
    SELECTED_WITHOUT_DENSITY_ROW = "selected_without_density_row"
    #: A site the manifest counts as selected but no output row identifies.
    SELECTED_SITE_UNRESOLVED = "selected_site_unresolved"


class DensityContextStatus(StrEnum):
    """Whether an entry's non-target density aggregates were computed."""

    AVAILABLE = "available"
    NOT_COMPUTED = "not_computed"


class CrystallizationDataStatus(StrEnum):
    """Whether crystallization conditions could be read for an entry."""

    AVAILABLE = "available"
    #: The source held no condition record.
    NOT_REPORTED = "not_reported"
    UNPARSEABLE = "unparseable"
    #: No source could be consulted.
    INPUT_UNAVAILABLE = "input_unavailable"


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


class ScoreLevel(StrEnum):
    """Verdict on one site or one evidence component.

    Deliberately upper-case, unlike every other vocabulary here: a site-level
    verdict is a different claim from a lower-case ``GeometryStatus`` or
    ``MultiDonorStatus`` and must never compare equal to one.
    """

    PASS = "PASS"
    REVIEW = "REVIEW"
    SUSPECT = "SUSPECT"
    #: The component (or every component) could not be assessed.
    INCOMPLETE = "INCOMPLETE"


class EvidenceBasis(StrEnum):
    """Which evidence components contributed to a site's classification level."""

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


class ScoreInputStatus(StrEnum):
    """Which evidence components a prepared score-input row carries."""

    COMPLETE = "complete"
    DENSITY_ONLY = "density_only"
    GEOMETRY_ONLY = "geometry_only"
    UNSCORABLE = "unscorable"


class ScoreMissingReason(StrEnum):
    """Why a prepared score-input row lacks part of its evidence.

    Written pipe-joined to ``score_inputs_missing_reasons``, which mixes
    this vocabulary with ``ReasonCode.NON_FINITE_METAL_COORDINATES``,
    ``CoordinateMappingStatus.DENSITY_ROW_UNAVAILABLE``, and whichever
    ``ReasonCode`` the entry's own failure recorded. A reader therefore
    resolves a value against all three vocabularies, not this one alone.
    """

    #: No finite ``ZDm``, so the site carries no density evidence.
    RSZD_UNAVAILABLE = "rszd_unavailable"
    #: The site has no assigned contact, so no geometry could be computed.
    NO_ASSIGNED_CONTACTS = "no_assigned_contacts"
    #: Contacts exist but the literature reference covers none of them.
    NO_GEOMETRY_REFERENCE = "no_geometry_reference"
    #: A reference-covered contact yielded no finite score-eligible z-score.
    ZBOND_UNAVAILABLE_FOR_REFERENCE = "zbond_unavailable_for_reference"
    #: The reference covers only some of the assigned contacts.
    PARTIAL_GEOMETRY_COVERAGE = "partial_geometry_coverage"
    #: A placeholder row: the manifest counts the site but nothing names it.
    SITE_IDENTITY_UNAVAILABLE = "site_identity_unavailable"
    #: A placeholder row: no evidence of any kind reached the site. Also
    #: written to that row's ``context_warning_reasons``.
    SITE_EVIDENCE_UNAVAILABLE = "site_evidence_unavailable"


class GeometryContactBasis(StrEnum):
    """Which coordination provenance the scored contacts of a site came from.

    Written to ``geometry_contact_basis``; it describes the score-eligible
    contacts only, not every assigned one.
    """

    #: Both a declared and an inferred contact were scored.
    DECLARED_AND_INFERRED = "declared_and_inferred"
    #: Every scored contact was declared by the source.
    DECLARED_ONLY = "declared_only"
    #: Every scored contact was inferred from geometry.
    INFERRED_ONLY = "inferred_only"
    #: No contact was scored, so no provenance applies.
    NONE = "none"


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
    ``coordination.policy``, which asserts the values agree at import.
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


class DonorRuleOverride(StrEnum):
    """Why a donor forbidden by the inference rule was admitted anyway.

    Written to ``donor_rule_override``; blank when no override applied.
    """

    #: A source declaration named the contact, inside a supported residue class.
    DECLARED_CONNECTION = "declared_connection"


class InferredDonorRule(StrEnum):
    """Which donor-atom rule decided whether geometry alone may infer a donor.

    Written to ``inferred_donor_rule`` on every bond and candidate row.
    ``eligibility._inferred_donor_rule`` is the sole producer.
    """

    #: The oxygen of a modeled water.
    WATER_OXYGEN = "water_oxygen"
    #: The backbone carbonyl oxygen of an amino acid.
    BACKBONE_CARBONYL_OXYGEN = "backbone_carbonyl_oxygen"
    #: A side-chain atom this residue's typical donor list names.
    TYPICAL_SIDECHAIN_DONOR = "typical_sidechain_donor"
    #: The amine nitrogen of a modeled polymer N terminus.
    N_TERMINAL_NITROGEN = "n_terminal_nitrogen"
    #: A carboxylate oxygen of a modeled polymer C terminus.
    C_TERMINAL_OXYGEN = "c_terminal_oxygen"
    #: No rule admits this atom, so geometry alone may not infer a donor.
    OUTSIDE_TYPICAL_DONOR_LIST = "outside_typical_donor_list"


class ScoreExclusionReason(StrEnum):
    """Why an assigned contact contributes no geometry evidence.

    Written to ``score_exclusion_reason``; blank when the contact is scored.
    """

    ZSCORE_UNAVAILABLE = "zscore_unavailable"


class NeighborClass(StrEnum):
    """Coarse chemical class of a contact's donor residue."""

    WATER = "water"
    NUCLEOTIDE = "nucleotide"
    AMINO_ACID = "amino_acid"
    OTHER = "other"


#: Values two or more vocabularies spell identically on purpose, each mapped to
#: the names of the enums that share it. A comparison between members of two
#: listed enums is meaningful; any overlap absent from this table is a defect.
#: Case-only differences (``ScoreLevel.SUSPECT`` versus
#: ``GeometryStatus.SUSPECT``) are not overlaps: the strings differ, so the
#: members never compare equal, and the test that checks this table also
#: confirms no lower-case vocabulary ever spells an upper-case value.
SHARED_VALUES: dict[str, frozenset[str]] = {
    "valid": frozenset({"OccupancyStatus", "ElementStatus"}),
    "missing": frozenset({"OccupancyStatus", "ElementStatus", "ReferenceKind"}),
    "suspect": frozenset({"GeometryStatus", "MultiDonorStatus"}),
    "available": frozenset({"DensityContextStatus", "CrystallizationDataStatus"}),
    "density_only": frozenset({"EvidenceBasis", "ScoreInputStatus"}),
    "geometry_only": frozenset({"EvidenceBasis", "ScoreInputStatus"}),
    "no_assessable_evidence": frozenset({"EvidenceBasis", "VerdictReason"}),
    "other": frozenset({"ParentType", "NeighborClass"}),
    "manual": frozenset({"RunMode", "RefinementState"}),
    "none": frozenset({"ContactScope", "GeometryContactBasis"}),
}
