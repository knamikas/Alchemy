# Output schema

This document defines the row grain, identifiers, serialization, and columns
of Alchemy's CSV outputs. The ordered machine-enforced schemas live in
`src/driver/writers.py`, `src/coordination/schema.py`,
`src/metal_identification.py`, `src/confidence_score.py`, and
`src/crystallization_conditions.py`.

## Shared conventions

- `pdbID` is the normalized entry identifier used for grouping entries.
- `metal_site_id` identifies one selected coordinate-model metal atom. It is
  built from the PDB ID and zero-based model, chain, residue, and atom indices.
  It is the supported site join key across the site-level files. A diagnostic
  density row that could not be matched to a coordinate site has a blank ID.
- `contact_id` identifies one deposited donor atom in one explicit or generated
  image around a metal site. It is the supported join key between bond and
  candidate rows. The ID includes every symmetry component needed to distinguish
  images but uses a fixed-width digest to keep joins manageable.
- Booleans are serialized as `true` or `false`.
- An empty cell means an Alchemy-derived value was unavailable or inapplicable.
  The associated status or reason column should be used where provided.
- `nan` and infinities are never written for derived values. The raw EDSTATS
  block preserves EDSTATS' own `n/a` marker.
- Distances and coordinate values are in ångströms. Occupancies and coverage
  values are unitless. `resolution` and `dpi` are in ångströms, and
  `asu_volume` is in cubic ångströms. B-factor values are in square ångströms.
- `*_model_index`, `*_chain_index`, `*_residue_index`, and `*_atom_index` are
  zero-based coordinate-model indices. Author-facing chain, residue, insertion,
  atom, and alternate-location labels are retained separately.

## Interpret a result

Use this sequence to inspect one entry without treating missing evidence as a
negative result:

1. Find the entry in `manifest.csv`. Read `status`, `retryable`, `reason_codes`,
   and `status_detail` before you use its scientific rows. An `ok` entry
   completed all enabled stages. A `partial` entry has usable but incomplete
   output. A `skip` or `error` entry didn't produce a complete analysis.
2. Check `no_metals` and `metal_site_limit_exceeded`. If `no_metals=true`,
   Alchemy found no selected positive-occupancy metal sites. If
   `metal_site_limit_exceeded=true`, the entry was excluded from the standard
   cohort and has no site, bond, candidate, or confidence rows.
3. Read one row per selected site in `metal_sites_all.csv`. Use
   `metal_site_id` for site-level joins. For a multi-metal cofactor, use
   `density_observation_id` to avoid counting one residue-level EDSTATS
   observation more than once in density analyses.
4. Join assigned contacts from `metal_bonds_all.csv` by `metal_site_id`. Use
   `metal_contact_candidates_all.csv` when you need to audit rejected or
   unreferenced candidates. Join an exact contact between those files with
   `contact_id`.
5. If `confidence_scores_all.csv` exists, treat `alchemy_level` as the
   authoritative classification. `PASS` means all assessable components pass
   their raw thresholds. `REVIEW` means one assessable component needs review.
   `SUSPECT` means at least one component is suspect or both components need
   review. `INCOMPLETE` means neither component is assessable. The empirical
   `alchemy_score` ranks sites but doesn't determine the classification.
6. Use `review_queue_all.csv` as a triage view. Its crystallization fields add
   context but don't change confidence levels or scores.

An empty cell means unavailable or inapplicable, not zero or `false`. Read the
associated status or reason field before drawing a conclusion. For the exact
classification thresholds and decision matrix, see
[Database-referenced confidence scoring](method.md#database-referenced-confidence-scoring).

## `manifest.csv`

Grain: one outcome row per processed PDB entry. Read this file before the
scientific tables because it records whether the enabled stages completed and
whether the output is complete enough for your analysis.

| Columns | Meaning |
| --- | --- |
| `pdbID` | Normalized entry identifier and entry-level join key. |
| `status`, `retryable` | Entry outcome and whether an ordinary resume should retry it. |
| `no_metals`, `metal_site_limit_exceeded` | Successful metal-free result and standard-cohort exclusion flag. |
| `n_metals`, `n_bonds`, `n_candidates` | Selected coordinate sites, assigned contacts, and candidate rows. Blank bond or candidate counts mean that stage didn't run; `0` means it ran and found no rows. |
| `runtime_s` | Total entry runtime in seconds. |
| `reason_codes`, `warning_codes`, `status_detail` | Pipe-separated outcome reasons, non-status warnings, and a bounded human-readable explanation. |
| `alchemy_version`, `alchemy_commit`, `gemmi_version`, `ccp4_version` | Software provenance. |
| `reference_data_id`, `analysis_config_id` | Bundled-reference identity and analysis-policy identity used for compatibility checks. |
| `refinement_state`, `pdb_redo_is_twin`, `pdb_redo_version`, `pdb_redo_date` | PDB-REDO refinement and source provenance. |
| `source_coordinate_format`, `analysis_coordinate_format`, `coordinate_conversion_performed`, `source_coordinate_path` | Source-coordinate identity and any conversion performed for analysis. |
| `model_policy`, `input_model_count`, `model_analyzed`, `multi_model_structure` | Model-selection policy, available model count, selected model, and multi-model flag. |
| `altloc_policy`, `symmetry_contact_policy` | Alternate-conformer and generated-contact policies. |

Resume requires a compatible manifest schema, `reference_data_id`, and
`analysis_config_id`. For retry behavior and the complete reason-code
vocabulary, see the [operations guide](operations.md#resume-a-run).

## `density_context_all.csv`

Grain: exactly one row per processed entry. This is an entry-level calibration
artifact, not a metal-site table and not an authoritative input to the current
PASS/REVIEW/SUSPECT verdict.

`density_context_status` is `available` when EDSTATS completed and
`not_computed` otherwise. `edstats_residue_count` counts coherent residue
observations retained after alternate-conformer selection;
`target_residue_count` counts the metal and catalog-cofactor observations
excluded from the control distribution.

The prefixes ordinary_, ordinary_nonwater_, and water_ identify the full
non-target control, its non-water subset, and its water subset. Each prefix has
these fields:

| Suffix | Meaning |
| --- | --- |
| residue_count | Residue observations in the group, including rows whose `ZDa` is `n/a`. |
| rszd_count | Observations with a finite `ZDa`; this is the denominator of both fractions. |
| median_abs_rszd | Median absolute all-atom `ZDa`, blank when no finite values exist. |
| abs_rszd_ge_3_count, abs_rszd_ge_3_fraction | Count and fraction at the density REVIEW threshold. |
| abs_rszd_ge_6_count, abs_rszd_ge_6_fraction | Count and fraction at the density SUSPECT threshold. |

Waters provide a single-atom solvent-region comparison measured by the same
EDSTATS calculation. They are a contextual control rather than known-negative
examples; modeled-water selection and chemistry differ from metal sites.

## `metal_sites_all.csv`

Grain: one row per selected metal site with an EDSTATS density observation.
A multi-metal cofactor can repeat the same residue-level observation once for
each site. Density analyses must deduplicate `density_observation_id`; site
analyses must use `metal_site_id`.

### Raw EDSTATS block

`RT`, `CI`, `RN`, `MN`, `CP`, and `NR` are EDSTATS' residue type, output-group
chain identifier, residue number, model number, deposited chain part, and
one-based residue ordinal within that chain part. They are retained verbatim
so the Alchemy row remains auditable against `stats.out`.

The 36 metric columns are the Cartesian product of these metric stems and atom
groups:

| Stems | Meaning |
| --- | --- |
| `BA`, `NP`, `R`, `RG`, `SRG` | Raw EDSTATS B-factor, point-count, and real-space residual metrics. |
| `CCS`, `CCP`, `ZCCP` | Raw EDSTATS real-space correlation metrics and population Z-score. |
| `ZO`, `ZD`, `ZD-`, `ZD+` | Raw EDSTATS occupancy- and difference-density Z metrics. |

The suffixes are `m` for main-chain, `s` for side-chain, and `a` for all atoms.
For example, `ZD-m` is the main-chain negative difference-density Z metric and
`CCPa` is the all-atom population correlation. If an `NP` count exceeds
EDSTATS' fixed-width output field, Alchemy writes `n/a` for that count and adds
`edstats_grid_point_count_overflow` to the entry's `warning_codes`; the other
density metrics remain usable. `aa_geometry_coverage` is an
Alchemy compatibility field containing image-inclusive geometry coverage when
available and explicit-only coverage otherwise.

The concrete metric columns are:

- Main-chain: `BAm`, `NPm`, `Rm`, `RGm`, `SRGm`, `CCSm`, `CCPm`, `ZCCPm`,
  `ZOm`, `ZDm`, `ZD-m`, and `ZD+m`.
- Side-chain: `BAs`, `NPs`, `Rs`, `RGs`, `SRGs`, `CCSs`, `CCPs`, `ZCCPs`,
  `ZOs`, `ZDs`, `ZD-s`, and `ZD+s`.
- All atoms: `BAa`, `NPa`, `Ra`, `RGa`, `SRGa`, `CCSa`, `CCPa`, `ZCCPa`,
  `ZOa`, `ZDa`, `ZD-a`, and `ZD+a`.

### Site identity and density mapping

| Columns | Meaning |
| --- | --- |
| `metal_site_id` | Stable join key for the selected coordinate metal site. |
| `category` | `metal` for a single-atom ion or `cofactor` for a cataloged metal-containing residue. |
| `model_policy`, `input_model_count`, `model_analyzed`, `model_id`, `multi_model_structure` | Model-selection policy, deposited model count, selected model, its source identifier, and whether additional models existed. |
| `metal_model_index`, `metal_chain_index`, `metal_residue_index`, `metal_atom_index` | Unambiguous zero-based coordinate location used by `metal_site_id`. |
| `metal_resname`, `metal_chain`, `metal_resnum`, `metal_atom`, `metal_element`, `metal_icode`, `metal_altloc` | Human-readable deposited identity of the selected metal atom. |
| `metal_occupancy`, `metal_occupancy_valid`, `metal_occupancy_status`, `metal_coordinates_valid` | Selected atom occupancy and coordinate validation. |
| `metal_x`, `metal_y`, `metal_z` | Selected metal's Cartesian coordinates in the same frame as bond and candidate `transformed_neighbor_*` coordinates. |
| `metal_b_iso`, `entry_nonwater_median_b_iso` | Coordinate-model B factor of the selected metal and the entry median across canonical, non-water, non-hydrogen, non-zero-occupancy atoms. Unlike EDSTATS `BAa`, `metal_b_iso` remains atom-specific for a multi-atom cofactor. |
| `metal_special_position`, `metal_site_symmetry_order` | Whether crystallographic symmetry places a non-identity image within 0.8 Å of the metal, and the resulting site-symmetry order (coincident non-identity images plus one). General positions report `false` and order 1. Strict-NCS operations are excluded. |
| `metal_expected_crystallographic_occupancy`, `metal_occupancy_matches_site_symmetry` | Symmetry-only occupancy expectation, `1 / metal_site_symmetry_order`, and whether the modeled occupancy agrees within an absolute tolerance of 0.015. Invalid symmetry or coordinates leave all four special-position fields blank; an invalid occupancy leaves only the agreement field blank. These are contextual fields and do not automatically exempt a site from scoring. |
| `donor_b_iso_count`, `donor_median_b_iso` | Number of assigned contacts with a finite donor B factor and their median. These use the primary contact scope, image-inclusive when symmetry search is available. |
| `metal_donor_b_ratio`, `metal_minus_donor_b_iso` | Directional metal-to-donor-median ratio and metal minus donor-median difference. Ratio is blank unless both values are positive. |
| `metal_donor_b_similarity` | Symmetric local agreement, `exp(-abs(ln(metal_b_iso / donor_median_b_iso)))`; 1 means equal. This median-based Alchemy measure is not CheckMyMetal's bond-valence-weighted environmental B score. |
| `nearest_metal_distance`, `nearest_metal_element`, `nearest_metal_site_id` | Nearest other canonical metal in the analyzed coordinate model. Distance is blank when none is assessable; the ID can name a coordinate metal without its own mapped density row. |
| `nearby_metal_count_6a` | Other canonical metals within 6 Å in the analyzed coordinate model. This is proximity context, not a metal-metal bond or multinuclear-site assignment. Crystallographic symmetry images are excluded. |
| `metal_conformer_mean_occupancy`, `metal_altloc_options`, `alternative_conformers_present`, `altloc_selection_fallback` | Residue conformer evidence and whether selection required a fallback. |
| `density_observation_id` | Join key for the underlying EDSTATS observation; not a metal-site key. |
| `density_scope`, `density_shared_site_count`, `density_is_shared` | Whether density describes an ion or whole cofactor residue and how many selected sites share it. |
| `coordinate_mapping_status`, `selected_metal_site_status` | Outcome of joining an EDSTATS residue to the selected coordinate site. |

### Resolution, DPI, and contact summaries

| Columns | Meaning |
| --- | --- |
| `dpi`, `resolution`, `dpi_unavailable_reason` | Diffraction precision index, input resolution, and the reason DPI could not be calculated. |
| `r_free`, `reflection_count`, `asu_volume` | Final free R factor, observed-reflection count, and asymmetric-unit volume used by the DPI calculation. Valid components remain available when another missing input prevents calculation of DPI. |
| `occupancy_weighted_atom_count`, `deposited_occupancy_weighted_atom_count`, `dpi_atom_count_multiplier` | Atom-count inputs and multiplier used by the DPI calculation. |
| `strict_ncs_operation_count`, `crystallographic_operation_count` | Numbers of generated operations available to the contact search. |
| `candidate_contact_count`, `reference_covered_contact_count` | Assigned contacts and the subset covered by the distance reference. |
| `geometry_outlier_contact_count`, `geometry_consistent_contact_count` | Reference-covered contacts classified before score exclusions. |
| `score_eligible_contact_count`, `score_excluded_contact_count` | Contacts admitted to or excluded from confidence geometry scoring. |
| `scored_geometry_outlier_contact_count`, `scored_geometry_consistent_contact_count` | Score-eligible contacts in each geometry class. |
| `multi_donor_residue_group_count`, `multi_donor_contact_count` | Chelating residue groups and contacts belonging to those groups. |
| `suspect_multi_donor_residue_group_count`, `indeterminate_multi_donor_residue_group_count` | Multi-donor groups with suspect or unassessable geometry. |
| `context_warning`, `context_warning_reasons` | Whether non-scoring structural context warrants attention and pipe-separated reason codes. |
| `non_typical_first_sphere_candidate_count`, `declared_donor_override_contact_count` | Non-typical nearby donors and declared contacts admitted by an explicit override. |
| `explicit_contact_count`, `symmetry_contact_count`, `image_inclusive_contact_count` | Contact counts before and after generated-image search. |
| `crystallographic_contact_count`, `strict_ncs_contact_count`, `combined_ncs_crystallographic_contact_count` | Generated contacts classified by operation source. |
| `geometry_outlier_count_explicit`, `geometry_outlier_count_image_inclusive` | Outlier counts under the two search scopes. |
| `geometry_coverage_explicit`, `geometry_coverage_image_inclusive` | Reference coverage under the two search scopes. |
| `explicit_geometry_status`, `image_inclusive_geometry_status` | Site classification under each search scope. |
| `generated_contact_scope` | Which generated-image sources contribute contacts. |
| `geometry_classification_changes_with_generated_images` | Whether generated contacts change the site classification. |
| `coordination_depends_on_crystallographic_symmetry`, `coordination_depends_on_strict_ncs` | Whether each generated-image source contributes to reported coordination. |
| `geometry_not_assessed_reason` | Pipe-separated reasons geometry could not be assessed. |
| `zscore_outlier_cutoff` | Absolute bond-distance Z threshold used for outlier classification. |

### Structure-validation provenance

| Columns | Meaning |
| --- | --- |
| `symmetry_search_available`, `symmetry_search_failure_reason` | Whether generated-image search completed and why it did not. |
| `occupancy_validation_failed`, `missing_occupancy_count`, `invalid_occupancy_count` | Entry-level occupancy parsing and validation results. |
| `overfull_occupancy_site_count`, `overfull_occupancy_excess`, `metal_overfull_occupancy` | Alternate-conformer occupancy excess and whether it affects this metal site. |
| `defaulted_occupancy_atom_count`, `zero_occupancy_atom_count` | Counts of defaulted and explicitly absent atoms. |
| `duplicate_atom_records_present`, `duplicate_atom_record_count`, `duplicate_atom_coordinate_conflict_count`, `malformed_duplicate_atom_name_count` | Duplicate-coordinate-record diagnostics. |
| `raw_occupancy_mapping_failed`, `raw_occupancy_mapping_failure_reason` | Whether raw occupancy records could be mapped to parsed atoms. |
| `unknown_element_atom_count`, `element_validation_warning` | Missing or suspect deposited element assignments. |
| `non_finite_coordinate_atom_count` | Atoms excluded because a coordinate was not finite. |

## `metal_bonds_all.csv`

Grain: one assigned inferred or source-declared metal–donor contact. Every
`contact_id` must occur once in `metal_contact_candidates_all.csv` with
`assigned_as_bond=true`.

| Columns | Meaning |
| --- | --- |
| `pdbID`, `metal_site_id`, `contact_id` | Entry, metal-site join key, and contact join key. |
| `metal_resname`, `metal_chain`, `metal_resnum`, `metal_element`, `metal_atom`, `metal_icode`, `metal_altloc` | Human-readable metal identity. |
| `neighbor_resname`, `neighbor_chain`, `neighbor_resnum`, `neighbor_atom`, `neighbor_element`, `neighbor_icode`, `neighbor_altloc` | Human-readable donor identity. |
| `neighbor_b_iso` | Coordinate-model isotropic or equivalent-isotropic B factor of the assigned donor atom. |
| `distance` | Measured metal–donor distance. |
| `coordination_status`, `coordination_source`, `declared_connection` | Whether assignment came from a declaration or inference and its source. |
| `connection_id`, `connection_type`, `connection_link_id`, `connection_asu`, `connection_reported_distance` | Pipe-aligned source-declaration records; blank for inference-only contacts. |
| `inferred_donor_allowed`, `inferred_donor_rule`, `donor_rule_override` | Donor-chemistry decision and any declaration override. |
| `context_warning`, `context_warning_reasons` | Non-scoring warning and pipe-separated reasons. |
| `literature_distance`, `literature_stdev`, `reference_covered` | Reference mean, spread, and coverage status. |
| `zscore`, `zscore_outlier_cutoff`, `geometry_outlier`, `geometry_consistent` | DPI-aware distance score, threshold, and classification. |
| `dpi`, `resolution`, `sigma_mag`, `sigma_neg`, `sigma_pos` | Precision and site-density inputs used for analysis. |
| `parent_type`, `bonded_to`, `neighbor_class` | Cofactor/protein context and broad donor class. |
| `model_id`, `metal_model_index`, `metal_chain_index`, `metal_residue_index`, `metal_atom_index` | Selected model and unambiguous metal location. |
| `neighbor_model_index`, `neighbor_chain_index`, `neighbor_residue_index`, `neighbor_atom_index` | Unambiguous deposited donor location. |
| `metal_occupancy`, `metal_occupancy_valid`, `metal_occupancy_status`, `metal_conformer_mean_occupancy`, `metal_altloc_options`, `metal_altloc_selection_fallback` | Metal occupancy and conformer provenance. |
| `neighbor_occupancy`, `neighbor_occupancy_valid`, `neighbor_occupancy_status`, `neighbor_conformer_mean_occupancy`, `neighbor_altloc_options`, `neighbor_altloc_selection_fallback` | Donor occupancy and conformer provenance. |
| `alternative_conformers_present`, `altloc_selection_fallback` | Combined metal/donor conformer flags. |
| `multi_donor_detected`, `multi_donor_contact_count`, `multi_donor_geometry_status`, `multi_donor_contains_suspect_bond` | Chelating-residue grouping and geometry. |
| `score_eligible`, `score_exclusion_reason` | Whether this contact contributes to confidence geometry and why not. |
| `contact_scope`, `symmetry_contact`, `crystallographic_contact`, `strict_ncs_contact`, `strict_ncs_operation_id` | Explicit/generated-image classification and strict-NCS provenance. |
| `symmetry_image_index`, `symmetry_operation`, `cell_translation_x`, `cell_translation_y`, `cell_translation_z` | Crystallographic image provenance. |
| `transformed_neighbor_x`, `transformed_neighbor_y`, `transformed_neighbor_z` | Donor coordinates in the image used for the measured distance. |

When several source declarations describe the same contact, the five
`connection_*` fields contain pipe-separated values in matching order.

## `metal_contact_candidates_all.csv`

Grain: one donor-like atom found by the broad 4 Å search or supplied by a source
declaration. Candidate discovery does not itself assign a bond.

| Columns | Meaning |
| --- | --- |
| `pdbID`, `metal_site_id`, `contact_id` | Entry, metal-site join key, and contact join key. |
| `assigned_as_bond` | Whether this exact candidate survives all filters and special-position deduplication into `metal_bonds_all.csv`. |
| `candidate_source` | Pipe-separated discovery sources such as proximity search and source declaration. |
| `eligibility_status`, `eligibility_reason`, `first_sphere_eligible` | First-sphere decision, machine-readable reason, and distance/reference result. |
| `candidate_distance` | Measured metal–candidate distance. |
| `assignment_target`, `assignment_tolerance`, `first_sphere_cutoff` | Reference target, fixed tolerance, and resulting assignment cutoff. |
| `assignment_reference_kind`, `assignment_reference` | Exact, fallback, or missing cutoff reference and its identity. |
| `inferred_contact_eligible`, `inferred_donor_allowed`, `inferred_donor_rule`, `donor_rule_override` | Final inference eligibility, chemical donor policy, and declaration override. |
| `context_warning`, `context_warning_reasons` | Non-scoring warning and pipe-separated reasons. |
| `coordination_status`, `coordination_source`, `declared_connection` | Candidate-level declaration or inference provenance; use `assigned_as_bond`, not this status, for bond membership. |
| `connection_id`, `connection_type`, `connection_link_id`, `connection_asu`, `connection_reported_distance` | Pipe-aligned source-declaration records. |
| `metal_resname`, `metal_chain`, `metal_resnum`, `metal_element`, `metal_atom`, `metal_icode`, `metal_altloc`, `metal_occupancy` | Human-readable metal identity and occupancy. |
| `model_id`, `metal_model_index`, `metal_chain_index`, `metal_residue_index`, `metal_atom_index` | Selected model and unambiguous metal location. |
| `neighbor_resname`, `neighbor_chain`, `neighbor_resnum`, `neighbor_atom`, `neighbor_element`, `neighbor_icode`, `neighbor_altloc`, `neighbor_occupancy`, `neighbor_class` | Human-readable candidate identity, occupancy, and broad class. |
| `neighbor_b_iso` | Coordinate-model isotropic or equivalent-isotropic B factor of the candidate atom. |
| `neighbor_model_index`, `neighbor_chain_index`, `neighbor_residue_index`, `neighbor_atom_index` | Unambiguous deposited candidate location. |
| `contact_scope`, `symmetry_contact`, `crystallographic_contact`, `strict_ncs_contact`, `strict_ncs_operation_id` | Explicit/generated-image classification and strict-NCS provenance. |
| `symmetry_image_index`, `symmetry_operation`, `cell_translation_x`, `cell_translation_y`, `cell_translation_z` | Crystallographic image provenance. |
| `transformed_neighbor_x`, `transformed_neighbor_y`, `transformed_neighbor_z` | Candidate coordinates in the image used for the measured distance. |

`coordination_status` describes the candidate's evidence before all final bond
filters. It is deliberately not an alias for `assigned_as_bond`.

## `crystallization_conditions_all.csv`

Grain: one row per deposited crystallization-condition record. Conditions are
entry-level experimental context and join to site, bond, and confidence rows by
`pdbID`. They are not confidence inputs.

| Columns | Meaning |
| --- | --- |
| `pdbID`, `crystallization_condition_id` | Entry join key and stable within-entry condition identifier. |
| `source_format` | `json` for a cached RCSB Data API record, `mmcif` for coordinate-file `_exptl_crystal_grow`, or `pdb` for legacy `REMARK 280`. |
| `metadata_source`, `metadata_retrieved_at_utc`, `entry_revision_date` | The selected source (`rcsb_data_api`, `pdb_redo_coordinate_file`, or `manual_coordinate_file`), API retrieval time when applicable, and deposited entry revision date supplied by RCSB. |
| `crystal_id` | Deposited crystal identifier when reported. |
| `method` | Reported method; for legacy PDB it is conservatively recognized from the remark text. |
| `pH`, `pH_range` | Deposited pH value or range, preserved separately. |
| `temperature_K`, `temperature_details` | Deposited temperature and its free-text qualification. mmCIF temperatures are in kelvin. |
| `raw_details` | Original deposited condition details: the auditable source for normalized summary flags. |

## `crystallization_summary_all.csv`

Grain: exactly one contextual row per processed manifest entry. A status of
`available` means a condition record was found, `not_reported` means the source
contained no condition record, `unparseable` means extraction failed, and
`input_unavailable` means entry preparation supplied no source file.

| Columns | Meaning |
| --- | --- |
| `pdbID`, `crystallization_data_status`, `crystallization_condition_count`, `crystallization_source_format`, `crystallization_condition_ids` | Join key, availability, serialization format, and links to canonical condition rows. |
| `crystallization_metadata_source`, `crystallization_metadata_retrieved_at_utc`, `crystallization_entry_revision_date` | Selected metadata source and its API retrieval/revision provenance. These remain populated for a valid cached record whose condition list is empty. |
| `crystallization_pH_min`, `crystallization_pH_max`, `crystallization_temperature_min_K`, `crystallization_temperature_max_K` | Entry ranges across reported records. |
| `crystallization_raw_text` | Distinct raw descriptions joined with ` || `. |
| `crystallization_detected_metals`, `crystallization_any_metal` | Sorted pipe-separated explicit metal detections and their summary flag. |
| `crystallization_promiscuous_transition_metal`, `crystallization_ni_co_like_metal`, `crystallization_buffer_light_metal`, `crystallization_heavy_additive_phasing_metal` | Positive-evidence chemical-class flags used for contextual analysis. |
| `crystallization_sulfate`, `crystallization_cacodylate`, `crystallization_acetate` | Positive-evidence ingredient flags discussed in the manuscript. |

When `crystallization_data_status` is not `available`, detection flags are
blank rather than `false`. A missing deposited record must not be interpreted
as proof that a reagent was experimentally absent.

## `confidence_inputs_all.csv`

Grain: one compact evidence row per manifest-counted selected metal site during
an uncapped database run. The file is retained so the score and frozen
reference can be reproduced without rerunning CCP4. Later targeted runs do not
create a database cohort; when a reference is installed, their prepared inputs
are embedded directly in `confidence_scores_all.csv`.

| Columns | Meaning |
| --- | --- |
| `pdbID`, `metal_site_id` | Entry and supported join key to `metal_sites_all.csv` and `metal_bonds_all.csv`. |
| `category` | `metal` or `cofactor`; blank only when an unresolved site cannot be classified. |
| `density_observation_id`, `density_scope`, `density_shared_site_count`, `density_is_shared` | Density-observation join key, measurement scope, multiplicity, and shared-observation flag. |
| `coordinate_mapping_status`, `selected_metal_site_status` | Whether density and coordinate evidence were resolved for the selected site. |
| `metal_model_index`, `metal_chain_index`, `metal_residue_index`, `metal_atom_index` | Unambiguous zero-based coordinate location. |
| `metal_resname`, `metal_chain`, `metal_resnum`, `metal_atom`, `metal_element`, `metal_icode`, `metal_altloc` | Human-readable deposited metal-site identity. |
| `rszd`, `rszd_abs`, `rszd_negative`, `rszd_positive`, `density_saturated` | Raw metal-site RSZD, its absolute magnitude, signed negative/positive difference-density statistics, and the EDSTATS saturation flag. |
| `assigned_contact_count`, `reference_covered_contact_count`, `geometry_bond_count` | Assigned contacts, contacts covered by the literature reference, and finite score-eligible contacts used for RMS geometry. |
| `geometry_coverage` | Literature-reference-covered contacts divided by all assigned contacts. This is a coverage annotation and does not modify the score or level. |
| `geometry_rms_zbond`, `geometry_max_abs_zbond`, `geometry_mean_abs_zbond`, `geometry_mean_signed_zbond` | Primary RMS geometry statistic and supporting Zbond diagnostics. |
| `worst_bond`, `worst_bond_source` | Contact ID and declared/inferred source of the largest absolute scored Zbond. |
| `worst_bond_neighbor_resname`, `worst_bond_neighbor_chain`, `worst_bond_neighbor_resnum`, `worst_bond_neighbor_atom` | Human-readable donor identity for `worst_bond`. |
| `declared_contact_count`, `inferred_contact_count`, `declared_scored_bond_count`, `inferred_scored_bond_count`, `geometry_contact_basis` | Coordination provenance before and after score eligibility. |
| `multi_donor_contact_count`, `suspect_multi_donor_residue_group_count` | Chelation context retained as a non-scoring diagnostic. |
| `context_warning`, `context_warning_reasons` | Interpretive warning carried into the score output without changing the score. |
| `confidence_inputs_status`, `confidence_inputs_missing_reasons` | Evidence completeness and pipe-separated reasons for missing or partial evidence. |

`confidence_inputs_status` is `complete`, `density_only`, `geometry_only`, or
`unscorable`. Density and geometry are independently available: one missing
component never prevents the other from determining the site level. Partial
geometry coverage remains explicit in `confidence_inputs_missing_reasons`, but
does not weaken or strengthen finite geometry evidence.

## `confidence_scores_all.csv`

Grain: one row per confidence input. All `confidence_inputs_all.csv` columns are
preserved as the leading block, followed by these analysis columns:

| Columns | Meaning |
| --- | --- |
| `density_level`, `geometry_level` | Raw-threshold component verdicts: `PASS`, `REVIEW`, `SUSPECT`, or `INCOMPLETE`. |
| `density_score`, `geometry_score` | Reverse average-rank empirical support scores from 0 to 100; higher means more ordinary relative to the frozen component cohort. Blank when no compatible reference or component measurement is available. |
| `alchemy_level` | Authoritative non-compensatory site verdict. Any SUSPECT component, or REVIEW in both components, makes the site SUSPECT. |
| `alchemy_score` | Minimum available component support score for ranking only. It does not define `alchemy_level`. |
| `evidence_basis` | `density_and_geometry`, `density_only`, `geometry_only`, or `no_assessable_evidence`. |
| `verdict_reason` | Machine-readable decision route, including `density_suspect`, `geometry_suspect`, `density_and_geometry_suspect`, and `review_plus_review`. |
| `score_policy_version` | Version of the raw-threshold and verdict-matrix policy. |
| `confidence_reference_version` | Identity of the compatible pair of frozen component distributions; blank for classification-only output. |
| `confidence_cohort_id` | Identity of the exact confidence-input artifact that produced the reference cohort. |
| `confidence_cohort_size` | Number of site rows in the frozen input cohort. |
| `density_reference_size`, `geometry_reference_size` | Assessable observations in each empirical component distribution. |

Support scores are published to six decimal places. Raw component values define
the levels; neither a score nor a population percentile can move a site across
a PASS/REVIEW/SUSPECT boundary.

Each component cohort is weighted per assessable metal site, not per structure.
This is the appropriate interpretation for a site-level empirical rank, but it
must not be mistaken for a structure-weighted statistic.

## `review_queue_all.csv`

Grain: one row per confidence row whose authoritative `alchemy_level` is
`REVIEW` or `SUSPECT`. All confidence-score columns are preserved, followed by
the crystallization-summary columns except the repeated `pdbID`, plus:

| Columns | Meaning |
| --- | --- |
| `crystallization_contains_modeled_metal` | Whether the row's `metal_element` was detected in the entry condition. Blank when condition data are unavailable. |
| `crystallization_contains_different_promiscuous_transition_metal` | Whether another Mn, Fe, Co, Ni, Cu, Zn, or Cd was detected. Blank when condition data are unavailable. |
| `crystallization_context_flags` | Pipe-separated positive findings for rapid review. |

The queue is a derived convenience view. Its membership is determined before
the crystallization join, and none of its condition columns changes a
confidence component, score, level, evidence basis, or verdict reason.

## `confidence_reference/`

The frozen reference is portable only as the following pair of files. The
metadata is its completion marker; Alchemy removes it before rebuilding so a
failed finalization cannot leave an older reference looking current.

### `component_distributions.csv`

| Columns | Meaning |
| --- | --- |
| `component` | `density` or `geometry`. |
| `value` | Raw absolute RSZD or site-level RMS Zbond value. |
| `count` | Assessable cohort sites with that component value. |

### `metadata.json`

The scoring contract is recorded by `confidence_method_version`,
`confidence_schema_version`, `score_decimal_places`, `metric_decimal_places`,
`density_thresholds`, `density_saturation_value`,
`density_saturation_policy`, `geometry_thresholds`, `geometry_statistic`,
`overall_rule`, `support_score_method`, `coverage_policy`,
`input_status_policy`, `cohort_weighting`, `maximum_entry_metal_sites`, and
`reference_data_id`. `analysis_config_id` additionally binds the model,
alternate-conformer, symmetry, cohort-limit, and bundled-reference policies.
Alchemy refuses to load a reference whose contract differs from the running
code.

The distributions are described by `reference_id`, `distribution_file`,
`density_distinct_value_count`, `geometry_distinct_value_count`,
`density_reference_size`, and `geometry_reference_size`. The source cohort is
described separately by `cohort_id`, `confidence_inputs_file`,
`confidence_inputs_sha256`, `input_row_count`, `input_entry_count`,
`scorable_entry_count`, and `input_status_counts`.

When database finalization receives the run manifest, metadata additionally
contains `source_manifest_file`, `source_manifest_sha256`, `source_entry_count`,
`manifest_status_counts`, `no_metals_entry_count`,
`metal_site_limit_exceeded_entry_count`, `metal_bearing_entry_count`, and
`software_versions`. `analysis_config_id` is copied from the manifest and
identifies the model, alternate-conformer, symmetry, cohort-limit, and
bundled-reference policies. Execution-only choices such as paths, worker count,
optional stage selection, map-cropping scope, logging, caching, and timeouts are
excluded from that identity. Hashes identify exact artifacts; the reference ID
and cohort ID deliberately answer different questions.
