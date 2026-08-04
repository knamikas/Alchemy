# Method

What Alchemy computes, stage by stage, and the limits of each answer. This is
the reference for interpreting a result: which contacts become bonds, which
donors have literature references and which do not, and what the confidence
score does and does not claim.

## Pipeline flow

### 1. Input preparation — `src/inputs.py` and `src/coordinate_conversion.py`

The automatic PDB-REDO workflow analyzes only the final re-refined and rebuilt
model. It prefers `{id}_final.cif` as the authoritative coordinate source and
converts it to an EDSTATS-compatible analysis PDB with gemmi. The
`{id}_final.pdb` compatibility export is used only when mmCIF is unavailable.
Alternative coordinate/MTZ pairs can still be supplied through the manual-file
options; when both coordinate formats are supplied, mmCIF takes precedence.

During mmCIF conversion, `.` and `?` occupancy values are written as blank PDB
occupancies rather than being replaced by `1.00`. Alchemy also embeds a
reversible mapping for component identifiers that exceed the three-character
legacy PDB residue-name field. The original CCD identifier is restored after
EDSTATS so cofactor catalog matching and output retain the mmCIF identity.

If a model has more chains than the one-character PDB namespace can represent,
Alchemy packs its residues into synthetic one-character chains with unique
four-column sequence numbers. ``REMARK 950 ALCHEMY RESIDUE`` records preserve
the original component, chain, sequence number, insertion code, source traversal
indices, and polymer-terminal position. EDSTATS and Gemmi analyze the same
packed coordinates, then statistics, contacts, declarations, and CSV identifiers
are mapped back to the source mmCIF identities. Conversion validates the atom
and residue membership before analysis, so an oversized structure cannot
silently lose sites at the legacy-PDB boundary.

The overall diffraction resolution comes from PDB-REDO `data.json` when
available, with an MTZ fallback through gemmi. EDSTATS instead receives the
common finite resolution range of `FWT`, `PHWT`, `DELFWT`, and `PHDELWT`, which
are the columns used to calculate its two maps.

### 2. Maps and real-space statistics — `src/density_analysis.py`

`run_density_analysis()` runs:

1. CCP4 `mtzfix` to check and, when needed, correct the Fourier map
   coefficients, including their centric and acentric consistency. If the input
   passes, MTZFIX intentionally writes no replacement and the original MTZ is
   used. If its consistency re-test fails for an entry whose PDB-REDO
   `data.json` explicitly has `properties.ISTWIN=true`, Alchemy can instead
   normalize recognizable Refmac composite coefficients on a temporary MTZ.
   This guarded path requires Refmac provenance, the complete expected column
   schema, and reflection-by-reflection agreement with Refmac's raw coefficient
   identity; its output is independently checked against the convention
   EDSTATS consumes. The source MTZ is never modified. Successful use is
   recorded as the `twin_refmac_coefficients_normalized` warning.
2. CCP4 `fft` with `FWT/PHWT` from that validated MTZ to produce a 2mFo-DFc map.
3. By default, CCP4 `mapmask` limits that full FFT map to the complete deposited
   coordinate-model envelope plus a 10 Angstrom border. The same crop is then
   applied to the mFo-DFc difference map calculated from `DELFWT/PHDELWT`.
   This retains every modeled atom while avoiding EDSTATS work over distant
   empty unit-cell volume. If the envelope would not be smaller, Alchemy uses
   the original full maps automatically. It also uses full maps when a
   translated model would produce a grid extent known to be incompatible with
   EDSTATS coordinate lookup.
4. CCP4 `edstats` uses the resulting map pair to produce per-residue real-space
   statistics and an RSZD coordinate file.

Use `--density-map-scope full` to select the legacy behavior in which EDSTATS
receives both complete unit-cell maps without model-envelope cropping.

The MTZ input must contain `FWT`, `PHWT`, `DELFWT`, and `PHDELWT` columns.

### 3. Metal and cofactor identification — `src/metal_identification.py`

`extract_metal_statistics()` parses the edstats table. Alchemy reads the
deposited PDB element field directly; blank or invalid fields are marked unknown
rather than inferred from atom names. During mmCIF conversion, the explicit
`_atom_site.type_symbol` is written into that PDB field. Metal-containing
cofactors are matched against the Chemical Component Dictionary list maintained
by `tools/build_metallocofactor_catalog.py`. A structure with unknown elements
does not receive a DPI because its non-hydrogen atom count is indeterminate.
EDSTATS' missing-chain markers are normalized before coordinate matching. When
EDSTATS omits its empty trailing chain field for a blank-chain residue, Alchemy
restores that field before validating the standard 42-column schema. Other row
width mismatches still fail. Decimal and hybrid-36 PDB residue numbers are
decoded to the same canonical integer representation used by Gemmi before raw
PDB atoms and EDSTATS rows are joined. If coordinate residues repeat the same
author identity, the one-based EDSTATS `NR` coordinate-residue ordinal resolves
them one-to-one; a duplicate, out-of-range, or inconsistent `NR` fails the entry
rather than expanding an ambiguous row across multiple sites. Completeness is
checked with residue multiplicity intact. The table must contain finite numeric
statistics or the documented `n/a` marker and a row for every selected metal or
cofactor residue. Empty, malformed, incomplete, or wrong-model output fails the
entry instead of being written to the aggregate CSV.

### 4. Bond-distance analysis — `src/bond/`

Gemmi parses the same coordinate representation supplied to EDSTATS. For a
multi-model structure, Alchemy analyzes the first model only for metal
identification, DPI, and contact searching; model count and the selected-model
policy are recorded with the results. Before map statistics are calculated,
Alchemy writes a wrapper-free, first-model-only PDB and supplies that exact
coordinate file to both EDSTATS and Gemmi. Removing explicit MODEL/ENDMDL
records prevents EDSTATS from adding a synthetic separator residue. Atoms and
density statistics from different models are never combined.

Alternative conformations are selected coherently per residue. Blank-altloc
atoms are shared, while the named conformer with the highest mean valid atomic
occupancy is selected (ties are resolved by altloc label). This avoids creating
an artificial residue by choosing A/B alternatives independently for each atom.
Every neighboring residue is considered independently, and the selected and
available alternatives are recorded.

`run_bond_analysis()` uses a 4 Å search only to discover broad
positive-occupancy N/O/S candidates around a configured metal, outside the
metal's own residue, in a recognized amino acid or water. Discovery does not
assign a candidate as a bond. A separate eligibility stage identifies likely
first-coordination-sphere candidates for the current bond output, and an
atom-level chemical rule determines which candidates Alchemy may infer as
bonds. Following [Harding's
coordination-group
definition](https://doi.org/10.1107/S0907444904004081), the upper limit is the
target metal-donor distance plus 0.75 Å. If the exact residue-specific reference
is absent, the largest target for the same metal and donor element is used only
for sphere membership. A pair with no such target is retained as candidate
evidence but is not inferred as a contact; the entry reports
`missing_first_sphere_reference`. DPI never expands the chemical cutoff. Atoms
belonging to the metal's own cofactor residue remain excluded, so these rows
describe external coordination rather than a complete cofactor coordination
number.

The geometry-inference donor table covers all 20 standard amino acids. Backbone
carbonyl `O` is allowed for every residue. Typical side-chain donors are ASN
OD1; ASP OD1/OD2; CYS SG; GLN OE1; GLU OE1/OE2; HIS ND1/NE2; LYS NZ; MET SD;
SER OG; THR OG1; and TYR OH. Water oxygen is allowed. Polymer N-terminal `N`
and C-terminal `OXT`/`OT1`/`OT2` are allowed only when deposited sequence
provenance identifies the residue at the corresponding polymer boundary.
For converted mmCIF this uses `label_seq_id` and the complete entity sequence;
for direct PDB input it requires a complete `SEQRES` sequence matching the
modeled polymer. A first or last modeled residue is not by itself treated as a
terminus, so missing or disordered endpoint residues cannot create terminal
donors. Other proximal N/O/S atoms are
retained in `metal_candidates_all.csv`, marked
`inferred_donor_allowed=false`, and cannot become geometry-inferred bonds.
This includes internal peptide N, ASN/GLN amide N, TRP pyrrole N, and ARG
guanidinium N. A declaration can still establish such an atom as a declared
bond; `donor_rule_override=declared_connection` makes that exception explicit.

### Reference coverage of the donor table

Inferring a contact and scoring one are separate questions. The donor table
above governs inference; scoring additionally requires a literature reference
distance in `src/data/metal_distances_info.txt`, and that reference does not
cover every donor Alchemy will infer:

- **ASN, GLN, LYS and MET have no reference entry.** Harding (2006) tabulates
  water `O`, ASP/GLU carboxylate `O`, backbone carbonyl `O`, HIS `N` and CYS
  `S` only. Contacts to these four side chains are therefore discovered,
  reported and measured, but never receive a Zbond: they carry the
  same-element fallback and NaN derived values. This is a limitation of the
  bundled reference data, not of the geometry.
- **SER, THR and TYR values are approximations** derived from statements in
  Harding (2006) rather than from its tables, so their `sigma_lit` is not an
  empirical spread. Treat z-scores for these three donors as indicative.
- **Nucleic acids, modified residues and other ligands have no reference at
  all.** A metal coordinated by, say, a DNA phosphate oxygen is real
  coordination, but no bundled distance can assess it.

A declared contact to a donor class with no reference is retained in
`metal_candidates_all.csv` with its measured distance and full connection
provenance, and the entry records
`declared_donor_outside_supported_classes`. It is deliberately **not** promoted
to a bond row: doing so would raise the site's coordination count and enlarge
the confidence geometry-coverage (`QG`) denominator on the strength of a
contact that nothing in the reference data can evaluate. The distinction that
matters for a consumer is that "no reference for this donor class" and "this
metal has no coordination" are now different, visibly, in the output.

Alchemy separately parses `_struct_conn` records from the authoritative source
mmCIF and `LINK` records from a source PDB, in
`src/bond/declared_connections.py`. A declared metal–donor contact is
merged with a matching proximity candidate when present and added independently
when it lies outside 4 Å. Declared contacts remain bonds even when they fail the
distance-based first-sphere eligibility rule; their measured distance and Zbond
are still calculated whenever the required literature reference and DPI exist.
The output labels these as model declarations rather than treating them as
geometric proof of coordination.

Alchemy reports assigned contacts to atoms explicitly present in the
analyzed model and contacts generated by crystallographic symmetry, strict NCS,
or a combination of the two. Image-inclusive geometry is the primary result,
while explicit-only counts and geometry are retained separately. Generated
rows independently record whether an NCS transform and a crystallographic
operation contributed. They also record the strict-NCS operation identifier,
Gemmi image index, symmetry code, and unit-cell translation. Near-coincident
images of the same deposited atom within Gemmi's 0.8 Å special-position cutoff
are collapsed, while the stricter 0.001 Å tolerance remains reserved for
conflicting duplicate coordinate records.

Where a literature reference is available, Alchemy calculates:

```text
z = (d_observed - mu) / sqrt(DPI^2 + sigma_lit^2)
```

A bond length depends on two atomic positions, but a single DPI enters the
denominator rather than the `sqrt(2) * DPI` an independent-error treatment of
both atoms would give. This assumes the metal is well enough ordered that its
positional uncertainty is negligible beside that of the lighter donor atom, so
the DPI term stands for the donor alone. Using two atoms' worth of uncertainty
would divide every z-score by up to `sqrt(2)` and reclassify borderline geometry
outliers.

Reference-covered contacts with `|Zbond| >= 6` are geometry outliers.
First-sphere contacts admitted by a same-element fallback, or without complete
DPI inputs, are still emitted with their measured geometry and NaN derived
values. The `geometry_outlier` and `geometry_consistent` columns are nullable
booleans: a blank value means that geometry was not assessed, not that the
contact passed or failed the cutoff.

Assigned contacts are also grouped by metal and donor-residue image, with no
upper limit on the number of contacts in a group. Because backbone atoms carry
the same residue identity in the coordinate model, this grouping automatically
includes backbone–side-chain combinations without a separate backbone rule.
Every bond retains its measured distance, Zbond, and ordinary geometry flag.
Groups with two or more donors set `multi_donor_detected=true` and record their
full contact count. Every assessable member contributes to aggregate scoring
normally. If any member is an outlier, every member of the group records
`multi_donor_geometry_status=suspect` and
`multi_donor_contains_suspect_bond=true`; the particular unusual bonds retain
`geometry_outlier=true`. This makes possible multidentate context conspicuous
without weakening or excluding the result. A group with unavailable Zbond
values and no detected outlier is labeled `indeterminate`; only the individual
unassessable bonds are omitted from scoring.

`context_warning` is a binary interpretive flag and does not alter the numerical
confidence calculation. Machine-readable `context_warning_reasons` explain the
trigger. Bond rows are flagged for declared non-typical donors and for membership
in a multi-donor group containing a suspect bond. Candidate rows additionally
flag every proximal atom outside the typical donor table. At site level, the
flag summarizes coordination-relevant cases: a non-typical atom satisfying the
first-sphere distance rule, a declared donor-rule override, or a suspect
multi-donor group. A distant non-typical atom found only by the broad 4 Å search
does not by itself place the complete metal site under warning.

### Database-referenced confidence scoring

Confidence inputs are collected as part of an uncapped full-database run: no
`--id`, `--id-file`, manual coordinate arguments, or `--max-pdbs`. As each
worker result reaches the driver, Alchemy combines its already in-memory density
and assigned-bond evidence and streams one compact row per selected metal site
to `confidence_inputs_all.csv`. It never rereads the complete statistics and
bond tables to reconstruct those inputs.

The compact row records the magnitude of metal-site `ZDm` as `rszd_magnitude`,
the largest absolute Zbond among assigned contacts, the responsible contact,
and geometry coverage. Geometry coverage is the number of assigned contacts
with an exact reference distance divided by the total number of assigned
contacts, matching the manuscript definition of `QG`. The finite-Zbond count is
recorded separately so missing DPI cannot be mistaken for usable geometry
evidence. Rejected broad-search candidates do not enter the denominator.
Missing density, absent bonds, partial coverage, diagnostic EDSTATS rows, and
shared-cofactor density provenance remain explicit. The streamed file retains
exactly one row per manifest-counted selected metal; a site with no recoverable
density or bond identity is represented by an unresolved, unscorable placeholder
rather than silently disappearing from the cohort denominator.

Only after the database run completes without operationally incomplete entries
does Alchemy finalize confidence. It scans the compact input—not the raw
analysis outputs—to write `confidence_scores_all.csv` and a reusable
`confidence_reference/` directory containing policy metadata and the empirical
score distribution. An interrupted run retains its compact inputs for
`--resume` but does not publish a completed reference.

The provisional June 2026 fixed score transforms absolute RSZD and maximum
absolute Zbond through piecewise-linear severity anchors and calculates:

```text
confidence = 100 * (1 - 0.50*SR - 0.35*QG*SG
                       - 0.15*QG*sqrt(SR*SG))
```

Alchemy also reports an average-rank percentile relative to the scorable full
database cohort, its size, and a policy identifier. The fixed-formula score and
database percentile are visibly distinct. A deterministic
`confidence_reference_id` identifies the exact frozen policy and score
distribution and prevents resumed output from mixing database snapshots.
`context_warning` is carried into the result as an interpretive annotation and
does not subtract points.

**No confidence reference is distributed with Alchemy, and the score is not
finalized.** A fresh clone therefore produces every other output but reports
that confidence scoring is not enabled; this is expected behaviour, not a
misconfiguration. Enabling it requires either completing an uncapped
full-database run, which writes a reference of your own, or supplying one
explicitly with `--confidence-reference-dir`. Treat the score described in this
section as provisional and subject to change until it is released.

For later single-entry, ID-file, manual, or capped runs, Alchemy first looks for
the reference produced under the current output directory's
`confidence_reference/`, then in the repository's `confidence_reference/`.
`--confidence-reference-dir` selects an explicit copy instead. Alchemy loads
that reference once, derives each new site's compact inputs while its normal
result is still in memory, and writes `confidence_scores_all.csv` directly. These runs
are compared with the frozen database and never generate percentiles from their
own small cohort. If no compatible reference is installed, the ordinary
Alchemy outputs are still produced and confidence scoring is explicitly
disabled.

`src/confidence_score.py` retains `finalize` and `score` subcommands for recovery
and reproducibility using already compact confidence-input CSVs; neither command
reconstructs inputs by rescanning `metal_stats_all.csv` or
`metal_bonds_all.csv`.

The DPI is calculated from PDB-REDO reflection and R-free metadata, the
asymmetric-unit volume, and `Ni`, the sum of occupancies for all non-hydrogen and
non-deuterium atoms in the complete first-model asymmetric unit. Alternate
positions contribute separately to this global sum. If non-given strict-NCS
operations generate copies that are not explicitly deposited, each copy is
included in `Ni`; NCS operations marked as already given are not counted again.
The deposited count, strict-NCS multiplier, and resulting complete count are all
reported. A missing, non-finite, negative, or greater-than-one occupancy makes
DPI unavailable rather than being silently repaired; contact distances that do
not require DPI are retained. Zero occupancy is valid for `Ni` but is not
accepted as evidence for a candidate contact.
For PDB input, raw occupancy records are matched to Gemmi atoms by chain,
residue number and insertion code, residue and atom names, alternate location,
and atom serial rather than parser traversal order.
