# Maintenance

The data that travels with the checkout, and how to change it. Both files below
decide results: the catalog says what counts as a metal cofactor, the distance
table sets every assignment cutoff and every z-score. Both are verified against
their checksums when a run first reads them, and every manifest row records the
`reference_data_id` they compose.

## Cofactor reference maintenance

Normal analysis always loads the fixed catalog bundled in `src/data`. It never
checks the catalog's age, accesses the network, selects a user cache, or
rebuilds the catalog.

Catalog updates are an explicit developer maintenance operation:

```console
python tools/build_metallocofactor_catalog.py
```

The isolated builder downloads the wwPDB Chemical Component Dictionary and
replaces the bundled catalog and its metadata only after a successful build,
using atomic replacement for each file. A local CCD snapshot can be supplied
with `--ccd`. Use `--status` to report the generation time, entry count,
checksum, and integrity of the currently bundled catalog
without downloading anything. Catalog changes should be reviewed and committed
as part of a software release before an analysis run.

Cluster and heme classes are derived from CCD atom connectivity, not formula
stoichiometry. Clusters require a bridging sulfur or selenium, while hemes
require an Fe-bound porphyrinoid core. Run a full rebuild whenever the component
list or classification rules change.

## Reference data

- `src/data/metallocofactors_id.txt` — fixed bundled list of metal-containing
  Chemical Component Dictionary IDs used by every analysis run.
- `src/data/metallocofactors_id.meta.json` — generation metadata for the
  committed cofactor list.
- `src/data/metal_distances_info.txt` — reference metal-ligand distances and
  standard deviations, keyed by donor residue, donor element and metal. Values
  are from Harding (2006), [Acta Cryst. D62,
  678-682](https://doi.org/10.1107/S0907444906014594), except NI, which is from
  Zheng et al. (2008), and SER/THR/TYR, which are approximated from statements
  in Harding (2006) rather than tabulated. See [method.md]
  (method.md#reference-coverage-of-the-donor-table) for which donors this file
  does and does not cover. Note the format
  trap: column 1 `CA` is the backbone-carbonyl pseudo residue, while column 3
  `CA` is calcium.
- `src/data/metal_distances_info.meta.json` — checksum, row count and citations
  for the distance table, written by `tools/stamp_distance_table.py`.

Both bundled files are verified against their sidecars when they are first
read, and a run stops rather than analyze against data that has drifted from
what the tools recorded. After editing the distance table by hand, re-stamp it:

```console
python tools/stamp_distance_table.py            # rewrite the sidecar
python tools/stamp_distance_table.py --check    # verify, change nothing
```
