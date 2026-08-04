"""The four streamed CSV outputs and the schemas they are written against.

Every entry's rows are appended and flushed as soon as the worker returns, so
an interrupted batch keeps the results it already has. That makes the column
lists an external contract twice over: users read the files, and ``--resume``
reads them back to decide what still needs running. Defining each schema once,
here, is what keeps the written header and the resume check from disagreeing.
"""

import csv
from typing import Optional

from bond_schema import (
    BOND_COLUMNS,
    CANDIDATE_COLUMNS,
    STATS_EXTRA_COLUMNS,
    _check_row_schema,
)
from confidence_score import CONFIDENCE_INPUT_COLUMNS
from metal_identification import EDSTATS_COLUMNS


# CSV column names keep the deposited-data spelling ``pdbID`` even though every
# Python identifier is ``pdb_id``. The columns are an external contract: users'
# scripts, notebooks and downstream joins address them by name, and renaming
# one would break those silently while gaining nothing. The two spellings meet
# only where a row dict is built, e.g. ``{"pdbID": pdb_id}``.
MANIFEST_COLUMNS = [
    "pdbID",
    "status",
    "retryable",
    "n_metals",
    "n_bonds",
    "n_candidates",
    "runtime_s",
    "reason_codes",
    "warning_codes",
    "error",
    "alchemy_version",
    "alchemy_commit",
    "gemmi_version",
    "ccp4_version",
    "refinement_state",
    "source_coordinate_format",
    "analysis_coordinate_format",
    "coordinate_conversion_performed",
    "source_coordinate_path",
    "analysis_coordinate_path",
    "model_policy",
    "input_model_count",
    "model_analyzed",
    "multi_model_structure",
    "altloc_policy",
    "symmetry_contact_policy",
]

# metal_stats_all.csv schema. The middle block is the EDSTATS residue table,
# whose column set and order `extract_metal_statistics` validates against
# EDSTATS_COLUMNS before emitting any row, so the full header is fixed. Defining
# it once keeps the written header and the --resume compatibility check from
# disagreeing about the columns between them.
STATS_COLUMNS = (
    ["pdbID", "category"]
    + list(EDSTATS_COLUMNS)
    + ["aa_geometry_coverage"]
    + list(STATS_EXTRA_COLUMNS)
)


def _manifest_row(
    result, resume, bonds_enabled, prior_bond_counts, prior_candidate_counts
):
    """Project one worker result onto the manifest schema."""
    row = {column: result.get(column, "") for column in MANIFEST_COLUMNS}
    n_bonds = result["n_bonds"]
    n_candidates = result["n_candidates"]
    if not bonds_enabled:
        n_bonds = prior_bond_counts.get(result["pdbID"].lower(), "") if resume else ""
        n_candidates = (
            prior_candidate_counts.get(result["pdbID"].lower(), "") if resume else ""
        )
    row.update(
        n_metals=result["n"],
        n_bonds=n_bonds,
        n_candidates=n_candidates,
        runtime_s=result["runtime"],
        reason_codes="|".join(result.get("reason_codes", [])),
        warning_codes="|".join(result.get("warning_codes", [])),
    )
    return row


class _OutputWriters:
    """The streamed CSV outputs, with running row counts.

    Each stream is flushed after every entry so an interrupted batch run
    retains the results it already completed. Headers are written on creation.
    """

    def __init__(
        self,
        manifest_fh,
        stats_fh,
        bonds_fh,
        candidates_fh,
        confidence_fh=None,
        confidence_columns=None,
        confidence_inputs_fh=None,
    ):
        self._manifest_fh = manifest_fh
        self._stats_fh = stats_fh
        self._bonds_fh = bonds_fh
        self._candidates_fh = candidates_fh
        self._confidence_fh = confidence_fh
        self._confidence_inputs_fh = confidence_inputs_fh
        self._manifest = csv.DictWriter(manifest_fh, fieldnames=MANIFEST_COLUMNS)
        self._stats = csv.writer(stats_fh)
        self._bonds = csv.writer(bonds_fh) if bonds_fh is not None else None
        self._candidates = (
            csv.writer(candidates_fh) if candidates_fh is not None else None
        )
        if confidence_fh is not None and confidence_columns is None:
            raise ValueError("confidence columns are required with a confidence output")
        # Annotated because both start ``None`` and are assigned a writer
        # below: left to inference the attribute types would be ``None``, and
        # every assignment after this point an error.
        self._confidence: Optional[csv.DictWriter] = None
        self._confidence_inputs: Optional[csv.DictWriter] = None
        if confidence_fh is not None and confidence_columns is not None:
            self._confidence = csv.DictWriter(
                confidence_fh, fieldnames=confidence_columns
            )
        if confidence_inputs_fh is not None:
            if confidence_fh is None:
                raise ValueError(
                    "confidence inputs synchronization requires scored output"
                )
            self._confidence_inputs = csv.DictWriter(
                confidence_inputs_fh, fieldnames=CONFIDENCE_INPUT_COLUMNS
            )
        self._confidence_columns = confidence_columns
        self.n_rows = 0
        self.n_bonds = 0
        self.n_candidates = 0
        self.n_confidence = 0
        self._manifest.writeheader()
        self._stats.writerow(STATS_COLUMNS)
        if self._bonds is not None:
            self._bonds.writerow(BOND_COLUMNS)
        if self._candidates is not None:
            self._candidates.writerow(CANDIDATE_COLUMNS)
        if self._confidence is not None:
            self._confidence.writeheader()
        if self._confidence_inputs is not None:
            self._confidence_inputs.writeheader()

    def write_stats_rows(self, rows):
        if not rows:
            return
        for row in rows:
            self._stats.writerow([row["pdbID"], row["category"]] + row["fields"])
            self.n_rows += 1
        self._stats_fh.flush()

    def write_bond_rows(self, bond_rows):
        if self._bonds is None or not bond_rows:
            return
        _check_row_schema(bond_rows[0], BOND_COLUMNS, "metal_bonds_all.csv")
        for bond in bond_rows:
            self._bonds.writerow([bond[column] for column in BOND_COLUMNS])
            self.n_bonds += 1
        self._bonds_fh.flush()

    def write_candidate_rows(self, candidate_rows):
        if self._candidates is None or not candidate_rows:
            return
        _check_row_schema(
            candidate_rows[0], CANDIDATE_COLUMNS, "metal_candidates_all.csv"
        )
        for candidate in candidate_rows:
            self._candidates.writerow(
                [candidate[column] for column in CANDIDATE_COLUMNS]
            )
            self.n_candidates += 1
        self._candidates_fh.flush()

    def write_manifest_row(self, row):
        self._manifest.writerow(row)
        self._manifest_fh.flush()

    def write_confidence_rows(self, rows):
        if self._confidence is None or not rows:
            return
        if self._confidence_columns is None or self._confidence_fh is None:
            raise RuntimeError("confidence output is not fully configured")
        expected = set(self._confidence_columns)
        if rows[0].keys() != expected:
            raise RuntimeError("confidence row does not match its output schema")
        self._confidence.writerows(rows)
        if self._confidence_inputs is not None:
            self._confidence_inputs.writerows(
                {column: row[column] for column in CONFIDENCE_INPUT_COLUMNS}
                for row in rows
            )
        self.n_confidence += len(rows)
        self._confidence_fh.flush()
        if self._confidence_inputs_fh is not None:
            self._confidence_inputs_fh.flush()
