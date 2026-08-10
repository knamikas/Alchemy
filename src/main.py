#!/usr/bin/env python
"""Batch-run the Alchemy core pipeline over PDB-REDO entries.

For each PDB entry this validates/corrects its Fourier map coefficients with
CCP4 `mtzfix`, computes 2mFo-DFc and mFo-DFc maps with `fft`, and runs
`edstats`, then extracts per-atom real-space statistics for metal ions and
metal-containing cofactors. Core results are streamed to seven CSVs under
--output-dir:

  metal_sites_all.csv  -- one row per selected metal site
  metal_bonds_all.csv  -- one row per inferred or declared contact
  metal_contact_candidates_all.csv -- one row per contact candidate
  density_context_all.csv -- one per-entry non-target RSZD control summary
  crystallization_conditions_all.csv -- deposited condition records
  crystallization_summary_all.csv -- one contextual row per entry
  manifest.csv         -- one row per entry with status and provenance

An uncapped database run additionally streams compact confidence inputs and,
after successful completion, writes final confidence scores plus a reusable
database reference. Smaller runs use an installed frozen reference when one is
available. Confidence-enabled runs also derive review_queue_all.csv by joining
REVIEW/SUSPECT sites to the separate crystallization summary; the metadata do
not participate in scoring.

Requirements
------------
* CCP4 `mtzfix`, `fft`, `mapmask`, and `edstats` on PATH -- either already
  sourced, or via --ccp4-setup pointing at a CCP4 setup script
  (e.g. <CCP4>/bin/ccp4.setup-sh on POSIX or ccp4.setup.bat on Windows).
* Run under Python 3.11+ with gemmi>=0.7.0, numpy>=1.17, and
  typing_extensions>=4.6. These are required; gemmi does not install numpy.

Examples
--------
  python src/main.py --id 109m \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
  python src/main.py --max-pdbs 20 --workers 4 \
      --ccp4-setup /opt/ccp4/bin/ccp4.setup-sh
"""

import sys

from cli import main


__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
