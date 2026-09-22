"""Paths of the CSV and score-reference artifacts one run writes.

Other files under ``--output-dir`` (the lock file, ``logs/``, scratch
directories) are named by the modules that own them.
"""

from __future__ import annotations

import os

from driver.errors import DriverError


class OutputLayout:
    """Paths to run artifacts derived from the output directory."""

    def __init__(self, output_dir: str) -> None:
        """Derive every run artifact path from an output directory."""
        self.manifest = os.path.join(output_dir, "manifest.csv")
        self.stats = os.path.join(output_dir, "metal_sites_all.csv")
        self.density_context = os.path.join(output_dir, "density_context_all.csv")
        self.bonds = os.path.join(output_dir, "metal_bonds_all.csv")
        self.candidates = os.path.join(output_dir, "metal_contact_candidates_all.csv")
        self.score_inputs = os.path.join(output_dir, "score_inputs_all.csv")
        self.scores = os.path.join(output_dir, "scores_all.csv")
        self.crystallization_conditions = os.path.join(
            output_dir, "crystallization_conditions_all.csv"
        )
        self.crystallization_summary = os.path.join(
            output_dir, "crystallization_summary_all.csv"
        )
        self.review_queue = os.path.join(output_dir, "review_queue_all.csv")
        self.reference_dir = os.path.join(output_dir, "score_reference")
        self.legacy_scientific_outputs = (
            os.path.join(output_dir, "metal_stats_all.csv"),
            os.path.join(output_dir, "metal_candidates_all.csv"),
        )


def prepare_output_directory(output_dir: str) -> None:
    """Create ``--output-dir`` before its stable lock file is opened."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise DriverError(
            f"Cannot use --output-dir {output_dir}: {exc.strerror or exc}"
        ) from None
