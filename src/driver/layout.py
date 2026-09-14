"""Paths of the artifacts one run writes under its output directory."""

from __future__ import annotations

import os

from driver.errors import DriverError
from run_logging import logger_for

logger = logger_for(__name__)


class OutputLayout:
    """Paths to run artifacts derived from the output directory."""

    def __init__(self, output_dir: str) -> None:
        """Derive every run artifact path from an output directory."""
        self.output_dir = output_dir
        self.manifest = os.path.join(output_dir, "manifest.csv")
        self.stats = os.path.join(output_dir, "metal_sites_all.csv")
        self.density_context = os.path.join(output_dir, "density_context_all.csv")
        self.bonds = os.path.join(output_dir, "metal_bonds_all.csv")
        self.candidates = os.path.join(output_dir, "metal_contact_candidates_all.csv")
        self.confidence_inputs = os.path.join(output_dir, "confidence_inputs_all.csv")
        self.confidence_scores = os.path.join(output_dir, "confidence_scores_all.csv")
        self.crystallization_conditions = os.path.join(
            output_dir, "crystallization_conditions_all.csv"
        )
        self.crystallization_summary = os.path.join(
            output_dir, "crystallization_summary_all.csv"
        )
        self.review_queue = os.path.join(output_dir, "review_queue_all.csv")
        self.reference_dir = os.path.join(output_dir, "confidence_reference")
        self.legacy_scientific_outputs = (
            os.path.join(output_dir, "metal_stats_all.csv"),
            os.path.join(output_dir, "metal_candidates_all.csv"),
        )

    @property
    def core(self) -> tuple[str, str, str, str]:
        """The four always-written outputs, in resume-validation order."""
        return (self.manifest, self.stats, self.bonds, self.candidates)


def prepare_output_directory(output_dir: str) -> None:
    """Create ``--output-dir`` before its stable lock file is opened."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise DriverError(
            f"Cannot use --output-dir {output_dir}: {exc.strerror or exc}"
        ) from None
