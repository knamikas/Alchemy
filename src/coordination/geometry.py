"""Score assigned contacts against literature distances.

    z = (d_observed - mu) / sqrt(DPI**2 + sigma_lit**2)

Distances come from Harding (2006) and Zheng et al. (2008, Ni); DPI follows
Blow (2002), equation 7. Missing scoring inputs retain measured geometry with
NaN derived values. Contacts that share one donor-residue image are then
grouped so a suspect member marks the whole group.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from codes import ContactScope, MultiDonorStatus, ScoreExclusionReason
from coordination.contact_record import Candidate, GeometryResult, MultiDonorResult
from coordination.eligibility import bonding_key
from coordination.policy import ZSCORE_OUTLIER_CUTOFF
from reference_data import literature_distances
from structure_analysis import NAN

#: What makes two contacts share one donor-residue image around one metal.
ResidueImageKey = tuple[
    tuple[int, int, int], ContactScope, str, int, str, tuple[int, int, int]
]


def zscore(dist: float, mu: float, stdev: float, dpi: float) -> float:
    """Return the bond-distance z-score using sqrt(stdev**2 + dpi**2).

    Use one DPI for donor uncertainty, treating the heavy metal as well ordered.
    An independent two-atom error model would use a different denominator.
    """
    if not (math.isfinite(dpi) and math.isfinite(mu) and math.isfinite(stdev)):
        return NAN
    denom = math.sqrt(dpi**2 + stdev**2)
    return (dist - mu) / denom if denom > 0 else NAN


def annotate_contacts(
    contacts: Iterable[Candidate], metal_element: str, dpi: float
) -> None:
    """Attach reference-based geometry results to candidate contacts."""
    for contact in contacts:
        neighbor = contact.neighbor
        reported_distance = round(contact.image.distance, 3)
        literature = literature_distances().get(bonding_key(neighbor, metal_element))
        if literature is None:
            mu = stdev = zscore_raw = NAN
        else:
            mu, stdev = literature
            zscore_raw = zscore(contact.image.distance, mu, stdev, dpi)
        rounded_zscore = round(zscore_raw, 4) if math.isfinite(zscore_raw) else NAN
        outlier: bool | str
        consistent: bool | str
        if math.isfinite(zscore_raw):
            magnitude = abs(zscore_raw)
            outlier = magnitude >= ZSCORE_OUTLIER_CUTOFF or math.isclose(
                magnitude,
                ZSCORE_OUTLIER_CUTOFF,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            consistent = not outlier
        else:
            outlier = ""
            consistent = ""
        contact.set_geometry(
            GeometryResult(
                distance=reported_distance,
                literature_distance=mu,
                literature_stdev=stdev,
                zscore=rounded_zscore,
                reference_covered=literature is not None,
                outlier=outlier,
                consistent=consistent,
            )
        )


def residue_image_key(contact: Candidate) -> ResidueImageKey:
    """Identity of one donor residue image around the current metal site."""
    return (
        contact.neighbor.residue_key,
        contact.image.scope,
        contact.image.strict_ncs_operation_id,
        contact.image.image_index,
        contact.image.symmetry_operation,
        contact.image.translation,
    )


def annotate_multi_donor_groups(contacts: Iterable[Candidate]) -> None:
    """Annotate assigned contacts that share one donor-residue image.

    Group status is contextual: a suspect member marks every member as
    belonging to a suspect group without weakening any individual result.
    """
    groups: dict[ResidueImageKey, list[Candidate]] = {}
    for contact in contacts:
        groups.setdefault(residue_image_key(contact), []).append(contact)

    for group in groups.values():
        count = len(group)
        multi_donor = count >= 2
        if not multi_donor:
            contact = group[0]
            geometry = contact.geometry()
            assessable = geometry.outlier is True or geometry.consistent is True
            contact.set_multi_donor(
                MultiDonorResult(
                    detected=False,
                    contact_count=1,
                    geometry_status=MultiDonorStatus.SINGLE_DONOR,
                    contains_suspect_bond=False,
                    score_eligible=assessable,
                    score_exclusion_reason=(
                        "" if assessable else ScoreExclusionReason.ZSCORE_UNAVAILABLE
                    ),
                )
            )
            continue

        any_outlier = any(contact.geometry().outlier is True for contact in group)
        all_consistent = all(contact.geometry().consistent is True for contact in group)
        if all_consistent:
            status = MultiDonorStatus.CONSISTENT
        elif any_outlier:
            status = MultiDonorStatus.SUSPECT
        else:
            status = MultiDonorStatus.INDETERMINATE
        for contact in group:
            geometry = contact.geometry()
            assessable = geometry.outlier is True or geometry.consistent is True
            contact.set_multi_donor(
                MultiDonorResult(
                    detected=True,
                    contact_count=count,
                    geometry_status=status,
                    contains_suspect_bond=any_outlier,
                    score_eligible=assessable,
                    score_exclusion_reason=(
                        "" if assessable else ScoreExclusionReason.ZSCORE_UNAVAILABLE
                    ),
                )
            )
