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

from codes import MultiDonorStatus, ScoreExclusionReason
from coordination.contact_record import Candidate, GeometryResult, MultiDonorResult
from coordination.eligibility import bonding_key
from coordination.metal_distances.distances import literature_distances
from coordination.policy import ZSCORE_OUTLIER_CUTOFF
from structure_analysis import NAN

#: What makes two contacts share one donor-residue image around one metal: the
#: deposited residue plus the symmetry image it was generated in. ``scope``,
#: ``strict_ncs_operation_id`` and ``symmetry_operation`` are all derived from
#: the image index, so naming them again would only repeat this key.
ResidueImageKey = tuple[tuple[int, int, int], int, tuple[int, int, int]]


def zscore(dist: float, mu: float, stdev: float, dpi: float) -> float:
    """Return the bond-distance z-score using sqrt(stdev**2 + dpi**2).

    Use one DPI for donor uncertainty, treating the heavy metal as well ordered.
    An independent two-atom error model would use a different denominator.
    """
    if not (
        math.isfinite(dist)
        and math.isfinite(dpi)
        and math.isfinite(mu)
        and math.isfinite(stdev)
    ):
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
        # ``None`` rather than ``False``: without a finite z-score the contact
        # was never assessed, which is not the same as assessed and in range.
        assessed = math.isfinite(zscore_raw)
        outlier = abs(zscore_raw) >= ZSCORE_OUTLIER_CUTOFF if assessed else None
        contact.set_geometry(
            GeometryResult(
                distance=reported_distance,
                literature_distance=mu,
                literature_stdev=stdev,
                zscore=rounded_zscore,
                reference_covered=literature is not None,
                outlier=outlier,
                score_eligible=assessed,
                score_exclusion_reason=(
                    "" if assessed else ScoreExclusionReason.ZSCORE_UNAVAILABLE
                ),
            )
        )


def residue_image_key(contact: Candidate) -> ResidueImageKey:
    """Identity of one donor residue image around the current metal site."""
    return (
        contact.neighbor.residue_key,
        contact.image.image_index,
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
        any_outlier = multi_donor and any(
            contact.geometry().outlier is True for contact in group
        )
        if not multi_donor:
            status = MultiDonorStatus.SINGLE_DONOR
        elif all(contact.geometry().consistent is True for contact in group):
            status = MultiDonorStatus.CONSISTENT
        elif any_outlier:
            status = MultiDonorStatus.SUSPECT
        else:
            status = MultiDonorStatus.INDETERMINATE
        for contact in group:
            contact.set_multi_donor(
                MultiDonorResult(
                    detected=multi_donor,
                    contact_count=count,
                    geometry_status=status,
                    contains_suspect_bond=any_outlier,
                )
            )
