"""Unit tests for the EDSTATS density Z-score index in ``coordination``."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

import pytest

from coordination.density_zscores import ZD_COLUMNS, DensityZScoreIndex
from output_rows import AtomKey, CsvValue, MetalStatsRow

HEADER = ["RT", "CI", "RN", *ZD_COLUMNS]
AUTHOR_KEY = ("ZN", "A", "1")


def stats_row(
    *,
    values: Sequence[CsvValue] = ("1.5", "-2.5", "3.5"),
    site_key: AtomKey | None = (1, 0, 0, 0),
    author_key: tuple[str, str, str] = AUTHOR_KEY,
) -> MetalStatsRow:
    """Build one metal statistics row carrying the three ZD values."""
    resname, chain, resnum = author_key
    row = MetalStatsRow.from_output_fields(
        "1abc", "metal", (resname, chain, resnum, *values)
    )
    return replace(row, resname=resname, chain=chain, resnum=resnum, site_key=site_key)


def test_lookup_returns_site_values() -> None:
    row = stats_row()
    index = DensityZScoreIndex.from_stats_rows([row], HEADER)

    assert index.zd_indices == (3, 4, 5)
    assert index.lookup(row.site_key, AUTHOR_KEY) == (1.5, -2.5, 3.5)


def test_missing_header_disables_the_join() -> None:
    row = stats_row()

    headers: tuple[list[str] | None, ...] = (None, [])
    for header in headers:
        index = DensityZScoreIndex.from_stats_rows([row], header)
        assert index.zd_indices is None
        assert all(
            math.isnan(value) for value in index.lookup(row.site_key, AUTHOR_KEY)
        )


def test_header_without_the_zd_columns_disables_the_join() -> None:
    row = stats_row()
    index = DensityZScoreIndex.from_stats_rows([row], ["RT", "CI", "RN", "ZDm", "ZD-m"])

    assert index.zd_indices is None
    assert all(math.isnan(value) for value in index.lookup(row.site_key, AUTHOR_KEY))


@pytest.mark.parametrize("text", ["n/a", "****", "", "-"])
def test_unparsable_values_yield_nans(text: str) -> None:
    row = stats_row(values=(text, text, text))
    index = DensityZScoreIndex.from_stats_rows([row], HEADER)

    assert all(math.isnan(value) for value in index.lookup(row.site_key, AUTHOR_KEY))


def test_row_shorter_than_the_header_yields_nans() -> None:
    row = stats_row(values=("1.5",))
    index = DensityZScoreIndex.from_stats_rows([row], HEADER)

    assert all(math.isnan(value) for value in index.lookup(row.site_key, AUTHOR_KEY))


def test_none_field_yields_nans_instead_of_raising() -> None:
    """``CsvValue`` admits ``None``; parsing it must not end the bond stage."""
    row = stats_row(values=(None, None, None))
    index = DensityZScoreIndex.from_stats_rows([row], HEADER)

    assert all(math.isnan(value) for value in index.lookup(row.site_key, AUTHOR_KEY))


def test_unknown_site_and_author_miss() -> None:
    index = DensityZScoreIndex.from_stats_rows([], HEADER)

    assert index.by_site == {}
    assert all(math.isnan(value) for value in index.lookup((1, 0, 0, 0), AUTHOR_KEY))
    assert all(math.isnan(value) for value in index.lookup(None, AUTHOR_KEY))


def test_author_fallback_serves_a_site_without_its_own_row() -> None:
    """A row whose coordinate join failed carries no site key, only an identity."""
    row = stats_row(site_key=None)
    index = DensityZScoreIndex.from_stats_rows([row], HEADER)

    assert index.by_site == {}
    assert index.lookup((7, 0, 0, 0), AUTHOR_KEY) == (1.5, -2.5, 3.5)


def test_ambiguous_author_identity_is_dropped_from_the_fallback() -> None:
    rows = [
        stats_row(site_key=None, values=("1.5", "-2.5", "3.5")),
        stats_row(site_key=None, values=("9.5", "-9.5", "9.5")),
    ]
    index = DensityZScoreIndex.from_stats_rows(rows, HEADER)

    assert index.by_author == {}
    assert all(math.isnan(value) for value in index.lookup(None, AUTHOR_KEY))


def test_duplicate_site_key_raises() -> None:
    rows = [stats_row(), stats_row(values=("9.5", "-9.5", "9.5"))]

    with pytest.raises(ValueError, match="duplicate site key: 1/0/0/0"):
        DensityZScoreIndex.from_stats_rows(rows, HEADER)


def test_distinct_site_keys_share_one_author_identity() -> None:
    """Two metals in one residue keep their own rows under the same identity."""
    rows = [
        stats_row(site_key=(1, 0, 0, 0)),
        stats_row(site_key=(1, 0, 0, 1), values=("4.5", "-4.5", "4.5")),
    ]
    index = DensityZScoreIndex.from_stats_rows(rows, HEADER)

    assert index.lookup((1, 0, 0, 0), AUTHOR_KEY) == (1.5, -2.5, 3.5)
    assert index.lookup((1, 0, 0, 1), AUTHOR_KEY) == (4.5, -4.5, 4.5)
    assert index.by_author == {}
