"""Join one entry's EDSTATS rows to its metal sites for the bond rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from output_rows import CsvValue, MetalStatsRow
from structure_analysis import NAN

#: The main-chain RSZD triple every bond row carries, in output order.
ZD_COLUMNS = ("ZDm", "ZD-m", "ZD+m")


@dataclass(frozen=True, slots=True)
class DensityZScoreIndex:
    """One entry's EDSTATS rows, addressable by metal site or author identity.

    ``by_site`` keys rows by the selected metal's coordinate ``source_key``;
    ``by_author`` keys them by ``(resname, chain, resnum)`` and omits every
    author identity that more than one row claims, so the fallback join can
    never pick an arbitrary residue. ``zd_indices`` locates :data:`ZD_COLUMNS`
    in the row fields, or is ``None`` when the header lacks any of them.
    """

    by_site: Mapping[tuple[Any, ...], Sequence[CsvValue]]
    by_author: Mapping[tuple[Any, ...], Sequence[CsvValue]]
    zd_indices: tuple[int, int, int] | None

    @classmethod
    def from_stats_rows(
        cls, stats_rows: Iterable[MetalStatsRow], header: Sequence[str] | None
    ) -> DensityZScoreIndex:
        """Index the extracted EDSTATS rows of one entry against their header."""
        by_site: dict[tuple[Any, ...], Sequence[CsvValue]] = {}
        by_author: dict[tuple[Any, ...], Sequence[CsvValue]] = {}
        ambiguous_authors: set[tuple[Any, ...]] = set()
        for row in stats_rows:
            if row.site_key is not None:
                by_site[tuple(row.site_key)] = row.fields
            author_key = (row.resname, str(row.chain), str(row.resnum))
            if author_key in by_author:
                ambiguous_authors.add(author_key)
            else:
                by_author[author_key] = row.fields
        for author_key in ambiguous_authors:
            del by_author[author_key]
        return cls(by_site, by_author, _zd_indices(header))

    def lookup(
        self, site_key: Sequence[Any] | None, author_key: tuple[str, str, str]
    ) -> tuple[float, float, float]:
        """Return ``(ZDm, ZD-m, ZD+m)`` for a site, or NaNs when unavailable.

        The site key wins; the author key is the fallback for rows that never
        received a site key. A missing row, a header without the ZD columns, or
        an unparsable value all yield the NaN triple.
        """
        resname, chain, resnum = author_key
        indexed: Sequence[CsvValue] | None = None
        if site_key is not None:
            indexed = self.by_site.get(tuple(site_key))
        if indexed is None:
            indexed = self.by_author.get((resname, str(chain), str(resnum)))
        if indexed is None or self.zd_indices is None:
            return NAN, NAN, NAN
        # ``zd_indices`` addresses EDSTATS header columns, which are parsed text;
        # only the site fields appended after them can hold non-string values.
        fields = cast(Sequence[str], indexed)
        try:
            return (
                float(fields[self.zd_indices[0]]),
                float(fields[self.zd_indices[1]]),
                float(fields[self.zd_indices[2]]),
            )
        except (IndexError, ValueError):
            return NAN, NAN, NAN


def _zd_indices(header: Sequence[str] | None) -> tuple[int, int, int] | None:
    """Header positions of :data:`ZD_COLUMNS`, or ``None`` when any is absent."""
    if not header:
        return None
    try:
        zdm, zd_minus, zd_plus = (header.index(name) for name in ZD_COLUMNS)
    except ValueError:
        return None
    return zdm, zd_minus, zd_plus
