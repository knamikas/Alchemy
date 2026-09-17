"""Join one entry's EDSTATS rows to its metal sites for the bond rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from output_rows import CsvValue, MetalStatsRow, finite_float
from structure_analysis import NAN, AtomKey

#: The main-chain RSZD triple every bond row carries, in output order.
ZD_COLUMNS = ("ZDm", "ZD-m", "ZD+m")


@dataclass(frozen=True, slots=True)
class DensityZScoreIndex:
    """One entry's EDSTATS rows, addressable by metal site or author identity.

    ``by_site`` keys rows by the selected metal's coordinate ``source_key``;
    ``by_author`` keys them by ``(resname, chain, resnum)`` and omits every
    author identity that more than one row claims, so the fallback join can
    never pick an arbitrary residue. ``zd_indices`` locates :data:`ZD_COLUMNS`
    in the row fields, or is ``None`` when the header is missing, empty, or
    lacks any of them.
    """

    by_site: Mapping[AtomKey, Sequence[CsvValue]]
    by_author: Mapping[tuple[str, str, str], Sequence[CsvValue]]
    zd_indices: tuple[int, int, int] | None

    @classmethod
    def from_stats_rows(
        cls, stats_rows: Iterable[MetalStatsRow], header: Sequence[str] | None
    ) -> DensityZScoreIndex:
        """Index the extracted EDSTATS rows of one entry against their header.

        Raise when two rows claim the same metal site, matching the duplicate
        check the confidence stage applies to the same table.
        """
        by_site: dict[AtomKey, Sequence[CsvValue]] = {}
        by_author: dict[tuple[str, str, str], Sequence[CsvValue]] = {}
        ambiguous_authors: set[tuple[str, str, str]] = set()
        for row in stats_rows:
            if row.site_key is not None:
                if row.site_key in by_site:
                    raise ValueError(
                        "metal statistics contain duplicate site key: "
                        + "/".join(str(part) for part in row.site_key)
                    )
                by_site[row.site_key] = row.fields
            author_key = (row.resname, row.chain, row.resnum)
            if author_key in by_author:
                ambiguous_authors.add(author_key)
            else:
                by_author[author_key] = row.fields
        for author_key in ambiguous_authors:
            del by_author[author_key]
        return cls(by_site, by_author, _zd_indices(header))

    def lookup(
        self, site_key: AtomKey | None, author_key: tuple[str, str, str]
    ) -> tuple[float, float, float]:
        """Return ``(ZDm, ZD-m, ZD+m)`` for a site, or NaNs when unavailable.

        The site key wins; the author key is the fallback for rows that never
        received a site key. A missing row, a header without the ZD columns, or
        a row too short to reach them yields the NaN triple; a single blank,
        malformed, or non-finite field becomes NaN on its own.
        """
        indexed: Sequence[CsvValue] | None = None
        if site_key is not None:
            indexed = self.by_site.get(site_key)
        if indexed is None:
            # Serves the selected metal site that has no row of its own
            # (``metal_site_without_density``): an EDSTATS row whose coordinate
            # join failed carries no site key, so a metal inside such a
            # cofactor-named residue is reachable only by author identity.
            indexed = self.by_author.get(author_key)
        if indexed is None or self.zd_indices is None:
            return NAN, NAN, NAN
        # ``zd_indices`` addresses EDSTATS header columns, which are parsed text;
        # only the site fields appended after them can hold non-string values.
        fields = cast(Sequence[str], indexed)
        # ``finite_float`` absorbs the blank, malformed, and ``None`` fields a
        # non-EDSTATS row could hold; only a short row still raises.
        try:
            return (
                finite_float(fields[self.zd_indices[0]]),
                finite_float(fields[self.zd_indices[1]]),
                finite_float(fields[self.zd_indices[2]]),
            )
        except IndexError:
            return NAN, NAN, NAN


def _zd_indices(header: Sequence[str] | None) -> tuple[int, int, int] | None:
    """Header positions of :data:`ZD_COLUMNS`, or ``None`` when any is absent."""
    if not header:
        return None
    try:
        found = [header.index(name) for name in ZD_COLUMNS]
    except ValueError:
        return None
    zdm, zd_minus, zd_plus = found
    return zdm, zd_minus, zd_plus
