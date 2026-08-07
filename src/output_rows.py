from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, ClassVar, override
from collections.abc import Iterator, Mapping, Sequence

from structure_analysis import AtomSite


CsvValue = str | int | float | bool | None
AtomKey = tuple[int, int, int, int]
ResidueKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MetalStatsRow(Mapping[str, Any]):
    pdb_id: str
    category: str
    resname: str
    chain: str
    resnum: str
    fields: tuple[CsvValue, ...]
    density_observation_id: str
    density_scope: str
    density_shared_site_count: int
    density_is_shared: bool
    coordinate_mapping_status: str
    selected_metal_site_status: str
    site: AtomSite | None
    site_key: AtomKey | None
    residue_key: ResidueKey | None

    field_names: ClassVar[tuple[str, ...]] = (
        "pdbID",
        "category",
        "resname",
        "chain",
        "resnum",
        "fields",
        "density_observation_id",
        "density_scope",
        "density_shared_site_count",
        "density_is_shared",
        "coordinate_mapping_status",
        "selected_metal_site_status",
        "site",
        "site_key",
        "residue_key",
    )

    @override
    def __getitem__(self, key: str) -> Any:
        if key == "pdbID":
            return self.pdb_id
        if key not in self.field_names:
            raise KeyError(key)
        return getattr(self, key)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self.field_names)

    @override
    def __len__(self) -> int:
        return len(self.field_names)

    def with_fields(self, fields: Sequence[CsvValue]) -> MetalStatsRow:
        return replace(self, fields=tuple(fields))

    @classmethod
    def from_output_fields(
        cls, pdb_id: str, category: str, fields: Sequence[CsvValue]
    ) -> MetalStatsRow:
        return cls(
            pdb_id=pdb_id,
            category=category,
            resname="",
            chain="",
            resnum="",
            fields=tuple(fields),
            density_observation_id="",
            density_scope="",
            density_shared_site_count=0,
            density_is_shared=False,
            coordinate_mapping_status="",
            selected_metal_site_status="",
            site=None,
            site_key=None,
            residue_key=None,
        )

    def as_output_dict(self, columns: Sequence[str]) -> dict[str, CsvValue]:
        values = (self.pdb_id, self.category, *self.fields)
        if len(values) != len(columns):
            raise ValueError("metal statistics row does not match its output schema")
        return dict(zip(columns, values, strict=True))


def csv_value(value: object) -> CsvValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"output value is not a CSV scalar: {type(value).__name__}")


def scientific_csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
