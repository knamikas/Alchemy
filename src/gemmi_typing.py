"""Typed views of Gemmi members that its type stub leaves untyped."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import numpy.typing as npt


class MtzColumnData(Protocol):
    """Gemmi MTZ column members whose stub currently lacks concrete types.

    The binding exposes the column's dataset id, its index into the data
    array, and the column values themselves; Gemmi's stub omits or leaves
    these untyped, so callers read them through this Protocol instead of
    ``Any``.
    """

    dataset_id: int
    array: npt.NDArray[np.float32]
    idx: int


def mtz_column_data(column: object) -> MtzColumnData:
    """Apply Gemmi's runtime MTZ-column contract at one typed boundary."""
    return cast(MtzColumnData, column)
