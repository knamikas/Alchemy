"""Behavioural tests for ``src/structure_model.py``.

The geometry helper is exercised directly; ``StructureContext`` methods are
exercised on contexts loaded from structures built in memory with the shared
helpers.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path

import gemmi
import helpers
import pytest
from helpers import StructureBuilder, approx, simple_metal_site

import structure_model
from structure_analysis import load_structure
from structure_model import StructureContext


def _write_pdb_with_ncs(
    builder: StructureBuilder,
    path: str | os.PathLike[str],
    operations: Sequence[tuple[str, bool]],
) -> str:
    """Write ``builder`` as PDB with MTRIX records for ``(id, given)`` ops."""
    structure = builder.to_gemmi()
    transform = gemmi.Transform()
    transform.mat.fromlist([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform.vec.fromlist([10.0, 0.0, 0.0])
    for identifier, given in operations:
        structure.ncs.append(gemmi.NcsOp(transform, identifier, bool(given)))
    path = str(path)
    helpers.write_pdb(structure, path)
    return path


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((0.0, 0.0, 0.0), (3.0, 4.0, 12.0), 13.0),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0),
        ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 0.0),
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), math.sqrt(12.0)),
        ((0.0, 0.0, 0.0), (2.03, 0.0, 0.0), 2.03),
        ((5.0, 0.0, 0.0), (0.0, 0.0, 0.0), 5.0),
    ],
)
def test_position_distance_is_euclidean(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    expected: float,
) -> None:
    assert structure_model.position_distance(a, b) == approx(expected, abs=1e-12)
    assert structure_model.position_distance(b, a) == approx(expected, abs=1e-12)


def test_position_distance_matches_gemmi_position_distance() -> None:
    first = (1.5, -2.25, 7.75)
    second = (-3.0, 4.5, 0.25)
    expected = gemmi.Position(*first).dist(gemmi.Position(*second))
    assert structure_model.position_distance(first, second) == approx(
        expected, abs=1e-12
    )


@pytest.fixture
def ncs_context(tmp_path: Path) -> StructureContext:
    """A loaded context with four symmetry operations and two strict-NCS ops."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = _write_pdb_with_ncs(
        builder, tmp_path / "prov.pdb", [("1", False), ("2", False)]
    )
    context = load_structure("test", path)
    assert context.symmetry.crystallographic_operation_count == 4
    assert context.symmetry.strict_ncs_operation_ids == ("1", "2")
    return context


@pytest.mark.parametrize(
    "image_index, translation, expected",
    [
        (0, (0, 0, 0), (False, False, "", "explicit")),
        (1, (0, 0, 0), (True, False, "", "crystallographic")),
        (3, (0, 0, 0), (True, False, "", "crystallographic")),
        (0, (1, 0, 0), (True, False, "", "crystallographic")),
        (0, (0, 0, -1), (True, False, "", "crystallographic")),
        (4, (0, 0, 0), (False, True, "1", "strict_ncs")),
        (5, (0, 0, 0), (True, True, "1", "strict_ncs_and_crystallographic")),
        (4, (1, 0, 0), (True, True, "1", "strict_ncs_and_crystallographic")),
        (8, (0, 0, 0), (False, True, "2", "strict_ncs")),
        (11, (0, 0, 0), (True, True, "2", "strict_ncs_and_crystallographic")),
    ],
)
def test_image_provenance_classifies_symmetry_and_ncs(
    ncs_context: StructureContext,
    image_index: int,
    translation: tuple[int, int, int],
    expected: tuple[bool, bool, str, str],
) -> None:
    """Verify Gemmi image indices preserve symmetry provenance.

    ``setup_cell_images()`` lists the identity, remaining space-group
    operations, and then one full block per strict-NCS transform. A nonzero
    cell translation is crystallographic even at the identity.
    """
    assert ncs_context.image_provenance(image_index, translation) == expected


def test_image_provenance_explicit_only_for_the_identity_image(
    ncs_context: StructureContext,
) -> None:
    crystallographic, strict_ncs, ncs_id, scope = ncs_context.image_provenance(
        0, (0, 0, 0)
    )
    assert (crystallographic, strict_ncs, ncs_id, scope) == (
        False,
        False,
        "",
        "explicit",
    )


def test_image_provenance_rejects_a_negative_image_index(
    ncs_context: StructureContext,
) -> None:
    with pytest.raises(ValueError, match="negative"):
        ncs_context.image_provenance(-1, (0, 0, 0))


def test_image_provenance_rejects_an_image_beyond_the_ncs_blocks(
    ncs_context: StructureContext,
) -> None:
    with pytest.raises(ValueError, match="strict-NCS"):
        ncs_context.image_provenance(12, (0, 0, 0))


def test_image_provenance_requires_symmetry_metadata(tmp_path: Path) -> None:
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)], cell=None)
    context = load_structure("test", builder.write_pdb(tmp_path / "nocell.pdb"))

    assert context.symmetry.search_available is False
    assert context.symmetry.search_failure_reason == "missing_or_invalid_unit_cell"
    assert context.symmetry.crystallographic_operation_count == 0
    with pytest.raises(ValueError, match="operation count"):
        context.image_provenance(0, (0, 0, 0))


def test_image_provenance_without_ncs_never_reports_a_strict_ncs_scope(
    tmp_path: Path,
) -> None:
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    context = load_structure("test", builder.write_pdb(tmp_path / "noncs.pdb"))

    scopes = {
        context.image_provenance(index, (0, 0, 0))[3]
        for index in range(context.symmetry.crystallographic_operation_count)
    }
    assert scopes == {"explicit", "crystallographic"}
    with pytest.raises(ValueError, match="strict-NCS"):
        context.image_provenance(
            context.symmetry.crystallographic_operation_count, (0, 0, 0)
        )


def test_image_provenance_ncs_id_tracks_the_operation_block(tmp_path: Path) -> None:
    """The reported NCS identifier is the deposited MTRIX id, not an index."""
    builder = simple_metal_site("ZN", [("HOH", "O", 2.09)])
    path = _write_pdb_with_ncs(
        builder, tmp_path / "ids.pdb", [("7", False), ("9", False)]
    )
    context = load_structure("test", path)

    operations = context.symmetry.crystallographic_operation_count
    assert context.image_provenance(operations, (0, 0, 0))[2] == "7"
    assert context.image_provenance(2 * operations, (0, 0, 0))[2] == "9"
