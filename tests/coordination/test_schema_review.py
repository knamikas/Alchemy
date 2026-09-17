"""Row-schema validation, producer coverage, and contact identity in ``schema``.

Every published column has exactly one producer. These tests pin the checks
that keep a renamed, misspelled, or duplicated column from reaching the CSVs as
a silently blanked field rather than an error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import helpers
import pytest
from helpers import atom_site

import codes
from coordination import schema
from coordination.contact_record import Candidate
from coordination.schema import (
    BOND_COLUMNS,
    CANDIDATE_COLUMNS,
    DENSITY_ROW_STATS_EXTRA_COLUMNS,
    STATS_EXTRA_COLUMNS,
    _merge_row_fields,
    _summary_supplied_columns,
    check_row_schema,
    contact_identifier,
    stats_extra_values,
    structure_extra_values,
)
from coordination.site_summary import SUMMARY_OWNED_STATS_EXTRA_COLUMNS
from structure_analysis import (
    AtomSite,
    ContactImage,
    StructureContext,
    load_structure,
)


class _PairRow(schema._CsvRow):
    """A two-column row used to exercise ``_CsvRow`` without a published schema."""

    columns = ("left", "right")
    schema_name = "pair.csv"


def _loaded_site(tmp_path: Path) -> tuple[StructureContext, AtomSite]:
    """Return a loaded single-zinc structure and its metal atom."""
    path = helpers.simple_metal_site().write_cif(tmp_path / "site.cif")
    context = load_structure("test", path)
    return context, context.metal_atoms(["ZN"])[0]


def _image(
    *,
    image_index: int = 0,
    scope: codes.ContactScope = codes.ContactScope.EXPLICIT,
    symmetry_operation: str = "1_555",
    translation: tuple[int, int, int] = (0, 0, 0),
) -> ContactImage:
    """One contact image; only the identity fields below distinguish images."""
    return ContactImage(
        distance=2.09,
        position=(2.09, 0.0, 0.0),
        crystallographic_contact=scope != codes.ContactScope.EXPLICIT,
        strict_ncs_contact=False,
        strict_ncs_operation_id="",
        scope=scope,
        image_index=image_index,
        symmetry_operation=symmetry_operation,
        translation=translation,
    )


def _candidate(neighbor: AtomSite, image: ContactImage) -> Candidate:
    """A candidate carrying only the fields ``contact_identifier`` reads."""
    return Candidate(
        neighbor=neighbor,
        image=image,
        candidate_sources={codes.CandidateSource.PROXIMITY_4A},
    )


def test_csv_row_rejects_a_value_that_is_not_a_csv_scalar() -> None:
    """A container reaching a cell would be written as its repr, so refuse it."""
    assert _PairRow({"left": "a", "right": None}).as_dict() == {
        "left": "a",
        "right": None,
    }
    with pytest.raises(TypeError, match="right is not a CSV scalar: list"):
        _PairRow({"left": "a", "right": ["b"]})


def test_check_row_schema_rejects_a_column_list_that_repeats_a_name() -> None:
    """A duplicated name would let a row omit another field and still validate."""
    with pytest.raises(RuntimeError, match="column list repeats: b"):
        check_row_schema({"a": 1, "b": 2}, ["a", "b", "b"], "duplicated.csv")


@pytest.mark.parametrize(
    ("name", "columns"),
    [
        ("BOND_COLUMNS", BOND_COLUMNS),
        ("CANDIDATE_COLUMNS", CANDIDATE_COLUMNS),
        ("STATS_EXTRA_COLUMNS", STATS_EXTRA_COLUMNS),
    ],
)
def test_published_column_lists_name_each_column_once(
    name: str, columns: list[str]
) -> None:
    """Every published header is a set of distinct names."""
    repeated = sorted({column for column in columns if columns.count(column) > 1})
    assert not repeated, f"{name} repeats {repeated}"


def test_merge_row_fields_rejects_a_field_produced_twice() -> None:
    """Splatting the field groups would let the last producer win in silence."""
    assert _merge_row_fields({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    with pytest.raises(RuntimeError, match="two row field groups both produce a"):
        _merge_row_fields({"a": 1}, {"a": 2})


def test_stats_extra_values_blanks_only_the_metal_columns_without_a_site(
    tmp_path: Path,
) -> None:
    """A density row that joined no metal site still fills the whole schema."""
    context, _metal = _loaded_site(tmp_path)
    summary: dict[str, Any] = {
        "selected_metal_site_status": codes.SelectedSiteStatus.NO_SELECTED_METAL,
        "coordinate_mapping_status": codes.CoordinateMappingStatus.MATCHED,
        "density_observation_id": "test/HOH",
    }

    values = stats_extra_values("test", context, None, summary)

    assert set(values) == set(STATS_EXTRA_COLUMNS)
    assert values["metal_site_id"] == ""
    assert values["metal_element"] == ""
    assert values["metal_altloc_options"] == ""
    assert math.isnan(values["metal_x"])
    assert math.isnan(values["metal_occupancy"])
    assert math.isnan(values["metal_conformer_mean_occupancy"])
    # The summary reports why there is no site, and the structure-level share
    # is unaffected by the missing metal.
    assert values["selected_metal_site_status"] == "no_selected_metal"
    assert values["density_observation_id"] == "test/HOH"
    assert values["model_policy"] == context.model_policy
    # A summary column the caller left out is the one blank that is allowed.
    assert values["dpi_unavailable_reason"] == ""


def test_stats_extra_values_reports_a_metal_site(tmp_path: Path) -> None:
    """With a metal the same call fills the metal columns from the atom."""
    context, metal = _loaded_site(tmp_path)

    values = stats_extra_values("test", context, metal)

    assert set(values) == set(STATS_EXTRA_COLUMNS)
    assert values["metal_site_id"].startswith("test:m0:")
    assert values["metal_element"] == "ZN"
    assert values["metal_coordinates_valid"] is True


def test_stats_extra_values_rejects_a_producer_that_drops_a_column(
    tmp_path: Path,
) -> None:
    """A dropped producer key would otherwise blank that column in every row."""
    context, metal = _loaded_site(tmp_path)
    structure_values = structure_extra_values(context)
    del structure_values["model_policy"]

    with pytest.raises(RuntimeError, match="missing model_policy"):
        stats_extra_values("test", context, metal, {}, structure_values)


def test_stats_extra_values_rejects_a_producer_that_misspells_a_column(
    tmp_path: Path,
) -> None:
    """A misspelled key is a blanked column and a dropped value, not a no-op."""
    context, metal = _loaded_site(tmp_path)
    structure_values = structure_extra_values(context)
    structure_values["model_polcy"] = structure_values.pop("model_policy")

    with pytest.raises(RuntimeError, match="unexpected model_polcy"):
        stats_extra_values("test", context, metal, {}, structure_values)


def test_stats_extra_columns_are_exactly_covered_by_their_producers(
    tmp_path: Path,
) -> None:
    """The four producing shares partition the published stats columns.

    A successful call already proves the shares do not overlap, because
    ``_merge_row_fields`` rejects a column two of them produce.
    """
    context, metal = _loaded_site(tmp_path)
    published = set(STATS_EXTRA_COLUMNS)
    structure_share = set(structure_extra_values(context))
    summary_share = _summary_supplied_columns()
    site_share = set(stats_extra_values("test", context, metal)) - (
        structure_share | summary_share
    )

    assert summary_share == SUMMARY_OWNED_STATS_EXTRA_COLUMNS | (
        DENSITY_ROW_STATS_EXTRA_COLUMNS
    )
    assert SUMMARY_OWNED_STATS_EXTRA_COLUMNS.isdisjoint(DENSITY_ROW_STATS_EXTRA_COLUMNS)
    assert structure_share.isdisjoint(summary_share)
    assert site_share.isdisjoint(structure_share | summary_share)
    assert site_share | structure_share | summary_share == published
    assert "metal_site_id" in site_share


def test_contact_identifier_separates_two_images_of_one_donor(
    tmp_path: Path,
) -> None:
    """Generated images of the same deposited atom are distinct contacts."""
    _context, metal = _loaded_site(tmp_path)
    neighbor = atom_site("O", atom_name="O", residue_name="HOH", is_water=True)
    explicit = _candidate(neighbor, _image())
    first_image = _candidate(
        neighbor,
        _image(
            image_index=1,
            scope=codes.ContactScope.CRYSTALLOGRAPHIC,
            symmetry_operation="2_555",
            translation=(0, 0, 0),
        ),
    )
    second_image = _candidate(
        neighbor,
        _image(
            image_index=2,
            scope=codes.ContactScope.CRYSTALLOGRAPHIC,
            symmetry_operation="2_555",
            translation=(1, 0, 0),
        ),
    )

    identifiers = {
        contact_identifier("test", metal, contact)
        for contact in (explicit, first_image, second_image)
    }

    assert len(identifiers) == 3
    # Every identifier joins back to the metal site it belongs to.
    site = schema.metal_site_identifier("test", metal)
    assert all(identifier.startswith(f"{site}:c") for identifier in identifiers)


def test_contact_identifier_is_stable_across_calls(tmp_path: Path) -> None:
    """The digest is a join key, so it must not vary between rows or runs."""
    _context, metal = _loaded_site(tmp_path)
    neighbor = atom_site("O", atom_name="O", residue_name="HOH", is_water=True)
    contact = _candidate(neighbor, _image())
    same = _candidate(neighbor, _image())

    identifier = contact_identifier("test", metal, contact)

    assert identifier == contact_identifier("test", metal, contact)
    assert identifier == contact_identifier("test", metal, same)
    assert len(identifier.rsplit(":c", 1)[1]) == 24
