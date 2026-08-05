"""EDSTATS table parsing and metal/cofactor identification.

EDSTATS emits a fixed 42-column table plus a synthetic model-separator row, and
for a chain-less coordinate file it omits the empty trailing CP field, so those
rows arrive with 41 fields. It also relabels every ordered water's CI field to
``0`` whatever the real chain, so restoring the missing field must key on the
row *shape*: a CI-gated guard rejects every water row and fails the entry.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

import confidence_score
import helpers
import worker
from driver.writers import STATS_COLUMNS
from helpers import AtomSpec, StructureBuilder, simple_metal_site

from metal_elements import METAL_ELEMENTS
from metal_identification import (
    EDSTATS_COLUMNS,
    EDSTATS_METRIC_COLUMNS,
    _classify_residue,
    _density_observation_id,
    _expected_edstats_residues,
    _is_edstats_separator,
    _normalize_edstats_row,
    _sigma_for,
    _sigma_index,
    _validate_edstats_row,
    _validated_edstats_header,
    _zd_indices,
    extract_metal_statistics,
)
from structure_analysis import load_structure


HEADER = list(helpers.EDSTATS_HEADER)
INDICES = _validated_edstats_header(HEADER)


def _normalize(fields):
    """``_normalize_edstats_row`` against the standard schema."""
    return _normalize_edstats_row(list(fields), HEADER, INDICES)


def _validate(fields, line_number=2):
    """``_validate_edstats_row`` against the standard schema."""
    return _validate_edstats_row(list(fields), HEADER, INDICES, line_number)


def _extract(stats_path, context, *, pdb_id="test", metals=None, cofactors=()):
    """``extract_metal_statistics`` with the usual metal set and no cofactors."""
    return extract_metal_statistics(
        pdb_id,
        str(stats_path),
        set(METAL_ELEMENTS) if metals is None else set(metals),
        set(cofactors),
        structure=context,
    )


def _write_rows(tmp_path, rows, name="stats.out", **kwargs):
    return helpers.write_edstats(tmp_path / name, rows, **kwargs)


class _RemappedStructure:
    """Read-only view of a real context with one author key resolving twice.

    Gemmi merges coordinate residues that share an author name, chain and
    sequence number, so a genuine duplicate cannot be produced from a file.
    """

    def __init__(self, context, author_key, residues, all_residues=None):
        self._context = context
        self._author_key = tuple(author_key)
        self._residues = tuple(residues)
        self._all_residues = (
            self._residues if all_residues is None else tuple(all_residues)
        )

    @property
    def model_analyzed(self):
        return self._context.model_analyzed

    @property
    def residues(self):
        """Every coordinate residue, which is the ambiguous pair unless the
        caller needs unrelated residues in the model as well."""
        return self._all_residues

    def residues_for_coordinate_author(self, residue_name, chain_id, resnum):
        if (residue_name, chain_id, resnum) == self._author_key:
            return self._residues
        return self._context.residues_for_coordinate_author(
            residue_name, chain_id, resnum
        )

    def __getattr__(self, name):
        return getattr(self._context, name)


def test_valid_header_maps_every_documented_column_to_its_position():
    """The standard header validates and yields the schema's column indices."""
    indices = _validated_edstats_header(list(helpers.EDSTATS_HEADER))

    # The fixture literal is independent of the production constants, so a
    # reorder fails here even if parser and constants change together.
    assert tuple(EDSTATS_COLUMNS) == helpers.EDSTATS_HEADER
    assert tuple(EDSTATS_METRIC_COLUMNS) == helpers.EDSTATS_METRICS
    assert len(helpers.EDSTATS_HEADER) == 42
    assert len(helpers.EDSTATS_METRICS) == 36
    assert indices == {
        name: position for position, name in enumerate(helpers.EDSTATS_HEADER)
    }
    assert (indices["RT"], indices["CI"], indices["RN"]) == (0, 1, 2)
    assert (indices["MN"], indices["CP"], indices["NR"]) == (39, 40, 41)


def _header_with(**changes):
    fields = list(helpers.EDSTATS_HEADER)
    for name, replacement in changes.items():
        fields[fields.index(name)] = replacement
    return fields


@pytest.mark.parametrize(
    "fields, fragment",
    [
        pytest.param(
            _header_with(CP="CI"), "duplicate columns: CI", id="duplicate-column"
        ),
        pytest.param(
            list(helpers.EDSTATS_HEADER)[:-1], "missing NR", id="missing-column"
        ),
        pytest.param(_header_with(BAm="BAX"), "unexpected BAX", id="unexpected-column"),
        pytest.param(
            list(helpers.EDSTATS_HEADER) + ["EXTRA"],
            "unexpected EXTRA",
            id="extra-column",
        ),
        pytest.param(
            ["CI", "RT", *helpers.EDSTATS_HEADER[2:]],
            "not in the standard order",
            id="reordered",
        ),
        pytest.param([], "missing", id="empty"),
    ],
)
def test_header_deviations_from_the_fixed_schema_are_rejected(fields, fragment):
    """Alchemy pins the EDSTATS schema; any deviation must fail loudly."""
    with pytest.raises(ValueError) as excinfo:
        _validated_edstats_header(list(fields))
    assert fragment in str(excinfo.value)


def test_a_duplicate_column_is_reported_before_the_ordering_check():
    """A repeated column is ambiguous, so it is named rather than reordered."""
    fields = _header_with(CP="RT")
    with pytest.raises(ValueError, match="duplicate columns: RT"):
        _validated_edstats_header(fields)


def test_extract_rejects_a_reordered_header_file(tmp_path):
    """A stats.out whose columns were shuffled fails the entry, not silently."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    rows = helpers.edstats_rows_for_structure(context)
    swapped = ["CI", "RT", *helpers.EDSTATS_HEADER[2:]]
    stats = _write_rows(tmp_path, rows, header=swapped)

    with pytest.raises(ValueError, match="not in the standard order"):
        _extract(stats, context)


def test_a_wellformed_row_validates_and_returns_its_model_number():
    """Validation reports the row's MN so the caller can enforce model policy."""
    row = helpers.edstats_row("ZN", "B", "1", mn=3, metrics={"ZDm": -2.5})
    assert _validate(row) == 3


@pytest.mark.parametrize(
    "width, mutate",
    [
        pytest.param(41, lambda row: row.pop(3), id="41-interior-field-dropped"),
        pytest.param(40, lambda row: [row.pop(), row.pop()], id="40-two-short"),
        pytest.param(43, lambda row: row.append("0.0"), id="43-one-long"),
    ],
)
def test_row_width_must_match_the_header(width, mutate):
    """Only the omitted trailing CP field is recoverable; other widths fail."""
    row = helpers.edstats_row("ZN", "B", "1")
    mutate(row)
    assert len(row) == width
    normalized = _normalize(row)

    with pytest.raises(ValueError) as excinfo:
        _validate(normalized, line_number=7)
    message = str(excinfo.value)
    assert f"row 7 has {width} columns" in message
    assert "expected 42" in message


@pytest.mark.parametrize("null", ["n/a", "N/A", "N/a"])
def test_the_documented_null_marker_is_accepted_for_every_metric(null):
    """``n/a`` is EDSTATS' null for an uncomputable statistic, case-blind."""
    row = helpers.edstats_row("ZN", "B", "1", default=null)
    assert _validate(row) == 1
    assert row[INDICES["ZDm"]] == null


@pytest.mark.parametrize("value", ["abc", "1.2.3", "--5", "n/a/", "2,5", "?"])
def test_nonnumeric_statistics_are_rejected_and_name_the_column(value):
    """A metric that is neither a number nor ``n/a`` fails the entry."""
    row = helpers.edstats_row("ZN", "B", "1", metrics={"CCPa": value})
    with pytest.raises(ValueError) as excinfo:
        _validate(row, line_number=4)
    message = str(excinfo.value)
    assert "nonnumeric CCPa" in message
    assert repr(value) in message
    assert "row 4" in message


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "NaN", "Infinity", "1e999"])
def test_non_finite_statistics_are_rejected(value):
    """README: the table must contain *finite* numbers or the null marker."""
    row = helpers.edstats_row("ZN", "B", "1", metrics={"ZD-s": value})
    with pytest.raises(ValueError) as excinfo:
        _validate(row)
    assert "non-finite ZD-s" in str(excinfo.value)


@pytest.mark.parametrize("metric", ["BAm", "ZOs", "ZD+a", "NPa"])
def test_every_metric_column_is_checked_not_just_the_first(metric):
    """All 36 statistics are validated, main-chain, side-chain and all-atom."""
    row = helpers.edstats_row("ZN", "B", "1", metrics={metric: "oops"})
    with pytest.raises(ValueError) as excinfo:
        _validate(row)
    assert f"nonnumeric {metric} value" in str(excinfo.value)


@pytest.mark.parametrize("value", ["1.0", "x", "n/a", "-", ""])
def test_a_model_number_that_is_not_an_integer_is_rejected(value):
    """MN selects the model; a float or marker cannot be silently truncated."""
    row = helpers.edstats_row("ZN", "B", "1")
    row[INDICES["MN"]] = value
    with pytest.raises(ValueError) as excinfo:
        _validate(row, line_number=9)
    assert "invalid EDSTATS MN model value on row 9" in str(excinfo.value)


def test_rows_from_another_model_fail_the_entry(tmp_path):
    """Alchemy analyses model 1 only; density from model 2 is never mixed in."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    assert context.model_analyzed == 1
    stats = _write_rows(tmp_path, helpers.edstats_rows_for_structure(context, model=2))

    with pytest.raises(ValueError) as excinfo:
        _extract(stats, context)
    message = str(excinfo.value)
    assert "returned model 2" in message
    assert "selected model 1" in message


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(helpers.edstats_separator_row(), id="39-field"),
        pytest.param(
            helpers.edstats_separator_row(model=2, row_number=17),
            id="39-field-other-model",
        ),
        pytest.param(["_", *(["N/A"] * 36), "1", "1"], id="uppercase-null"),
        pytest.param(["_"], id="bare-marker"),
    ],
)
def test_model_separator_rows_are_recognized(fields):
    """EDSTATS' synthetic MODEL row is skipped, in both captured shapes."""
    assert _is_edstats_separator(list(fields))


@pytest.mark.parametrize(
    "fields, why",
    [
        pytest.param(helpers.edstats_row("ZN", "B", "1"), "a real residue row"),
        pytest.param(
            ["_", *(["n/a"] * 35), "0.0", "1", "1"], "a numeric metric is not a null"
        ),
        pytest.param(
            ["ZN", *(["n/a"] * 36), "1", "1"], "the leading marker is a residue name"
        ),
        pytest.param(
            ["_", *(["n/a"] * 36), "1", "x"], "the row number is not an integer"
        ),
        pytest.param(["_", *(["n/a"] * 36), "1"], "one field short of a separator"),
        pytest.param(["_", "1"], "a bare marker with anything appended"),
        pytest.param([], "an empty field list"),
    ],
)
def test_rows_that_only_resemble_a_separator_are_left_to_validation(fields, why):
    """Recognition is semantic, so malformed rows still reach row validation."""
    assert not _is_edstats_separator(list(fields)), why


def test_extract_skips_separator_rows_without_counting_them(tmp_path):
    """A MODEL separator contributes no residue and no density observation."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    residue_rows = helpers.edstats_rows_for_structure(context)
    mixed = [helpers.edstats_separator_row(row_number=0)]
    mixed.extend(residue_rows)
    mixed.append(["_"])
    stats = _write_rows(tmp_path, mixed)

    rows, header = _extract(stats, context)
    plain, _ = _extract(_write_rows(tmp_path, residue_rows, name="plain.out"), context)

    assert header == list(helpers.EDSTATS_HEADER)
    assert [row["density_observation_id"] for row in rows] == [
        row["density_observation_id"] for row in plain
    ]
    assert [row["resname"] for row in rows] == ["ZN"]


def test_a_file_of_separators_alone_has_no_residue_rows(tmp_path):
    """Separator-only output is not a successful parse of zero metals."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats = _write_rows(tmp_path, [helpers.edstats_separator_row(), ["_"]])

    with pytest.raises(ValueError, match="no residue rows"):
        _extract(stats, context)


@pytest.mark.parametrize(
    "ci, kind",
    [
        pytest.param("0", "ordered water (EDSTATS relabels CI to 0)"),
        pytest.param("_", "non-water residue with no chain identifier"),
    ],
)
def test_omitted_trailing_chain_field_is_restored_whatever_ci_says(ci, kind):
    """Restoration keys on the row shape, never on CI.

    EDSTATS reports every ordered water as chain ``0`` regardless of its real
    chain, so a CI-gated guard restores the non-water rows and rejects every
    water row of the same blank-chain file.
    """
    row = helpers.edstats_row(
        "HOH" if ci == "0" else "ZN", ci, "101", nr=7, omit_cp=True
    )
    assert len(row) == 41, kind

    normalized = _normalize(row)

    assert len(normalized) == 42
    assert normalized[INDICES["CP"]] == ""
    assert normalized[INDICES["NR"]] == "7"
    assert normalized[INDICES["MN"]] == "1"
    assert _validate(normalized) == 1


@pytest.mark.parametrize("marker", ["", ".", "?", "_"])
def test_missing_chain_markers_normalize_to_blank(marker):
    """The several missing-chain spellings collapse to one canonical blank."""
    row = helpers.edstats_row("ZN", marker or "_", "1", cp=marker)
    row[INDICES["CI"]] = marker
    normalized = _normalize(row)
    assert normalized[INDICES["CI"]] == ""
    assert normalized[INDICES["CP"]] == ""


def test_a_water_group_label_is_not_mistaken_for_a_chain_marker():
    """CI ``0`` is EDSTATS' water group label and stays as reported."""
    row = helpers.edstats_row("HOH", "0", "101", cp="A")
    normalized = _normalize(row)
    assert normalized[INDICES["CI"]] == "0"
    assert normalized[INDICES["CP"]] == "A"


def test_a_complete_row_is_passed_through_untouched():
    """The 42-field form must not be rewritten by the restoration path."""
    row = helpers.edstats_row("ZN", "B", "1", cp="B", nr=4, metrics={"ZDm": 2.5})
    assert _normalize(row) == row


def test_a_short_row_that_is_not_the_blank_chain_shape_stays_short():
    """Only the unambiguous CP-omitted shape is restored; other losses fail."""
    row = helpers.edstats_row("ZN", "B", "1")
    row.pop(3)  # drop an interior metric instead of the trailing chain
    assert len(row) == 41 and row[-2:] == ["B", "1"]

    normalized = _normalize(row)

    assert len(normalized) == 41
    with pytest.raises(ValueError, match="has 41 columns"):
        _validate(normalized)


def test_blank_chain_entry_with_omitted_cp_parses_end_to_end(tmp_path):
    """A chain-less entry must not fail on its waters.

    Every row of a blank-chain coordinate file arrives with 41 fields; the
    waters carry CI ``0`` and everything else CI ``_``.
    """
    builder = StructureBuilder()
    builder.add_amino_acid("HIS", 10, chain="", positions={"NE2": (2.03, 0, 0)})
    builder.add_metal("ZN", 1, chain="", pos=(0.0, 0.0, 0.0))
    builder.add_water(101, (0.0, 2.09, 0.0), chain="")
    builder.add_water(102, (0.0, -2.09, 0.0), chain="")
    path = builder.write_pdb(tmp_path / "blank.pdb")

    context = load_structure("test", path)
    assert {residue.chain_id for residue in context.residues} == {""}
    stats = helpers.write_edstats_for_structure(
        tmp_path / "stats.out", context, blank_chain_form=True
    )

    lines = [
        line.split() for line in open(stats, encoding="utf-8").read().splitlines()[1:]
    ]
    assert [len(fields) for fields in lines] == [41, 41, 41, 41]
    water_rows = [fields for fields in lines if fields[0] == "HOH"]
    assert len(water_rows) == 2
    assert {fields[1] for fields in water_rows} == {"0"}
    assert {fields[1] for fields in lines if fields[0] != "HOH"} == {"_"}

    rows, header = _extract(stats, context)

    assert header == list(helpers.EDSTATS_HEADER)
    assert [(row["resname"], row["chain"], row["resnum"]) for row in rows] == [
        ("ZN", "", "1")
    ]
    assert rows[0]["fields"][INDICES["CP"]] == ""
    assert rows[0]["density_observation_id"].count("chain=_") == 1


def test_density_observation_id_names_the_edstats_row_not_an_atom():
    """The identifier is residue-level and case-normalized on the entry id."""
    fields = _normalize(helpers.edstats_row("FES", "B", "5", mn=1, nr=12))
    observation = _density_observation_id("1ABC", fields, INDICES)

    assert observation == (
        "1abc/model=1/chain=B/residue=5/component=FES/edstats_row=12"
    )
    assert observation == _density_observation_id("1ABC", fields, INDICES)


def test_density_observation_id_distinguishes_repeated_author_identifiers():
    """NR disambiguates two EDSTATS rows with the same author residue id."""
    first = _normalize(helpers.edstats_row("ZN", "B", "1", nr=1))
    second = _normalize(helpers.edstats_row("ZN", "B", "1", nr=2))
    assert _density_observation_id("test", first, INDICES) != _density_observation_id(
        "test", second, INDICES
    )


def test_density_observation_id_renders_a_blank_chain_explicitly():
    """A blank chain becomes ``_`` so the identifier keeps six labelled parts."""
    fields = _normalize(helpers.edstats_row("ZN", "_", "1", omit_cp=True))
    observation = _density_observation_id("test", fields, INDICES)
    assert observation == "test/model=1/chain=_/residue=1/component=ZN/edstats_row=1"
    assert len(observation.split("/")) == 6


def test_distinct_metal_residues_get_distinct_observation_ids(tmp_path):
    """Two separate ions are two density observations, neither one shared."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_metal("MG", 2, chain="B", pos=(12.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "two.pdb")

    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(tmp_path / "stats.out", context)
    rows, _ = _extract(stats, context)

    assert [row["resname"] for row in rows] == ["ZN", "MG"]
    assert len({row["density_observation_id"] for row in rows}) == 2
    assert all(row["density_shared_site_count"] == 1 for row in rows)
    assert not any(row["density_is_shared"] for row in rows)


def test_shared_cofactor_repeats_one_observation_once_per_metal_site(tmp_path):
    """A 2Fe cluster emits two site rows over a single density observation."""
    builder = StructureBuilder()
    builder.add_hetero_residue(
        "FES",
        3,
        [
            AtomSpec("FE1", "FE", (0.0, 0.0, 0.0)),
            AtomSpec("FE2", "FE", (2.7, 0.0, 0.0)),
            AtomSpec("S1", "S", (1.35, 1.8, 0.0)),
            AtomSpec("S2", "S", (1.35, -1.8, 0.0)),
        ],
        chain="B",
    )
    path = builder.write_pdb(tmp_path / "fes.pdb")

    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(
        tmp_path / "stats.out", context, metrics={"ZDm": 1.75}
    )
    rows, header = _extract(stats, context, cofactors={"FES"})

    assert [row["category"] for row in rows] == ["cofactor", "cofactor"]
    assert [row["site"].atom_name for row in rows] == ["FE1", "FE2"]
    assert len({row["density_observation_id"] for row in rows}) == 1
    assert all(row["density_shared_site_count"] == 2 for row in rows)
    assert all(row["density_is_shared"] for row in rows)
    assert all(row["density_scope"] == "cofactor_residue" for row in rows)
    assert {row["fields"][header.index("ZDm")] for row in rows} == {"1.75"}
    assert len({row["site_key"] for row in rows}) == 2
    assert len({row["residue_key"] for row in rows}) == 1


@pytest.fixture(scope="module")
def classification_context(tmp_path_factory):
    """One structure holding every classification case, loaded once."""
    directory = tmp_path_factory.mktemp("classify")
    builder = StructureBuilder()
    builder.add_amino_acid("HIS", 10, chain="A")
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    # A metal-ion CCD id that is not itself an element symbol.
    builder.add_metal("FE", 2, chain="B", pos=(10.0, 0.0, 0.0), resname="FE2")
    builder.add_water(101, (20.0, 0.0, 0.0), chain="B")
    # Nitric oxide: the component id "NO" collides with nobelium.
    builder.add_hetero_residue(
        "NO",
        3,
        [
            AtomSpec("N", "N", (30.0, 0.0, 0.0)),
            AtomSpec("O", "O", (31.1, 0.0, 0.0)),
        ],
        chain="B",
    )
    # A hydrated ion: metal-containing, but not a bare ion.
    builder.add_hetero_residue(
        "MZN",
        4,
        [
            AtomSpec("ZN", "ZN", (40.0, 0.0, 0.0)),
            AtomSpec("O1", "O", (42.0, 0.0, 0.0)),
        ],
        chain="B",
    )
    builder.add_hetero_residue(
        "FES",
        5,
        [
            AtomSpec("FE1", "FE", (50.0, 0.0, 0.0)),
            AtomSpec("FE2", "FE", (52.7, 0.0, 0.0)),
            AtomSpec("S1", "S", (51.3, 1.8, 0.0)),
            AtomSpec("S2", "S", (51.3, -1.8, 0.0)),
        ],
        chain="B",
    )
    path = builder.write_pdb(directory / "classify.pdb")
    return load_structure("test", path)


@pytest.mark.parametrize(
    "resname, cofactors, category, site_count",
    [
        pytest.param("ZN", set(), "metal", 1, id="monatomic-ion"),
        pytest.param("FE2", set(), "metal", 1, id="ion-ccd-id-is-not-an-element"),
        pytest.param("FES", {"FES"}, "cofactor", 2, id="catalogued-cluster"),
        pytest.param("FES", set(), "", 2, id="cluster-absent-from-the-catalog"),
        pytest.param("MZN", set(), "", 1, id="multi-atom-component-is-not-an-ion"),
        pytest.param("MZN", {"MZN"}, "cofactor", 1, id="multi-atom-catalogued"),
        pytest.param("NO", set(), "", 0, id="component-id-collides-with-an-element"),
        pytest.param("HOH", set(), "", 0, id="water"),
        pytest.param("HIS", set(), "", 0, id="amino-acid"),
        pytest.param("ZN", {"ZN"}, "cofactor", 1, id="catalog-outranks-the-ion-rule"),
    ],
)
def test_residue_classification(
    classification_context, resname, cofactors, category, site_count
):
    """Category follows the atoms' elements and the catalog, not the CCD id."""
    residue = next(
        r for r in classification_context.residues if r.residue_name == resname
    )
    metals = {element.upper() for element in METAL_ELEMENTS}

    result_category, metal_sites = _classify_residue(residue, metals, cofactors)

    assert result_category == category
    assert len(metal_sites) == site_count
    assert all(site.element in metals for site in metal_sites)


def test_an_element_outside_the_configured_set_is_not_a_metal(classification_context):
    """Classification is driven by the caller's metal set, not a fixed list."""
    residue = next(r for r in classification_context.residues if r.residue_name == "ZN")
    assert _classify_residue(residue, {"CU"}, set()) == ("", [])
    assert _classify_residue(residue, {"ZN"}, set())[0] == "metal"


def test_an_ion_split_over_two_conformers_is_still_one_chemical_site(tmp_path):
    """Alternate conformers of one ion are one chemical atom site, so a metal."""
    builder = StructureBuilder()
    zinc = builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_conformers(zinc, [("A", 0.4, {}), ("B", 0.6, {"ZN": (0.7, 0, 0)})])
    path = builder.write_pdb(tmp_path / "altmetal.pdb")

    context = load_structure("test", path)
    residue = context.residues[0]
    assert len(residue.source_atoms) == 2
    assert residue.chemical_atom_site_count == 1
    assert residue.selected_altloc == "B"

    category, sites = _classify_residue(residue, {"ZN"}, set())
    assert category == "metal"
    assert [site.altloc for site in sites] == ["B"]


def test_zero_occupancy_ion_is_not_a_selected_density_site(tmp_path):
    """A deposited zero contributes no metal-site density evidence."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", occupancy=0.0)
    context = load_structure("test", builder.write_pdb(tmp_path / "zero.pdb"))
    residue = context.residues[0]

    assert _classify_residue(residue, {"ZN"}, set()) == ("", [])
    assert _expected_edstats_residues(context, {"ZN"}, set()) == {}


def test_only_metal_and_cofactor_residues_produce_rows(tmp_path):
    """Donor residues and waters are density-scored but never emitted."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    assert {r.residue_name for r in context.residues} == {"ZN", "HIS", "ASP", "HOH"}
    stats = helpers.write_edstats_for_structure(
        tmp_path / "stats.out",
        context,
        metrics={"ZDm": 2.5},
        per_residue={("ZN", "B", "1"): {"ZD-m": -3.25}},
    )

    rows, header = _extract(stats, context)

    assert len(rows) == 1
    row = rows[0]
    assert (row["pdbID"], row["category"], row["resname"]) == ("test", "metal", "ZN")
    assert (row["chain"], row["resnum"]) == ("B", "1")
    assert row["density_scope"] == "metal_residue"
    assert row["density_shared_site_count"] == 1
    assert row["density_is_shared"] is False
    assert row["coordinate_mapping_status"] == "matched"
    assert row["selected_metal_site_status"] == "selected"
    assert row["fields"][header.index("ZD-m")] == "-3.25"
    assert row["fields"][header.index("RT")] == "ZN"

    residue = next(r for r in context.residues if r.residue_name == "ZN")
    assert row["residue_key"] == residue.key
    assert row["site"].atom_name == "ZN"
    assert row["site_key"] == residue.contact_atoms[0].source_key


def test_null_statistics_reach_the_emitted_row_unchanged(tmp_path):
    """An uncomputable statistic stays ``n/a`` instead of becoming a number."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(
        tmp_path / "stats.out", context, default=helpers.EDSTATS_NULL
    )

    rows, header = _extract(stats, context)

    assert len(rows) == 1
    metrics = {
        name: rows[0]["fields"][header.index(name)] for name in helpers.EDSTATS_METRICS
    }
    assert set(metrics.values()) == {"n/a"}


def test_the_metal_element_set_is_matched_case_insensitively(tmp_path):
    """Callers may pass lower-case element symbols; matching upper-cases them."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(tmp_path / "stats.out", context)

    rows, _ = _extract(stats, context, metals={"zn"})
    assert [row["resname"] for row in rows] == ["ZN"]


def test_an_entry_with_no_configured_metal_yields_no_rows(tmp_path):
    """Zero metal sites is a valid, complete parse -- not an error."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(tmp_path / "stats.out", context)

    rows, header = _extract(stats, context, metals={"CU"})
    assert rows == []
    assert header == list(helpers.EDSTATS_HEADER)


def test_a_cofactor_without_a_selected_metal_keeps_one_diagnostic_row(tmp_path):
    """The residue-level observation survives, flagged as siteless."""
    builder = StructureBuilder()
    builder.add_hetero_residue(
        "XCF",
        7,
        [
            AtomSpec("C1", "C", (0.0, 0.0, 0.0)),
            AtomSpec("N1", "N", (1.4, 0.0, 0.0)),
        ],
        chain="B",
    )
    path = builder.write_pdb(tmp_path / "apo.pdb")

    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(tmp_path / "stats.out", context)
    rows, _ = _extract(stats, context, cofactors={"XCF"})

    assert len(rows) == 1
    row = rows[0]
    assert (row["category"], row["resname"]) == ("cofactor", "XCF")
    assert row["site"] is None and row["site_key"] is None
    assert row["selected_metal_site_status"] == "no_selected_metal"
    assert row["coordinate_mapping_status"] == "matched"
    assert row["density_shared_site_count"] == 0
    assert row["density_is_shared"] is False
    assert row["residue_key"] == context.residues[0].key


def test_a_cofactor_row_with_no_coordinate_residue_reports_a_join_failure(tmp_path):
    """A row Alchemy cannot map is kept, but never claims a site."""
    path = simple_metal_site("ZN", [("HIS", "NE2", 2.03)]).write_pdb(
        tmp_path / "site.pdb"
    )
    context = load_structure("test", path)
    rows_in = helpers.edstats_rows_for_structure(context)
    rows_in.append(helpers.edstats_row("FES", "C", "77", nr=99))
    stats = _write_rows(tmp_path, rows_in)

    rows, _ = _extract(stats, context, cofactors={"FES"})

    metal, phantom = rows
    assert metal["resname"] == "ZN"
    assert phantom["category"] == "cofactor"
    assert (phantom["resname"], phantom["chain"], phantom["resnum"]) == (
        "FES",
        "C",
        "77",
    )
    assert phantom["coordinate_mapping_status"] == "coordinate_residue_not_found"
    assert phantom["selected_metal_site_status"] == "no_selected_metal"
    assert phantom["site"] is None and phantom["residue_key"] is None
    assert phantom["density_shared_site_count"] == 0


def _repeated_coordinate_identity(context):
    first, second = context.residues
    second = replace(
        second,
        coordinate_residue_name="ZN",
        coordinate_chain_id="B",
        coordinate_residue_number=1,
        coordinate_resnum="1",
    )
    return _RemappedStructure(context, ("ZN", "B", "1"), (first, second))


def test_nr_maps_repeated_author_rows_one_to_one(tmp_path):
    """Two observations and two residues produce two rows, never four."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_metal("ZN", 1, chain="C", pos=(12.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "twochain.pdb")
    context = load_structure("test", path)
    duplicated = _repeated_coordinate_identity(context)

    stats = _write_rows(
        tmp_path,
        [
            helpers.edstats_row("ZN", "B", "1", nr=1, metrics={"ZDm": 2.0}),
            helpers.edstats_row("ZN", "B", "1", nr=2, metrics={"ZDm": 8.0}),
        ],
    )
    rows, header = _extract(stats, duplicated)

    assert len(rows) == 2
    assert all(row["coordinate_mapping_status"] == "matched" for row in rows)
    assert all(row["selected_metal_site_status"] == "selected" for row in rows)
    assert len({row["density_observation_id"] for row in rows}) == 2
    assert all(row["density_shared_site_count"] == 1 for row in rows)
    assert len({row["residue_key"] for row in rows}) == 2
    assert len({row["site_key"] for row in rows}) == 2
    assert [row["fields"][header.index("ZDm")] for row in rows] == ["2.0", "8.0"]

    sigma_index = _sigma_index(rows)
    zd_indices = _zd_indices(header)
    assert [
        _sigma_for(
            sigma_index,
            row["resname"],
            row["chain"],
            row["resnum"],
            zd_indices,
            site_key=row["site_key"],
        )[0]
        for row in rows
    ] == [2.0, 8.0]

    worker._append_site_fields(rows, {}, duplicated)
    confidence_inputs = confidence_score.prepare_result_confidence_inputs(
        rows, [], STATS_COLUMNS
    )
    assert len(confidence_inputs) == 2
    assert (
        len(
            {
                (
                    row["metal_model_index"],
                    row["metal_chain_index"],
                    row["metal_residue_index"],
                    row["metal_atom_index"],
                )
                for row in confidence_inputs
            }
        )
        == 2
    )


def test_repeated_author_identity_completeness_keeps_multiplicity(tmp_path):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_metal("ZN", 1, chain="C", pos=(12.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "twochain.pdb")
    context = load_structure("test", path)
    duplicated = _repeated_coordinate_identity(context)
    stats = _write_rows(tmp_path, [helpers.edstats_row("ZN", "B", "1", nr=1)])

    with pytest.raises(ValueError, match="incomplete"):
        _extract(stats, duplicated)


def test_nr_must_resolve_an_ambiguous_author_identity(tmp_path):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B", pos=(0.0, 0.0, 0.0))
    builder.add_metal("ZN", 1, chain="C", pos=(12.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "twochain.pdb")
    context = load_structure("test", path)
    duplicated = _repeated_coordinate_identity(context)
    stats = _write_rows(tmp_path, [helpers.edstats_row("ZN", "B", "1", nr=3)])

    with pytest.raises(ValueError, match=r"NR 3.*does not resolve"):
        _extract(stats, duplicated)


@pytest.mark.parametrize("nr", ["0", "-1", "not-a-number"])
def test_nr_must_be_a_positive_integer(tmp_path, nr):
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    path = builder.write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    row = helpers.edstats_row("ZN", "B", "1", nr=1)
    row[INDICES["NR"]] = nr
    stats = _write_rows(tmp_path, [row])

    with pytest.raises(ValueError, match="invalid EDSTATS NR"):
        _extract(stats, context)


def test_nr_restarting_per_chain_is_the_normal_case_not_a_duplicate(tmp_path):
    """EDSTATS numbers NR within each chain, so every chain begins again at 1.

    Reading NR as an ordinal over the whole model made the first row of the
    second chain collide with the first row of the first chain, which failed
    every multi-chain entry outright.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="A", pos=(0.0, 0.0, 0.0))
    builder.add_metal("MG", 1, chain="B", pos=(12.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "twochain.pdb")
    context = load_structure("test", path)
    stats = _write_rows(
        tmp_path,
        [
            helpers.edstats_row("ZN", "A", "1", nr=1),
            helpers.edstats_row("MG", "B", "1", nr=1),
        ],
    )

    rows, _ = _extract(stats, context)

    assert [(row["resname"], row["chain"]) for row in rows] == [
        ("ZN", "A"),
        ("MG", "B"),
    ]
    assert len({row["density_observation_id"] for row in rows}) == 2


def test_nr_is_resolved_within_its_own_chain(tmp_path):
    """A repeated author identity resolves by position in its chain.

    Chain B's rows restart at NR 1, so indexing a model-wide residue list would
    land on chain A's residue and resolve nothing.
    """
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="A", pos=(0.0, 0.0, 0.0))
    builder.add_metal("MG", 1, chain="B", pos=(12.0, 0.0, 0.0))
    builder.add_metal("MG", 2, chain="B", pos=(24.0, 0.0, 0.0))
    path = builder.write_pdb(tmp_path / "twochain.pdb")
    context = load_structure("test", path)

    first, second, third = context.residues
    third = replace(
        third,
        coordinate_residue_name="MG",
        coordinate_chain_id="B",
        coordinate_residue_number=1,
        coordinate_resnum="1",
    )
    duplicated = _RemappedStructure(
        context,
        ("MG", "B", "1"),
        (second, third),
        all_residues=(first, second, third),
    )

    stats = _write_rows(
        tmp_path,
        [
            helpers.edstats_row("ZN", "A", "1", nr=1),
            helpers.edstats_row("MG", "B", "1", nr=1, metrics={"ZDm": 2.0}),
            helpers.edstats_row("MG", "B", "1", nr=2, metrics={"ZDm": 8.0}),
        ],
    )
    rows, _ = _extract(stats, duplicated)

    assert [row["fields"][INDICES["ZDm"]] for row in rows] == ["0.0", "2.0", "8.0"]
    assert [row["resnum"] for row in rows] == ["1", "1", "2"]
    assert len({row["density_observation_id"] for row in rows}) == 3


def test_duplicate_nr_fails_instead_of_reusing_a_coordinate_residue(tmp_path):
    """Within one chain NR must still be unique; both rows here are chain B."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 1, chain="B")
    builder.add_metal("MG", 2, chain="B")
    path = builder.write_pdb(tmp_path / "two.pdb")
    context = load_structure("test", path)
    stats = _write_rows(
        tmp_path,
        [
            helpers.edstats_row("ZN", "B", "1", nr=1),
            helpers.edstats_row("MG", "B", "2", nr=1),
        ],
    )

    with pytest.raises(ValueError, match="duplicate EDSTATS NR 1"):
        _extract(stats, context)


def test_insertion_coded_residue_ids_join_after_canonicalization(tmp_path):
    """EDSTATS writes ``10:A``; Alchemy joins it to the author id ``10A``."""
    builder = StructureBuilder()
    builder.add_metal("ZN", 10, chain="B", pos=(0.0, 0.0, 0.0), icode="A")
    path = builder.write_pdb(tmp_path / "icode.pdb")
    context = load_structure("test", path)
    assert context.residues[0].resnum == "10A"

    stats = _write_rows(tmp_path, [helpers.edstats_row("ZN", "B", "10:A")])
    rows, _ = _extract(stats, context)

    assert [row["resnum"] for row in rows] == ["10A"]
    assert rows[0]["fields"][INDICES["RN"]] == "10A"
    assert "residue=10A" in rows[0]["density_observation_id"]


@pytest.mark.parametrize("residue_id", ["1:AB", "12345A", "10*", "-"])
def test_an_undecodable_residue_identifier_fails_the_entry(tmp_path, residue_id):
    """A residue number Alchemy cannot decode must not silently drop the row."""
    path = simple_metal_site("ZN", [("HIS", "NE2", 2.03)]).write_pdb(
        tmp_path / "site.pdb"
    )
    context = load_structure("test", path)
    rows_in = helpers.edstats_rows_for_structure(context)
    rows_in.append(helpers.edstats_row("HOH", "B", residue_id, nr=50))
    stats = _write_rows(tmp_path, rows_in)

    with pytest.raises(ValueError) as excinfo:
        _extract(stats, context)
    assert "invalid EDSTATS RN residue identifier" in str(excinfo.value)


def test_a_missing_metal_residue_fails_the_entry(tmp_path):
    """Incomplete EDSTATS output fails rather than under-reporting metals."""
    path = simple_metal_site("ZN", [("HOH", "O", 2.09)]).write_pdb(
        tmp_path / "site.pdb"
    )
    context = load_structure("test", path)
    without_metal = [
        row for row in helpers.edstats_rows_for_structure(context) if row[0] != "ZN"
    ]
    assert without_metal
    stats = _write_rows(tmp_path, without_metal)

    with pytest.raises(ValueError) as excinfo:
        _extract(stats, context)
    message = str(excinfo.value)
    assert "incomplete" in message
    assert "ZN/B/1" in message


def test_a_missing_cofactor_residue_also_fails_the_entry(tmp_path):
    """The completeness demand covers exactly the rows Alchemy would emit."""
    builder = StructureBuilder()
    builder.add_hetero_residue(
        "XCF",
        7,
        [
            AtomSpec("C1", "C", (0.0, 0.0, 0.0)),
            AtomSpec("N1", "N", (1.4, 0.0, 0.0)),
        ],
        chain="B",
    )
    builder.add_water(101, (10.0, 0.0, 0.0), chain="B")
    path = builder.write_pdb(tmp_path / "apo.pdb")
    context = load_structure("test", path)
    rows_in = [
        row for row in helpers.edstats_rows_for_structure(context) if row[0] != "XCF"
    ]
    stats = _write_rows(tmp_path, rows_in)

    # Without the catalog the component is not expected and the file is fine.
    assert _extract(stats, context)[0] == []
    with pytest.raises(ValueError, match=r"XCF/B/7"):
        _extract(stats, context, cofactors={"XCF"})


@pytest.mark.parametrize(
    "content, fragment",
    [
        pytest.param("", "EDSTATS output is empty", id="empty-file"),
        pytest.param("\n   \n\n", "EDSTATS output is empty", id="whitespace-only"),
        pytest.param(
            " ".join(helpers.EDSTATS_HEADER) + "\n", "no residue rows", id="header-only"
        ),
    ],
)
def test_unusable_edstats_output_fails_the_entry(tmp_path, content, fragment):
    """Empty output is a failure, never an entry with zero metal sites."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats = tmp_path / "stats.out"
    stats.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=fragment):
        _extract(stats, context)


def test_blank_lines_between_rows_are_ignored(tmp_path):
    """Stray blank lines in the table are not rows and must not fail the parse."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    body = helpers.edstats_text(helpers.edstats_rows_for_structure(context))
    lines = body.splitlines()
    padded = "\n\n".join(["   "] + lines) + "\n"
    stats = tmp_path / "stats.out"
    stats.write_text(padded, encoding="utf-8")

    rows, _ = _extract(stats, context)
    assert [row["resname"] for row in rows] == ["ZN"]


def test_a_structure_is_required_for_identification(tmp_path):
    """Element-based identification cannot run against the table alone."""
    stats = _write_rows(tmp_path, [helpers.edstats_row("ZN", "B", "1")])
    with pytest.raises(TypeError, match="structure"):
        # Omitting the argument is the point of the test.
        extract_metal_statistics(  # type: ignore[call-arg]
            "test", str(stats), set(METAL_ELEMENTS), set()
        )


def test_parsing_leaves_no_files_behind(tmp_path):
    """The parser only reads; it writes nothing next to its input."""
    path = simple_metal_site().write_pdb(tmp_path / "site.pdb")
    context = load_structure("test", path)
    stats = helpers.write_edstats_for_structure(tmp_path / "stats.out", context)
    before = sorted(os.listdir(tmp_path))

    _extract(stats, context)

    assert sorted(os.listdir(tmp_path)) == before
