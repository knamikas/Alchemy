#!/usr/bin/env python3
"""Rebuild Alchemy's bundled metallocofactor catalog.

A developer utility: the analysis pipeline never imports it and never networks.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

if TYPE_CHECKING:
    import gemmi

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from metal_elements import METAL_ELEMENTS  # noqa: E402


CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
DEFAULT_OUTPUT_DIR = SOURCE_DIR / "data"
CATALOG_FILENAME = "metallocofactors_id.txt"
METADATA_FILENAME = "metallocofactors_id.meta.json"
COMPONENT_ID_PATTERN = re.compile(r"[A-Z0-9]+")
ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")

#: Published ``parent_type`` values for the two reference populations.
CLASS_CLUSTER = "cluster"
CLASS_HEME = "heme"

# Metals that form polynuclear metal-chalcogen clusters in biological cofactors.
CLUSTER_METALS = ("FE", "NI", "MO", "V", "W")
CLUSTER_CHALCOGENS = ("S", "SE")
# The porphyrinoid core is one biconnected conjugated system. Canonical
# porphyrins have 20 carbon and four nitrogen atoms in a 24-atom core;
# oxaporphyrins replace one carbon, while iron phthalocyanine is larger.
# Three Fe-N bonds admit covalently modified porphyrins such as CCD ``WRK``.
PORPHYRINOID_CORE_MIN_ATOMS = 24
PORPHYRINOID_CORE_MIN_CARBON = 18
PORPHYRINOID_CORE_MIN_NITROGEN = 4
PORPHYRINOID_MIN_FE_N_BONDS = 3

# Canonical cofactors named in the Alchemy manuscript Methods.
CANONICAL_CLASSES = {
    "HEM": CLASS_HEME,
    "HEC": CLASS_HEME,
    "HEA": CLASS_HEME,
    "HEB": CLASS_HEME,
    "DHE": CLASS_HEME,
    "HEO": CLASS_HEME,
    "HEV": CLASS_HEME,
    "SF4": CLASS_CLUSTER,
    "FES": CLASS_CLUSTER,
    "F3S": CLASS_CLUSTER,
    "SF3": CLASS_CLUSTER,
    "F4S": CLASS_CLUSTER,
    "NFS": CLASS_CLUSTER,
    "FSX": CLASS_CLUSTER,
    "FS5": CLASS_CLUSTER,
}


def element_counts(formula: str | None) -> dict[str, int]:
    """Return ``{ELEMENT: count}`` parsed from a CCD formula string."""
    counts: dict[str, int] = {}
    for symbol, number in ELEMENT_PATTERN.findall(formula or ""):
        if symbol:
            key = symbol.upper()
            counts[key] = counts.get(key, 0) + int(number or 1)
    return counts


def _component_graph(
    block: gemmi.cif.Block,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Return CCD atom elements and undirected bond adjacency for ``block``."""
    elements = {
        atom_id: symbol.strip().upper()
        for atom_id, symbol in block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
            ]
        )
    }
    adjacency: dict[str, set[str]] = {atom_id: set() for atom_id in elements}
    for atom_1, atom_2 in block.find(
        [
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
        ]
    ):
        if atom_1 not in elements or atom_2 not in elements:
            continue
        adjacency[atom_1].add(atom_2)
        adjacency[atom_2].add(atom_1)
    return elements, adjacency


def _biconnected_components(
    adjacency: Mapping[str, set[str]], allowed_atoms: Iterable[str]
) -> list[set[str]]:
    """Return vertex sets for biconnected components of an atom subgraph."""
    allowed = set(allowed_atoms)
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str] = {}
    edge_stack: list[tuple[str, str]] = []
    components: list[set[str]] = []
    clock = 0

    def visit(atom: str) -> None:
        nonlocal clock
        clock += 1
        discovery[atom] = low[atom] = clock
        for neighbor in sorted(adjacency.get(atom, ())):
            if neighbor not in allowed:
                continue
            if neighbor not in discovery:
                parent[neighbor] = atom
                edge_stack.append((atom, neighbor))
                visit(neighbor)
                low[atom] = min(low[atom], low[neighbor])
                if low[neighbor] >= discovery[atom]:
                    component: set[str] = set()
                    while edge_stack:
                        edge = edge_stack.pop()
                        component.update(edge)
                        if edge == (atom, neighbor):
                            break
                    if component:
                        components.append(component)
            elif parent.get(atom) != neighbor and discovery[neighbor] < discovery[atom]:
                low[atom] = min(low[atom], discovery[neighbor])
                edge_stack.append((atom, neighbor))

    for atom in sorted(allowed):
        if atom not in discovery:
            visit(atom)
    return components


def classify_component(block: gemmi.cif.Block) -> str:
    """Return ``"cluster"``, ``"heme"``, or ``""`` from CCD connectivity.

    Formula stoichiometry cannot establish either architecture: sulfur may be
    part of an unrelated substituent, and many siderophores and synthetic iron
    chelates have Fe/N/C counts resembling a heme. Cluster is tested first,
    matching the precedence used by the analysis.
    """
    elements, adjacency = _component_graph(block)
    cluster_metals = {
        atom_id for atom_id, element in elements.items() if element in CLUSTER_METALS
    }
    for atom_id, element in elements.items():
        if element not in CLUSTER_CHALCOGENS:
            continue
        bonded_metals = cluster_metals.intersection(adjacency.get(atom_id, ()))
        if len(bonded_metals) >= 2:
            return CLASS_CLUSTER

    iron_atoms = [atom_id for atom_id, element in elements.items() if element == "FE"]
    if len(iron_atoms) != 1:
        return ""
    iron = iron_atoms[0]
    nitrogen_donors = {
        atom_id for atom_id in adjacency.get(iron, ()) if elements.get(atom_id) == "N"
    }
    if len(nitrogen_donors) < PORPHYRINOID_MIN_FE_N_BONDS:
        return ""

    ligand_heavy_atoms = {
        atom_id
        for atom_id, element in elements.items()
        if element != "H" and element not in METAL_ELEMENTS
    }
    for component in _biconnected_components(adjacency, ligand_heavy_atoms):
        carbon = sum(elements[atom_id] == "C" for atom_id in component)
        nitrogen = sum(elements[atom_id] == "N" for atom_id in component)
        donors = len(component.intersection(nitrogen_donors))
        if (
            len(component) >= PORPHYRINOID_CORE_MIN_ATOMS
            and carbon >= PORPHYRINOID_CORE_MIN_CARBON
            and nitrogen >= PORPHYRINOID_CORE_MIN_NITROGEN
            and donors >= PORPHYRINOID_MIN_FE_N_BONDS
        ):
            return CLASS_HEME
    return ""


def _verify_canonical_classes(classes: Mapping[str, str]) -> None:
    """Fail the build if a manuscript-named reference cofactor was lost."""
    wrong = {
        component_id: (expected, classes.get(component_id, "<absent>"))
        for component_id, expected in CANONICAL_CLASSES.items()
        if classes.get(component_id) != expected
    }
    if wrong:
        details = ", ".join(
            f"{component_id} expected {expected}, got {actual}"
            for component_id, (expected, actual) in sorted(wrong.items())
        )
        raise ValueError("canonical cofactor classification changed: " + details)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def formula_has_metal(formula: str) -> bool:
    return any(element in METAL_ELEMENTS for element in element_counts(formula))


def symbol_is_metal(symbol: str) -> bool:
    """Return whether one complete atom type symbol is a configured metal."""
    return bool(symbol) and symbol.strip().upper() in METAL_ELEMENTS


def _decompress_ccd(source_path: str, destination_path: str) -> None:
    with gzip.open(source_path, "rb") as source:
        with open(destination_path, "wb") as destination:
            shutil.copyfileobj(source, destination)


def download_ccd(temp_dir: str) -> tuple[str, dict[str, str | None]]:
    """Download and decompress the CCD, returning its path and provenance."""
    print(f"Downloading CCD from {CCD_URL} ...", flush=True)
    compressed_path = os.path.join(temp_dir, "components.cif.gz")
    try:
        response = urlopen(CCD_URL, timeout=60)
    except HTTPError as exc:
        raise RuntimeError(f"CCD download failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"CCD download failed: {exc.reason}") from exc

    with response:
        status = response.getcode()
        if status != 200:
            raise RuntimeError(f"CCD download failed with HTTP {status}")
        response_metadata: dict[str, str | None] = {
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
        }
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 10 * 1024 * 1024
        with open(compressed_path, "wb") as handle:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    amount = downloaded / (1024 * 1024)
                    if total_size:
                        percent = 100 * downloaded / total_size
                        print(
                            f"  ...downloaded {amount:.0f} MB ({percent:.0f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"  ...downloaded {amount:.0f} MB",
                            flush=True,
                        )
                    next_report += 10 * 1024 * 1024

    print("Download complete. Decompressing CCD ...", flush=True)
    cif_path = os.path.join(temp_dir, "components.cif")
    _decompress_ccd(compressed_path, cif_path)
    return cif_path, {
        "source": CCD_URL,
        "compressed_sha256": _sha256(compressed_path),
        **response_metadata,
    }


def prepare_local_ccd(
    source_path: str, temp_dir: str
) -> tuple[str, dict[str, str | None]]:
    """Return an uncompressed local CCD path and source provenance."""
    source_path = os.path.abspath(source_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"CCD input not found: {source_path}")

    provenance: dict[str, str | None] = {"source": source_path}
    if source_path.lower().endswith(".gz"):
        cif_path = os.path.join(temp_dir, "components.cif")
        _decompress_ccd(source_path, cif_path)
        provenance["compressed_sha256"] = _sha256(source_path)
        return cif_path, provenance
    return source_path, provenance


def build_metallocofactors_list(cif_path: str, output_path: str) -> dict[str, int]:
    """Build a deterministic catalog from a wwPDB CCD components file."""
    import gemmi

    print(f"Reading CCD from {cif_path}", flush=True)
    ccd = gemmi.cif.read_file(cif_path)
    records: dict[str, tuple[str, str]] = {}
    counts = {
        "with_metal": 0,
        "skipped_ions": 0,
        "missing_formula": 0,
        "missing_formula_with_metal": 0,
        "cluster_entries": 0,
        "heme_entries": 0,
    }

    total = len(ccd)
    print(f"Analyzing {total} CCD components", flush=True)
    for index, block in enumerate(ccd, 1):
        if index % 5000 == 0 or index == total:
            print(f"  ...{index}/{total} components checked", flush=True)

        component_id = block.find_value("_chem_comp.id")
        if not component_id or component_id in (".", "?"):
            raise ValueError(f"CCD block {block.name!r} has no usable component ID")
        if not COMPONENT_ID_PATTERN.fullmatch(component_id):
            raise ValueError(
                f"CCD block {block.name!r} has invalid component ID {component_id!r}"
            )
        formula = block.find_value("_chem_comp.formula") or ""
        atom_symbols = [
            symbol.strip().upper()
            for symbol in block.find_values("_chem_comp_atom.type_symbol")
        ]
        single_metal_atom = len(atom_symbols) == 1 and atom_symbols[0] in METAL_ELEMENTS

        has_metal = False
        if single_metal_atom:
            counts["skipped_ions"] += 1
        elif component_id == "UNL":
            pass
        elif formula in ("", "?"):
            counts["missing_formula"] += 1
            if any(symbol_is_metal(symbol) for symbol in atom_symbols):
                has_metal = True
                counts["missing_formula_with_metal"] += 1
        elif formula_has_metal(formula):
            has_metal = True
            counts["with_metal"] += 1

        if has_metal:
            if component_id in records:
                raise ValueError(f"duplicate CCD component ID: {component_id}")
            records[component_id] = (formula or "?", classify_component(block))

    classes = {
        component_id: structural_class
        for component_id, (_, structural_class) in records.items()
        if structural_class
    }
    _verify_canonical_classes(classes)
    counts["cluster_entries"] = sum(
        1 for value in classes.values() if value == CLASS_CLUSTER
    )
    counts["heme_entries"] = sum(1 for value in classes.values() if value == CLASS_HEME)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        for component_id in sorted(records):
            formula, structural_class = records[component_id]
            handle.write(f"{component_id}\t{formula}\t{structural_class}\n")

    counts["catalog_entries"] = len(records)
    print(f"Catalog contains {len(records)} metallocofactors", flush=True)
    print(f"Skipped {counts['skipped_ions']} single-metal ions", flush=True)
    print(
        f"Classified {counts['cluster_entries']} clusters and "
        f"{counts['heme_entries']} hemes",
        flush=True,
    )
    return counts


def rebuild_catalog(output_dir: str, ccd_path: str | None = None) -> dict[str, object]:
    """Rebuild the bundled catalog and its provenance metadata."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    catalog_path = os.path.join(output_dir, CATALOG_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        if ccd_path:
            prepared_ccd, ccd_provenance = prepare_local_ccd(ccd_path, temp_dir)
        else:
            prepared_ccd, ccd_provenance = download_ccd(temp_dir)

        temporary_catalog = os.path.join(temp_dir, CATALOG_FILENAME)
        counts = build_metallocofactors_list(prepared_ccd, temporary_catalog)
        catalog_hash = _sha256(temporary_catalog)
        metadata: dict[str, object] = {
            "generated": datetime.now(UTC).isoformat(),
            "ccd_source": ccd_provenance["source"],
            "ccd_sha256": _sha256(prepared_ccd),
            "ccd_compressed_sha256": ccd_provenance.get("compressed_sha256"),
            "ccd_etag": ccd_provenance.get("etag"),
            "ccd_last_modified": ccd_provenance.get("last_modified"),
            "catalog_sha256": catalog_hash,
            "counts": counts,
        }
        temporary_metadata = os.path.join(temp_dir, METADATA_FILENAME)
        with open(temporary_metadata, "w", encoding="utf-8", newline="") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        os.replace(temporary_catalog, catalog_path)
        os.replace(temporary_metadata, metadata_path)

    print(f"Updated {catalog_path}", flush=True)
    print(f"Catalog SHA-256: {catalog_hash}", flush=True)
    return metadata


def report_status(output_dir: str) -> dict[str, Any]:
    """Print the committed catalog's provenance and integrity status."""
    catalog_path = os.path.join(output_dir, CATALOG_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    with open(metadata_path, encoding="utf-8", errors="strict") as handle:
        metadata: dict[str, Any] = json.load(handle)
    actual_hash = _sha256(catalog_path)
    recorded_hash = metadata.get("catalog_sha256")

    print(f"Generated: {metadata.get('generated', 'unknown')}")
    entry_count = metadata.get("counts", {}).get("catalog_entries", "unknown")
    print(f"Entries: {entry_count}")
    counts = metadata.get("counts", {})
    print(
        f"Structural classes: {counts.get('cluster_entries', 'unknown')} "
        f"cluster, {counts.get('heme_entries', 'unknown')} heme"
    )
    print(f"Catalog SHA-256: {actual_hash}")
    if recorded_hash is None:
        print("Recorded hash: unavailable in legacy metadata")
    elif recorded_hash != actual_hash:
        raise RuntimeError(
            "catalog checksum does not match its metadata: "
            f"recorded {recorded_hash}, actual {actual_hash}"
        )
    else:
        print("Integrity: verified")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Explicitly rebuild Alchemy's bundled metallocofactor catalog.")
    )
    parser.add_argument(
        "--ccd",
        help=(
            "local components.cif or components.cif.gz; downloads the current "
            "wwPDB CCD when omitted"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory receiving the catalog and metadata",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="report bundled catalog metadata without rebuilding or networking",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        report_status(os.path.abspath(args.output_dir))
    else:
        rebuild_catalog(
            args.output_dir,
            ccd_path=args.ccd,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
