#!/usr/bin/env python3
"""Explicitly rebuild Alchemy's bundled metallocofactor catalog.

This is a developer maintenance utility, not part of the analysis pipeline.
Normal Alchemy runs never import it, check catalog age, or access the network.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPOSITORY_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from metal_elements import METAL_ELEMENTS  # noqa: E402


CCD_URL = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"
DEFAULT_OUTPUT_DIR = SOURCE_DIR / "data"
CATALOG_FILENAME = "metallocofactors_id.txt"
METADATA_FILENAME = "metallocofactors_id.meta.json"
CATALOG_SCHEMA_VERSION = 1
BUILDER_VERSION = 1
COMPONENT_ID_PATTERN = re.compile(r"[A-Z0-9]+")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_metal_match(formula):
    """Return whether a CCD formula contains a configured metal element."""
    if not formula:
        return False
    tokens = re.findall(r"[A-Z][a-z]?", formula)
    return any(token.upper() in METAL_ELEMENTS for token in tokens)


def symbol_is_metal(symbol):
    """Return whether one complete atom type symbol is a configured metal."""
    return bool(symbol) and symbol.strip().upper() in METAL_ELEMENTS


def _decompress_ccd(source_path: str, destination_path: str) -> None:
    with gzip.open(source_path, "rb") as source:
        with open(destination_path, "wb") as destination:
            shutil.copyfileobj(source, destination)


def download_ccd(temp_dir):
    """Download and decompress the CCD, returning its path and provenance."""
    print(f"Downloading CCD from {CCD_URL} ...", flush=True)
    compressed_path = os.path.join(temp_dir, "components.cif.gz")
    try:
        response = urlopen(CCD_URL, timeout=60)
    except HTTPError as exc:
        raise RuntimeError(
            f"CCD download failed with HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"CCD download failed: {exc.reason}") from exc

    with response:
        status = response.getcode()
        if status != 200:
            raise RuntimeError(f"CCD download failed with HTTP {status}")
        response_metadata = {
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
                            f"  ...downloaded {amount:.0f} MB "
                            f"({percent:.0f}%)",
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


def prepare_local_ccd(source_path, temp_dir):
    """Return an uncompressed local CCD path and source provenance."""
    source_path = os.path.abspath(source_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"CCD input not found: {source_path}")

    provenance = {"source": source_path}
    if source_path.lower().endswith(".gz"):
        cif_path = os.path.join(temp_dir, "components.cif")
        _decompress_ccd(source_path, cif_path)
        provenance["compressed_sha256"] = _sha256(source_path)
        return cif_path, provenance
    return source_path, provenance


def build_metallocofactors_list(cif_path, output_path, debug_dir=None):
    """Build a deterministic catalog from a wwPDB CCD components file."""
    import gemmi

    print(f"Reading CCD from {cif_path}", flush=True)
    ccd = gemmi.cif.read_file(cif_path)
    missing_components = gemmi.cif.Document() if debug_dir else None
    missing_component_ids = []
    records = {}
    counts = {
        "with_metal": 0,
        "skipped_ions": 0,
        "missing_formula": 0,
        "missing_formula_with_metal": 0,
    }

    total = len(ccd)
    print(f"Analyzing {total} CCD components", flush=True)
    for index, block in enumerate(ccd, 1):
        if index % 5000 == 0 or index == total:
            print(f"  ...{index}/{total} components checked", flush=True)

        component_id = block.find_value("_chem_comp.id")
        if not component_id or component_id in (".", "?"):
            raise ValueError(
                f"CCD block {block.name!r} has no usable component ID")
        if not COMPONENT_ID_PATTERN.fullmatch(component_id):
            raise ValueError(
                f"CCD block {block.name!r} has invalid component ID "
                f"{component_id!r}")
        formula = block.find_value("_chem_comp.formula") or ""
        atom_symbols = [
            symbol.strip().upper()
            for symbol in block.find_values("_chem_comp_atom.type_symbol")
        ]
        single_metal_atom = (
            len(atom_symbols) == 1 and atom_symbols[0] in METAL_ELEMENTS
        )

        has_metal = False
        if single_metal_atom:
            counts["skipped_ions"] += 1
        elif component_id == "UNL":
            pass
        elif formula in ("", "?"):
            counts["missing_formula"] += 1
            missing_component_ids.append(component_id)
            if missing_components is not None:
                missing_components.add_copied_block(block, pos=-1)
            if any(symbol_is_metal(symbol) for symbol in atom_symbols):
                has_metal = True
                counts["missing_formula_with_metal"] += 1
        elif find_metal_match(formula):
            has_metal = True
            counts["with_metal"] += 1

        if has_metal:
            if component_id in records:
                raise ValueError(
                    f"duplicate CCD component ID: {component_id}")
            records[component_id] = formula or "?"

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        for component_id in sorted(records):
            handle.write(f"{component_id}\t{records[component_id]}\n")

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        missing_ids_path = os.path.join(debug_dir, "missing_formulas.txt")
        with open(
                missing_ids_path, "w", encoding="utf-8", newline=""
                ) as handle:
            for component_id in missing_component_ids:
                handle.write(f"{component_id}\n")
        if missing_components is not None:
            missing_components.write_file(
                os.path.join(debug_dir, "missingCIF.cif"))

    counts["catalog_entries"] = len(records)
    print(f"Catalog contains {len(records)} metallocofactors", flush=True)
    print(f"Skipped {counts['skipped_ions']} single-metal ions", flush=True)
    return counts


def rebuild_catalog(output_dir, ccd_path=None, debug_dir=None):
    """Explicitly rebuild the bundled catalog and provenance metadata."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    catalog_path = os.path.join(output_dir, CATALOG_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        if ccd_path:
            prepared_ccd, ccd_provenance = prepare_local_ccd(
                ccd_path, temp_dir)
        else:
            prepared_ccd, ccd_provenance = download_ccd(temp_dir)

        temporary_catalog = os.path.join(temp_dir, CATALOG_FILENAME)
        counts = build_metallocofactors_list(
            prepared_ccd, temporary_catalog, debug_dir=debug_dir)
        catalog_hash = _sha256(temporary_catalog)
        metadata = {
            "catalog_schema_version": CATALOG_SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "generated": datetime.now(timezone.utc).isoformat(),
            "ccd_source": ccd_provenance["source"],
            "ccd_sha256": _sha256(prepared_ccd),
            "ccd_compressed_sha256": ccd_provenance.get(
                "compressed_sha256"),
            "ccd_etag": ccd_provenance.get("etag"),
            "ccd_last_modified": ccd_provenance.get("last_modified"),
            "catalog_sha256": catalog_hash,
            "counts": counts,
        }
        temporary_metadata = os.path.join(temp_dir, METADATA_FILENAME)
        with open(
                temporary_metadata, "w", encoding="utf-8", newline=""
                ) as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        os.replace(temporary_catalog, catalog_path)
        os.replace(temporary_metadata, metadata_path)

    print(f"Updated {catalog_path}", flush=True)
    print(f"Catalog SHA-256: {catalog_hash}", flush=True)
    return metadata


def report_status(output_dir):
    """Print the committed catalog's recorded version and integrity status."""
    catalog_path = os.path.join(output_dir, CATALOG_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    with open(metadata_path, encoding="utf-8", errors="strict") as handle:
        metadata = json.load(handle)
    actual_hash = _sha256(catalog_path)
    recorded_hash = metadata.get("catalog_sha256")

    print(f"Generated: {metadata.get('generated', 'unknown')}")
    print(
        "Catalog schema version: "
        f"{metadata.get('catalog_schema_version', 'unknown')}")
    print(f"Builder version: {metadata.get('builder_version', 'unknown')}")
    entry_count = metadata.get("counts", {}).get(
        "catalog_entries", "unknown")
    print(f"Entries: {entry_count}")
    print(f"Catalog SHA-256: {actual_hash}")
    if recorded_hash is None:
        print("Recorded hash: unavailable in legacy metadata")
    elif recorded_hash != actual_hash:
        raise RuntimeError(
            "catalog checksum does not match its metadata: "
            f"recorded {recorded_hash}, actual {actual_hash}")
    else:
        print("Integrity: verified")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly rebuild Alchemy's bundled metallocofactor catalog."))
    parser.add_argument(
        "--ccd",
        help=(
            "local components.cif or components.cif.gz; downloads the current "
            "wwPDB CCD when omitted"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory receiving the catalog and metadata",
    )
    parser.add_argument(
        "--debug-dir",
        help="optional directory for missing-formula diagnostic files",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="report bundled catalog metadata without rebuilding or networking",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        report_status(os.path.abspath(args.output_dir))
    else:
        rebuild_catalog(
            args.output_dir,
            ccd_path=args.ccd,
            debug_dir=args.debug_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
