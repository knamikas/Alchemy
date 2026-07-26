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
# Schema 2 adds the structural-class column; schema 1 was id and formula only.
CATALOG_SCHEMA_VERSION = 2
BUILDER_VERSION = 2
COMPONENT_ID_PATTERN = re.compile(r"[A-Z0-9]+")
ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")

# --------------------------------------------------------------------------- #
# Structural classes
# --------------------------------------------------------------------------- #
# Alchemy tags each metal's environment in the parent_type column. Two
# architectures are singled out because their coordination geometry is
# chemically constrained, which is what makes them usable as reference
# populations. Deriving them here keeps them tracking the CCD instead of
# drifting behind a hand-maintained list in the analysis code.
CLASS_CLUSTER = "cluster"
CLASS_HEME = "heme"

# Metals that form polynuclear metal-sulfide clusters in biological cofactors.
CLUSTER_METALS = ("FE", "NI", "MO", "V", "W")
# Porphine (CCD "POR"), the unsubstituted macrocycle, is C20. Nothing smaller
# can carry a tetrapyrrole ring.
PORPHYRIN_MIN_CARBON = 20

# Canonical cofactors named in the Alchemy manuscript Methods. Asserted present
# after every build so a change to the rules above cannot silently drop a
# member of the published reference populations.
CANONICAL_CLASSES = {
    "HEM": CLASS_HEME, "HEC": CLASS_HEME, "HEA": CLASS_HEME,
    "HEB": CLASS_HEME, "DHE": CLASS_HEME, "HEO": CLASS_HEME,
    "HEV": CLASS_HEME,
    "SF4": CLASS_CLUSTER, "FES": CLASS_CLUSTER, "F3S": CLASS_CLUSTER,
    "SF3": CLASS_CLUSTER, "F4S": CLASS_CLUSTER, "NFS": CLASS_CLUSTER,
    "FSX": CLASS_CLUSTER, "FS5": CLASS_CLUSTER,
}


def element_counts(formula):
    """Return ``{ELEMENT: count}`` parsed from a CCD formula string."""
    counts = {}
    for symbol, number in ELEMENT_PATTERN.findall(formula or ""):
        if symbol:
            key = symbol.upper()
            counts[key] = counts.get(key, 0) + int(number or 1)
    return counts


def classify_component(formula):
    """Return ``"cluster"``, ``"heme"``, or ``""`` for one CCD formula.

    A cluster is polynuclear metal-sulfide: two or more cluster-forming metals
    together with bridging sulfur. The metals are counted collectively rather
    than as iron alone because NiFe hydrogenase active sites (CCD ``NFE``,
    ``FNE``) carry only one Fe, and nitrogenase FeMo/FeV cofactors mix metals.

    A heme is a single Fe held in a tetrapyrrole macrocycle, approximated by
    one Fe with at least four nitrogens and a porphine-sized carbon skeleton.

    Cluster is tested first, matching the precedence the analysis has always
    applied to components that satisfy both.
    """
    counts = element_counts(formula)
    if (sum(counts.get(metal, 0) for metal in CLUSTER_METALS) >= 2 and
            counts.get("S", 0) >= 1):
        return CLASS_CLUSTER
    if (counts.get("FE", 0) == 1 and counts.get("N", 0) >= 4 and
            counts.get("C", 0) >= PORPHYRIN_MIN_CARBON):
        return CLASS_HEME
    return ""


def _verify_canonical_classes(classes):
    """Fail the build if a manuscript-named reference cofactor was lost."""
    wrong = {component_id: (expected, classes.get(component_id, "<absent>"))
             for component_id, expected in CANONICAL_CLASSES.items()
             if classes.get(component_id) != expected}
    if wrong:
        details = ", ".join(
            f"{component_id} expected {expected}, got {actual}"
            for component_id, (expected, actual) in sorted(wrong.items()))
        raise ValueError(
            "canonical cofactor classification changed: " + details)


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
    return any(element in METAL_ELEMENTS for element in element_counts(formula))


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
            records[component_id] = (formula or "?", classify_component(formula))

    classes = {component_id: structural_class
               for component_id, (_, structural_class) in records.items()
               if structural_class}
    _verify_canonical_classes(classes)
    counts["cluster_entries"] = sum(
        1 for value in classes.values() if value == CLASS_CLUSTER)
    counts["heme_entries"] = sum(
        1 for value in classes.values() if value == CLASS_HEME)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        for component_id in sorted(records):
            formula, structural_class = records[component_id]
            handle.write(f"{component_id}\t{formula}\t{structural_class}\n")

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
    print(f"Classified {counts['cluster_entries']} clusters and "
          f"{counts['heme_entries']} hemes", flush=True)
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


def reclassify_catalog(output_dir):
    """Re-derive structural classes from the bundled catalog's own formulas.

    The classes depend only on formulas already in the catalog, so a change to
    the rules can be applied without downloading the CCD again. The recorded
    CCD provenance is untouched because the component data is unchanged.
    """
    output_dir = os.path.abspath(output_dir)
    catalog_path = os.path.join(output_dir, CATALOG_FILENAME)
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)

    records = {}
    with open(catalog_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            component_id = fields[0].strip()
            formula = fields[1] if len(fields) > 1 else "?"
            records[component_id] = (formula, classify_component(formula))

    classes = {component_id: structural_class
               for component_id, (_, structural_class) in records.items()
               if structural_class}
    _verify_canonical_classes(classes)

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        temporary_catalog = os.path.join(temp_dir, CATALOG_FILENAME)
        with open(temporary_catalog, "w", encoding="utf-8",
                  newline="") as handle:
            for component_id in sorted(records):
                formula, structural_class = records[component_id]
                handle.write(f"{component_id}\t{formula}\t{structural_class}\n")
        catalog_hash = _sha256(temporary_catalog)

        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata["catalog_schema_version"] = CATALOG_SCHEMA_VERSION
        metadata["builder_version"] = BUILDER_VERSION
        metadata["reclassified"] = datetime.now(timezone.utc).isoformat()
        metadata["catalog_sha256"] = catalog_hash
        metadata.setdefault("counts", {})
        metadata["counts"]["cluster_entries"] = sum(
            1 for value in classes.values() if value == CLASS_CLUSTER)
        metadata["counts"]["heme_entries"] = sum(
            1 for value in classes.values() if value == CLASS_HEME)

        temporary_metadata = os.path.join(temp_dir, METADATA_FILENAME)
        with open(temporary_metadata, "w", encoding="utf-8",
                  newline="") as handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")

        os.replace(temporary_catalog, catalog_path)
        os.replace(temporary_metadata, metadata_path)

    print(f"Reclassified {len(records)} entries in {catalog_path}", flush=True)
    print(f"  clusters: {metadata['counts']['cluster_entries']}, "
          f"hemes: {metadata['counts']['heme_entries']}", flush=True)
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
    counts = metadata.get("counts", {})
    print(f"Structural classes: {counts.get('cluster_entries', 'unknown')} "
          f"cluster, {counts.get('heme_entries', 'unknown')} heme")
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
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help=("re-derive structural classes from the bundled catalog's "
              "formulas without downloading the CCD"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.status:
        report_status(os.path.abspath(args.output_dir))
    elif args.reclassify:
        reclassify_catalog(args.output_dir)
    else:
        rebuild_catalog(
            args.output_dir,
            ccd_path=args.ccd,
            debug_dir=args.debug_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
