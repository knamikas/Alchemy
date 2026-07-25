# Metallocofactor catalog builder 6/12/2025
# Downloads the wwPDB Chemical Component Dictionary (CCD) and builds
# metallocofactors_id.txt: the list of CCD component ids that contain a metal
# (used by metal_identification.py alongside plain metal-ion matching).
#
# Normal pipeline runs should NOT need to run this file directly -- main.py
# calls refresh_cofactors_if_needed() at startup, which uses the committed
# metallocofactors_id.txt as-is unless it's missing or stale (per
# metallocofactors_id.meta.json), in which case it refreshes automatically
# when network access is available. Use --refresh-cofactors on main.py, or
# run this file directly with --force, to refresh explicitly.

import gzip
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.error import HTTPError
from urllib.request import urlopen

import gemmi

from metal_identification import metals as common_metals, uncommonMetals

URL_CCD = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _default_cache_dir():
    """Untracked, per-user location for runtime cofactor-list refreshes.

    Kept separate from DATA_DIR (the tracked repo copy) so a normal run never
    dirties `git status`, two processes never race on the same tracked file,
    and this works even if the repo checkout itself is read-only.
    """
    override = os.environ.get("ALCHEMY_CACHE_DIR")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "alchemy")


CACHE_DIR = _default_cache_dir()

# Use the same configured elements for plain-site analysis and CCD catalog
# generation so a symbol cannot be recognized in only one pipeline stage.
metals = common_metals + uncommonMetals


def find_metal_match(string):
    tokens = re.findall(r'[A-Z][a-z]?', string)  # split into element-symbol-like tokens
    return any(t.upper() in metals for t in tokens)


def download_ccd(tmp_dir):
    """Download and decompress the CCD into tmp_dir. Returns the .cif path."""
    print(f"Downloading CCD from {URL_CCD} ...", flush=True)
    gz_path = os.path.join(tmp_dir, "components.cif.gz")
    try:
        response = urlopen(URL_CCD, timeout=60)
    except HTTPError as e:
        raise RuntimeError(f"CCD download failed: HTTP {e.code}") from e

    with response:
        status = response.getcode()
        if status != 200:
            raise RuntimeError(f"CCD download failed: HTTP {status}")

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 10 * 1024 * 1024  # report every 10 MB
        with open(gz_path, "wb") as f:
            while chunk := response.read(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    mb = downloaded / (1024 * 1024)
                    if total_size:
                        pct = 100 * downloaded / total_size
                        print(f"  ...downloaded {mb:.0f} MB ({pct:.0f}%)", flush=True)
                    else:
                        print(f"  ...downloaded {mb:.0f} MB", flush=True)
                    next_report += 10 * 1024 * 1024
    print(f"Download complete ({downloaded / (1024 * 1024):.0f} MB). Decompressing...", flush=True)

    cif_path = os.path.join(tmp_dir, "components.cif")
    with gzip.open(gz_path, "rb") as fi, open(cif_path, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print("Decompression complete.", flush=True)
    return cif_path


def build_metallocofactors_list(
        cif_path: str,
        output_path: str,
        debug_dir: Optional[str] = None,
        ):
    """Parse a CCD components.cif file and write metallocofactors_id.txt.

    `debug_dir`, if given, also writes missing_formulas.txt and
    missingCIF.cif there (components with no recorded formula, checked
    instead by individual atom symbols) -- optional, for inspection only.

    Returns a dict of counts for logging.
    """
    print("Reading CCD")
    ccd = gemmi.cif.read_file(cif_path)

    missing_cif = gemmi.cif.Document() if debug_dir else None
    missing_formulas_path = (os.path.join(debug_dir, "missing_formulas.txt")
                              if debug_dir else None)
    if missing_formulas_path and os.path.exists(missing_formulas_path):
        os.remove(missing_formulas_path)

    counts = {"with_metal": 0, "skipped_ions": 0, "missing_formula": 0,
              "missing_formula_with_metal": 0}

    total = len(ccd)
    print(f"Analyzing {total} components in CCD")
    with open(output_path, "w") as f_write:
        for i, block in enumerate(ccd, 1):
            if i % 5000 == 0 or i == total:
                print(f"  ...{i}/{total} components checked", flush=True)
            has_metal = False
            comp_id = block.find_value('_chem_comp.id')
            formula = block.find_value('_chem_comp.formula')
            atom_symbols = [s.upper() for s in block.find_values('_chem_comp_atom.type_symbol')]
            is_single_metal_atom = len(atom_symbols) == 1 and atom_symbols[0] in metals

            
            if is_single_metal_atom:
                # exactly one atom, and it's a metal -- a plain ion (any charge
                # state/CCD code), not a cofactor. Handled separately by
                # element-based ion detection at analysis time.
                counts["skipped_ions"] += 1
            elif comp_id == 'UNL':
                pass  # generic "unknown ligand" marker, never include
            elif formula == '?':
                counts["missing_formula"] += 1
                if missing_cif is not None:
                    missing_cif.add_copied_block(block, pos=-1)
                if missing_formulas_path:
                    with open(missing_formulas_path, 'a') as mf:
                        mf.write(f"{comp_id}\n")
                for symbol in block.find_values('_chem_comp_atom.type_symbol'):
                    if find_metal_match(symbol):
                        has_metal = True
                        counts["missing_formula_with_metal"] += 1
                        break
            else:
                if find_metal_match(formula):
                    has_metal = True
                    counts["with_metal"] += 1

            if has_metal:
                f_write.write(f"{comp_id}\t{formula}\n")

    if missing_cif is not None and debug_dir is not None:
        missing_cif.write_file(os.path.join(debug_dir, "missingCIF.cif"))

    print(f"Found {counts['with_metal']} components with metal")
    print(f"Skipped {counts['skipped_ions']} metal ions")
    print(f"Checked {counts['missing_formula']} components missing formulas "
          f"and added {counts['missing_formula_with_metal']} that contained "
          f"metals to metallocofactors list")
    return counts


def _is_stale(meta_path, max_age_days):
    """True if metadata is missing, unreadable, or older than max_age_days."""
    if not os.path.exists(meta_path):
        return True
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        generated = datetime.fromisoformat(meta["generated"])
        if generated.tzinfo is None:
            return True
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return True
    return (datetime.now(timezone.utc) - generated) > timedelta(days=max_age_days)


def _catalog_is_fresh(data_dir, max_age_days):
    """Whether a catalog directory contains a usable, current file pair."""
    return (
        os.path.exists(os.path.join(data_dir, "metallocofactors_id.txt")) and
        not _is_stale(
            os.path.join(data_dir, "metallocofactors_id.meta.json"),
            max_age_days,
        )
    )


def _refresh_cofactors(data_dir):
    output_path = os.path.join(data_dir, "metallocofactors_id.txt")
    meta_path = os.path.join(data_dir, "metallocofactors_id.meta.json")
    # dir=data_dir keeps the temp files on the same filesystem as the target,
    # so the os.replace() calls below are guaranteed atomic (os.replace raises
    # "Invalid cross-device link" if src/dst are on different filesystems).
    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        cif_path = download_ccd(tmp)
        tmp_output_path = os.path.join(tmp, "metallocofactors_id.txt")
        counts = build_metallocofactors_list(cif_path, tmp_output_path, debug_dir=data_dir)

        meta = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "ccd_source": URL_CCD,
            "counts": counts,
        }
        tmp_meta_path = os.path.join(tmp, "metallocofactors_id.meta.json")
        with open(tmp_meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Both files generated successfully -- swap them into place atomically
        # so a failure above never leaves the committed list/metadata
        # truncated or partially written.
        os.replace(tmp_output_path, output_path)
        os.replace(tmp_meta_path, meta_path)
        # cif_path (and the rest of the tmp dir) is deleted automatically here
    return counts


def refresh_cofactors_if_needed(data_dir=None, cache_dir=None, max_age_days=30, force=False):

    """Ensure metallocofactors_id.txt is present and current.

    Runtime refreshes are written to `cache_dir` (an untracked, per-user
    location) -- never to `data_dir`, the tracked copy shipped in the repo.

    If the committed list exists and is fresh (per its metadata) and
    force=False, does nothing and uses it silently. If missing or stale,
    attempts an automatic refresh from the CCD. On network/download failure,
    falls back to the existing list (if any) with a warning, rather than
    failing the whole pipeline. Raises RuntimeError only if no usable list
    exists at all and the refresh also failed.
    """
    data_dir = data_dir or DATA_DIR
    cache_dir = cache_dir or CACHE_DIR

    cache_list_path = os.path.join(cache_dir, "metallocofactors_id.txt")

    if not force:
        if _catalog_is_fresh(cache_dir, max_age_days):
            return
        if _catalog_is_fresh(data_dir, max_age_days):
            return

    os.makedirs(cache_dir, exist_ok=True)
    try:
        _refresh_cofactors(cache_dir)
        print("Refreshed metallocofactors_id.txt in the local cache.", flush=True)
    except Exception as e:
        if force:
            # The caller explicitly asked for a refresh -- silently falling
            # back to a stale list would look like the refresh succeeded.
            raise RuntimeError(f"Forced cofactor refresh failed: {e}") from e
        if os.path.exists(cache_list_path):
            print(f"Warning: could not refresh cofactor list from CCD ({e}); "
                  f"using existing cached metallocofactors_id.txt.", flush=True)
        elif os.path.exists(os.path.join(data_dir, "metallocofactors_id.txt")):
            print(f"Warning: could not refresh cofactor list from CCD ({e}); "
                  f"using the committed metallocofactors_id.txt from {data_dir}.", flush=True)
        else:
            raise RuntimeError(
                f"No cofactor list available and refresh failed: {e}") from e

def active_cofactors_path(data_dir=None, cache_dir=None, max_age_days=30):
    """Return the best available cached or bundled cofactor catalog."""
    data_dir = data_dir or DATA_DIR
    cache_dir = cache_dir or CACHE_DIR
    cache_path = os.path.join(cache_dir, "metallocofactors_id.txt")
    data_path = os.path.join(data_dir, "metallocofactors_id.txt")

    if _catalog_is_fresh(cache_dir, max_age_days):
        return cache_path
    if _catalog_is_fresh(data_dir, max_age_days):
        return data_path
    if os.path.exists(cache_path):
        return cache_path
    return data_path

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Refresh metallocofactors_id.txt from the wwPDB CCD.")
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--max-age-days", type=int, default=30)
    p.add_argument("--force", action="store_true",
                   help="refresh even if the current list is not stale")
    args = p.parse_args()
    refresh_cofactors_if_needed(data_dir=args.data_dir,
                                max_age_days=args.max_age_days, force=args.force)
