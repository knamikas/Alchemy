# Alloy 6/12/2025
# Downloads the wwPDB Chemical Component Dictionary (CCD) and builds
# metallocofactors_id.txt: the list of CCD component ids that contain a metal
# (used by Analysisv2_kn.py alongside plain metal-ion matching).
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

import gemmi
import requests

URL_CCD = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(REPO_DIR, "src", "data")

# all metals to search for
metals = ['NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
          'CD', 'HG', 'PT', 'MO', 'AL', 'BE', 'BA', 'RU', 'V', 'SR',
          'CS', 'W', 'AU', 'YB', 'LI', 'GD', 'PB', 'U', 'Y', 'LR',
          'TI', 'RB', 'AG', 'SM', 'OS', 'PR', 'PD', 'EU', 'TB', 'RE',
          'RH', 'TA', 'LU', 'HO', 'CR', 'GA', 'LA', 'SN', 'SB', 'CE',
          'ZR', 'ER', 'TH', 'IN', 'HR', 'SC', 'DY', 'BI', 'PA', 'PU',
          'AM', 'CM', 'CF', 'GE', 'NB', 'TC', 'ND', 'PM', 'TM', 'PO',
          'FR', 'RA', 'AC', 'NP', 'BK', 'ES', 'FM', 'MD', 'NO', 'LR',
          'RF', 'DB', 'SG']


def find_metal_match(string):
    tokens = re.findall(r'[A-Z][a-z]?', string)  # split into element-symbol-like tokens
    return any(t.upper() in metals for t in tokens)


def download_ccd(tmp_dir):
    """Download and decompress the CCD into tmp_dir. Returns the .cif path."""
    print(f"Downloading CCD from {URL_CCD} ...", flush=True)
    gz_path = os.path.join(tmp_dir, "components.cif.gz")
    response = requests.get(URL_CCD, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"CCD download failed: HTTP {response.status_code}")

    total_size = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    next_report = 10 * 1024 * 1024  # report every 10 MB
    with open(gz_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
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


def build_metallocofactors_list(cif_path, output_path, debug_dir=None):
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

    with open(output_path, "w") as f_write:
        for block in ccd:
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

    if missing_cif is not None:
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
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return True
    return (datetime.now(timezone.utc) - generated) > timedelta(days=max_age_days)


def _refresh_cofactors(data_dir):
    output_path = os.path.join(data_dir, "metallocofactors_id.txt")
    meta_path = os.path.join(data_dir, "metallocofactors_id.meta.json")
    with tempfile.TemporaryDirectory() as tmp:
        cif_path = download_ccd(tmp)
        counts = build_metallocofactors_list(cif_path, output_path, debug_dir=data_dir)
        # cif_path (and the whole tmp dir) is deleted automatically here
    meta = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "ccd_source": URL_CCD,
        "counts": counts,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return counts


def refresh_cofactors_if_needed(data_dir=None, max_age_days=30, force=False):
    """Ensure metallocofactors_id.txt is present and current.

    If the committed list exists and is fresh (per its metadata) and
    force=False, does nothing and uses it silently. If missing or stale,
    attempts an automatic refresh from the CCD. On network/download failure,
    falls back to the existing list (if any) with a warning, rather than
    failing the whole pipeline. Raises RuntimeError only if no usable list
    exists at all and the refresh also failed.
    """
    data_dir = data_dir or DATA_DIR
    list_path = os.path.join(data_dir, "metallocofactors_id.txt")
    meta_path = os.path.join(data_dir, "metallocofactors_id.meta.json")

    if not force and os.path.exists(list_path) and not _is_stale(meta_path, max_age_days):
        return

    try:
        _refresh_cofactors(data_dir)
        print("Refreshed metallocofactors_id.txt from the CCD.", flush=True)
    except Exception as e:
        if os.path.exists(list_path):
            print(f"Warning: could not refresh cofactor list from CCD ({e}); "
                  f"using existing metallocofactors_id.txt.", flush=True)
        else:
            raise RuntimeError(
                f"No cofactor list available and refresh failed: {e}") from e


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