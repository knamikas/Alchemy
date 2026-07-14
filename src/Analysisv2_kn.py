# Analysis v2: extract metal / metallocofactor real-space stats from edstats output.
#
# edstats `stats.out` is a whitespace table: the first non-empty line is the column
# header, and each data line begins with a residue/component name. Column layout
# (0-indexed): 0 = residue name (RT), 1 = chain (CI), 2 = residue number (RN);
# field 12 is the RSZD magnitude used historically for ranking.
#
# `run_analysis` returns structured rows for metal ions and metal-containing
# cofactors; `main.py` aggregates these across many structures. A standalone
# __main__ reproduces the original per-structure file dump for manual use.
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# common metals to search for
metals = ['NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN']

# uncommon metals to search for
uncommonMetals = ['CD', 'HG', 'PT', 'MO', 'AL', 'BE', 'BA', 'RU', 'V', 'SR', 'CS',
                  'W', 'AU', 'YB', 'LI', 'GD', 'PB', 'U', 'Y', 'LR', 'TI', 'RB',
                  'AG', 'SM', 'OS', 'PR', 'PD', 'EU', 'TB', 'RE', 'RH', 'TA', 'LU',
                  'HO', 'CR', 'GA', 'LA', 'SN', 'SB', 'CE', 'ZR', 'ER', 'TH', 'IN',
                  'HR', 'SC', 'DY', 'BI', 'PA', 'PU', 'AM', 'CM', 'CF', 'GE', 'NB',
                  'TC', 'ND', 'PM', 'TM', 'PO', 'FR', 'RA', 'AC', 'NP', 'BK', 'ES',
                  'FM', 'MD', 'NO', 'LR', 'RF', 'DB', 'SG']


def load_cofactors(path=None):
    """Return a set of metal-containing CCD ids from metallocofactors_id.txt.

    The file is "{ccd_id}\\t{formula}" per line (see Alloy_kn.py). We keep only
    the id token.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "metallocofactors_id.txt")
    cofactor_set = set()
    with open(path) as f:
        for line in f:
            cid = line.split("\t")[0].strip()
            if cid:
                cofactor_set.add(cid)
    return cofactor_set

def _build_residue_elements(structure):
    """Map (resname, chain, resnum) -> set of element symbols (upper) present
    in that residue, from an already-parsed Biopython structure.

    edstats rows are per-residue, so this is looked up by the same
    (resname, chain, resnum) key as the row itself -- no per-atom matching
    against edstats lines is needed. `resnum` includes the insertion code
    (e.g. "42A"), matching how edstats concatenates it into the residue
    number token in stats.out; residues with no insertion code (icode " ")
    keep a bare numeric string (e.g. "42").
    """
    lookup = {}
    try:
        for model in structure:
            for chain in model:
                for residue in chain:
                    for residue in chain:
                        _, resseq, icode = residue.get_id()
                        resnum = f"{resseq}{icode.strip()}"
                        key = (residue.get_resname(), str(chain.id), resnum)
                        lookup.setdefault(key, set())
                        for atom in residue:
                            lookup[key].add(atom.element.upper())
        return lookup
    except Exception as e:
        raise RuntimeError(
            f"Failed to build residue-element lookup from structure: {e}") from e
    

def run_analysis(pdbID, stats_out, metals_set, cofactor_set, structure=None):
    """Parse an edstats stats.out file, returning (rows, header).
    `structure` is a parsed Biopython structure, shared with run_bond_analysis

    Cofactors are matched by CCD component name (fields[0]) against
    cofactor_set, as before. Plain metals are matched by the residue's
    actual atom element(s), read from `structure` -- a single-atom residue
    whose element is in metals_set is classified as a metal. This avoids
    misclassifying components whose CCD id happens to look like an element
    symbol (RNA "U", nitric oxide "NO") and catches metal-ion CCD ids that
    don't themselves match an element string (e.g. "FE2").
    If `structure` is not supplied, falls back to name-based metal matching
    (legacy behaviour) so existing callers without a parsed structure still work.
    

    Matching is done on the parsed residue-name token (fields[0]) against the
    given sets -- this fixes the old `line.startswith(f"{x:<4}")` approach, which
    padded names to 4 chars and so mishandled 5-character CCD ids.

    `rows` is a list of dicts: {pdbID, category, resname, chain, resnum, fields}
    where `fields` is the full whitespace-split edstats line (aligned with
    `header`). `header` is the column-label list from the file's first line.
    """

    if structure is None:
        raise ValueError(
            "A parsed structure is required for element-based metal identification")
    residue_elements = _build_residue_elements(structure)

    rows = []
    header = None
    with open(stats_out) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if header is None:
                # first non-empty line is the edstats column header
                header = fields
                continue
            resname = fields[0]
            chain = fields[1] if len(fields) > 1 else ""
            resnum = fields[2] if len(fields) > 2 else ""

            if resname in cofactor_set:
                category = "cofactor"
            else:
                elements = residue_elements.get((resname, chain, resnum))
                if elements and len(elements) == 1 and elements[0] in metals_set:
                    category = "metal"
                else:
                    continue

            rows.append({
                "pdbID": pdbID,
                "category": category,
                "resname": resname,
                "chain": fields[1] if len(fields) > 1 else "",
                "resnum": fields[2] if len(fields) > 2 else "",
                "fields": fields,
            })
    return rows, header


if __name__ == "__main__":
    # Legacy standalone behaviour: read a comma-separated PDB-id file, and for each
    # id (whose {id}_stats.out already exists in `directory`) dump per-metal data
    # files under {id}metals/. Batch PDB-REDO runs should use main.py instead.
    directory = BASE_DIR
    cofactor_set = load_cofactors()
    metals_set = set(metals) | set(uncommonMetals)

    fileIn = os.path.join(BASE_DIR, "alchemyTest.txt")
    with open(fileIn) as file:
        pdbString = file.read()
    pdbList = [s.strip() for s in pdbString.split(",") if s.strip()]
    print(pdbList)

    for pdbID in pdbList:
        stats_out = os.path.join(directory, f"{pdbID}_stats.out")
        rows, header = run_analysis(pdbID, stats_out, metals_set, cofactor_set)

        metal_directory = os.path.join(directory, f"{pdbID}metals")
        os.makedirs(metal_directory, exist_ok=True)
        # one file of matched lines per category, plus a sigma-sorted summary
        with open(os.path.join(metal_directory, "Metal_data"), "w") as mf, \
             open(os.path.join(metal_directory, "Cofactor_metal_data"), "w") as cf:
            for r in rows:
                line = " ".join(r["fields"]) + "\n"
                (mf if r["category"] == "metal" else cf).write(line)

        def _sigma(r):
            try:
                return abs(float(r["fields"][12]))
            except (IndexError, ValueError):
                return 0.0
        with open(os.path.join(metal_directory, "Metal_data_sorted"), "w") as sf:
            for r in sorted(rows, key=_sigma, reverse=True):
                sf.write(" ".join(r["fields"]) + "\n")

        print(f"{pdbID}: {sum(r['category'] == 'metal' for r in rows)} metals, "
              f"{sum(r['category'] == 'cofactor' for r in rows)} cofactors")
