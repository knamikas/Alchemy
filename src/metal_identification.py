# Analysis v2: extract metal / metallocofactor real-space stats from edstats output.
#
# edstats `stats.out` is a whitespace table: the first non-empty line is the column
# header, and each data line begins with a residue/component name. Column layout
# (0-indexed): 0 = residue name (RT), 1 = chain (CI), 2 = residue number (RN).
#
# `extract_metal_statistics` returns structured rows for metal ions and metal-containing
# cofactors; `main.py` aggregates these across many structures.

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


def load_cofactor_ids(path=None):
    """Return a set of metal-containing CCD ids from metallocofactors_id.txt.

    The file is "{ccd_id}\\t{formula}" per line (see build_metallocofactor_catalog.py). We keep only
    the id token.

    If `path` is not given, defers to build_metallocofactor_catalog.active_cofactors_path(), which
    picks up an untracked, per-user cache refresh if one exists and falls
    back to the bundled `src/data` copy otherwise.
    """
    if path is None:
        from build_metallocofactor_catalog import active_cofactors_path  # lazy: avoid import cycle at module load
        path = active_cofactors_path()
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
    

def extract_metal_statistics(pdbID, stats_out, metals_set, cofactor_set, structure=None):
    """Parse an edstats stats.out file, returning (rows, header).
    `structure` is a parsed Biopython structure, shared with run_bond_analysis

    Cofactors are matched by CCD component name (fields[0]) against
    cofactor_set, as before. Plain metals are matched by the residue's
    actual atom element(s), read from `structure` -- a single-atom residue
    whose element is in metals_set is classified as a metal. This avoids
    misclassifying components whose CCD id happens to look like an element
    symbol (RNA "U", nitric oxide "NO") and catches metal-ion CCD ids that
    don't themselves match an element string (e.g. "FE2").

    Matching is done on the parsed residue-name token (fields[0]) against the
    given sets, including CCD ids longer than four characters.

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
                if elements and len(elements) == 1 and next(iter(elements)) in metals_set:
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
