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

def extract_metal_statistics(pdbID, stats_out, metals_set, cofactor_set, structure=None):
    """Parse an edstats stats.out file, returning (rows, header).
    `structure` is the shared first-model Gemmi context used by bond analysis.

    Cofactors are matched by CCD component name (fields[0]) against
    cofactor_set, as before. Plain metals are matched by the residue's
    actual atom element(s), read from `structure` -- a single-atom residue
    whose element is in metals_set is classified as a metal. This avoids
    misclassifying components whose CCD id happens to look like an element
    symbol (RNA "U", nitric oxide "NO") and catches metal-ion CCD ids that
    don't themselves match an element string (e.g. "FE2").

    Matching is done on the parsed residue-name token (fields[0]) against the
    given sets, including CCD ids longer than four characters.

    Output is site-level: a multi-metal cofactor repeats its residue-level
    EDSTATS values once per selected metal site. Each row carries ``site`` and
    ``site_key`` internally so downstream contact summaries cannot collide for
    multiple metals or duplicate author residue identifiers.

    A cofactor row that has no matching coordinate residue or no selected metal
    site is retained once with ``site=None``. Machine-readable row status fields
    distinguish an identifier-join failure from a matched cofactor that simply
    has no selected configured metal. This preserves the residue-level EDSTATS
    observation without pretending that a metal site was available for geometry
    analysis.
    """

    if structure is None:
        raise ValueError(
            "A parsed structure is required for element-based metal identification")
    metals_upper = {element.upper() for element in metals_set}

    rows = []
    header = None
    model_column = None
    with open(stats_out) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if header is None:
                # first non-empty line is the edstats column header
                header = fields
                model_column = next(
                    (index for index, name in enumerate(header)
                     if name.upper() == "MN"),
                    None,
                )
                continue
            if model_column is not None:
                if model_column >= len(fields):
                    # EDSTATS emits one short "_" separator row when XYZIN
                    # contains an explicit MODEL/ENDMDL wrapper.
                    if fields and fields[0] == "_":
                        continue
                    raise ValueError("EDSTATS row is missing its MN model value")
                try:
                    row_model = int(fields[model_column])
                except ValueError as exc:
                    raise ValueError(
                        f"invalid EDSTATS MN model value: "
                        f"{fields[model_column]!r}") from exc
                if row_model != structure.model_analyzed:
                    raise ValueError(
                        f"EDSTATS returned model {row_model}, but Alchemy's "
                        f"model policy selected model {structure.model_analyzed}")
            resname = fields[0]
            chain = fields[1] if len(fields) > 1 else ""
            if chain in (".", "?"):
                chain = ""
            resnum = fields[2] if len(fields) > 2 else ""
            matched_residues = structure.residues_for_author(
                resname, chain, resnum)
            if not matched_residues:
                mapping_status = "coordinate_residue_not_found"
            elif len(matched_residues) == 1:
                mapping_status = "matched"
            else:
                mapping_status = "multiple_coordinate_residues"

            is_cofactor = resname in cofactor_set
            emitted_site = False
            for residue in matched_residues:
                metal_sites = [atom for atom in residue.contact_atoms
                               if atom.element_known and
                               atom.element in metals_upper]
                if is_cofactor:
                    category = "cofactor"
                elif (residue.chemical_atom_site_count == 1 and
                      len(metal_sites) == 1):
                    category = "metal"
                else:
                    continue
                for site in metal_sites:
                    emitted_site = True
                    rows.append({
                        "pdbID": pdbID,
                        "category": category,
                        "resname": resname,
                        "chain": chain,
                        "resnum": resnum,
                        "fields": list(fields),
                        "coordinate_mapping_status": mapping_status,
                        "selected_metal_site_status": "selected",
                        "site": site,
                        "site_key": site.source_key,
                        "residue_key": residue.key,
                    })

            if is_cofactor and not emitted_site:
                rows.append({
                    "pdbID": pdbID,
                    "category": "cofactor",
                    "resname": resname,
                    "chain": chain,
                    "resnum": resnum,
                    "fields": list(fields),
                    "coordinate_mapping_status": mapping_status,
                    "selected_metal_site_status": "no_selected_metal",
                    "site": None,
                    "site_key": None,
                    "residue_key": (matched_residues[0].key
                                    if len(matched_residues) == 1 else None),
                })
    return rows, header
