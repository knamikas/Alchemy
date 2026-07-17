"""Extract PDB IDs from MetalPDB TSV exports."""

import os
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory = BASE_DIR

file="metalpdb_feion.tsv"
filename = os.path.join(BASE_DIR, "metalpdb_report.tsv")
hemefile = os.path.join(directory, "metalpdb_heme.tsv")
newhemefile = os.path.join(directory, "heme_ccds.txt")
new_file = os.path.join(directory, "fe_s_ccds.txt")

nifile = os.path.join(BASE_DIR, "metalpdb_report(1).tsv")
nilist = os.path.join(directory, "NI/ni_pdbs.txt")



ids = []
countsite = 0


with open(nifile, 'r') as f:
    for line in f:
        if '_' in line:        
            pdbID = line.split('\t')[0]
            info = line.split('\t')[2].strip()
            group =  line.split('\t')[-1]
            groups = group.split(';')
            sites = info.split(';')

            i = 0
##            while i < len(sites):
##                name = sites[i]
##                #if len(name) > 4:
##                ccd = name[:3]
##                if ccd in ids or '_' in ccd:
##                    pass
##                else:
##                    ids.append(ccd)
##                i += 1
            ids.append(pdbID[:4])

            countsite += 1

            #if pdbID in ids:
            #    pass
            #else:
            #    ids.append(pdbID)

with open(nilist, 'w') as f:
    i = 0
    while i < len(ids):
        f.write(f"{ids[i]}, ")
        i+=1
    

print(ids)
print(countsite)
