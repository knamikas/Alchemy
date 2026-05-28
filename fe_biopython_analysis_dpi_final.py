# Script to calculate bond lengths between metals and protein atoms
# output: CSV with 1 line for every Fe-protein bond

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa, standard_aa_names
from Bio.SeqUtils import seq1
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) #bunch of extra warnings printed for
warnings.filterwarnings("ignore", category=UserWarning) # every entry dont want them printing
                                                        # for 10,000 entries
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
debug = 1

# make this 1 if you want only positive distributions and -1 if you want both sides
sign =-1

# these metals have bond length data from harding 2006 in metal_distances_info.
# used 0.2 as the standard deviation for residues that indicated dev >0.2 for
# calculation purposes
# TYR, SER, and THR residue M-OH bond lengths are given approximately in the 2006 paper
# tyr - 0.1 less
# ser & thr are between hoh & cooh
# i left the st dev in the file the same as the original table 4 for tyr
# for thr and ser I used the avg as suggested in fig caption 4 - kept higher dev

metals = ['NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN']

aa = ['ASP', 'CYS', 'GLU', "HIS", "TYR", "CYS", 'THR', 'SER', 'HOH']

cutoff = 4.0 # in angstroms - max distance away for neighbor search
z_cutoff = 10 # number of st deviations from avg bond length to plot
max_distance = 4.0 # max distance two atoms can be to still be considered
                   # interacting - taken from Gucwa et al 2024 and the current
                   # CMM program

w = 14.12 # avg atomic weight

skipped = []
# calculate the DPI - the st deviation of atom coordinates for the model. used equation 7
# from Blow 2002, which depends on R free. This formula accounts for resolution, as
# well as other factors like number of measurements and completeness. 
def calculate_dpi(pdbStructure, pdbID):
        global skipped
        #find matthews coefficient Vm
        vm = None
        nobs = None
        rfree = None

        with open(os.path.join(directory, f'{pdbID}.pdb'), 'r') as f:
                for line in f:
                        if "MATTHEWS COEFFICIENT" in line:
                                vm = line.split()[-1]
                        if "UNIQUE REFLECTIONS" in line:
                                nobs = line.split()[-1]
                        if "FREE R VALUE" in line and "TEST" not in line and "ESTIMATED" not in line and 'BIN' not in line:
                                match = re.search(r"FREE R VALUE\s*:\s*(\d+\.\d+)", line)
                                if match:
                                        rfree = match.group(1)
                        if "RESOLUTION RANGE HIGH (AN" in line:
                                res = line.split()[-1]
                        if vm and nobs and rfree and res:
                                break
        if vm == None or vm =="NULL":
                print(f"No matthews found {pdbID}")
                #skipped.append(pdbID)
                return
        if nobs == None or nobs=="NULL":
                print(f"Unique reflections not included {pdbID}")
                #skipped.append(pdbID)
                return
        if rfree == None or rfree=="NULL":
                print(f"R free not found {pdbID}")
                #skipped.append(pdbID)
                return
        if res == None or res =='NULL':
                print(f"Resolution not found {pdbID}")
                #skipped.append(pdbID)
                return
        # count atoms/solvent and calculate MW of the asymmetric unit
        # the structure contains the whole PDB file, which is 1 asymmetric
        # unit by convention
        number_atoms_mw = 0
        number_w = 0
        for model in structure:
                for chain in model:
                        for residue in chain:
                                hetfield = residue.id[0]
                                #exclude water
                                if hetfield == "W":
                                        number_w += 1
                                for atom in residue:
                                        number_atoms_mw += 1
        mw = round(w*int(number_atoms_mw))
        va = round(mw*float(vm))
        ni = int(number_atoms_mw)+int(number_w)
        #print(f'MW estimate: {mw}')
        #print(f'Va: {va}')
        #print(f'Number of molecules in 1 asymmetric unit: {ni}')
        # calculate DPI
        dpi = 1.28 * (ni ** 0.5) * (va ** (1/3)) * (float(nobs) ** (-5/6)) * float(rfree)
        rounded_dpi = round(dpi, ndigits=4)
        return rounded_dpi, res

# function determines what should be searched in the file containing bond
# information in the format (parent residue) (neighbor atom type) (metal)
def get_bonding_info(name):
        #water
        if neighbor_name == "HOH":
                bonding_info = f'HOH O {metal.element}'
                
        elif neighbor.get_name() == "O":
                # CA = alpha carbon, for carbonyl groups
                bonding_info = f'CA O {metal.element}'
                #print(neighbor_name)

        # for side chain oxygens - neighbor name is smth like OE1 or OD1                        
        elif neighbor.get_name().startswith("O"):
                bonding_info = f"{neighbor_name} O {metal.element}"

        #something else - like his N
        else:
                bonding_info = (f'{neighbor_name},{neighbor.element},'
                                f'{metal.element}')
        return bonding_info

# function to read the file of bond distances and calculate z score
# it's set up to calculate a difference from each set of bond information from zheng 2008, which
# had seperate values based on oxidation number, pdb, and csd. however im not using that information
# for anything right now, every further step is calcualted based on the first z score, from
# hardings paper. for the standard version (file with multiple metal types) the lists are
# unecessary because theres only 1 distance value per bond. 
def find_dist(info, dpi):
        lines = []
        diffs = []
        groups = []
        scores = []
        with open(os.path.join(directory, "fe_distances_literature_combined.csv"), 'r') as f:
                for line in f:
                        if info in line:
                                lines.append(line.strip())
        if len(lines) == 0:
            return


        i = 0
        while i < len(lines):
            lit_dist = lines[i].split(',')
            print(lit_dist)
        
            difference = dist2-float(lit_dist[5])
            diffs.append(difference)
            groups.append(f'{lit_dist[0]}')
            st_dev = float(lit_dist[6])

            zscore = difference / ((dpi**2 + st_dev**2) **.5)
            scores.append(zscore)
            i += 1

        close_i = diffs.index(min(diffs, key=abs))
        
        if sign == 1:
                zscore_use_avg = abs(round(scores[0], ndigits=4))
                zscore_closest = abs(round(scores[close_i], ndigits=4))
        if sign == -1:
                zscore_use_avg = round(scores[0], ndigits=4)
                zscore_closest = round(scores[close_i], ndigits=4)
                zscore_pdb2=round(scores[1], ndigits=4)
                zscore_pdb3=round(scores[2], ndigits=4)
                zscore_csd2=round(scores[3], ndigits=4)
                zscore_csd3=round(scores[4], ndigits=4)
        

        result = [zscore_use_avg, groups[close_i], zscore_closest,
                  zscore_pdb2, zscore_pdb3, zscore_csd2, zscore_csd3]
        
        return result
        

# open the metal data file to get the sigma value
def get_sigma(atom, chain, number):
        
        if atom in metals:
                search_for = f'{atom}  {chain} {number}'
                try:
                        #print('trying metal')
                        with open(os.path.join(metal_directory, f'{atom}_Data')) as f:
                                for line in f:
                                        if search_for in line:
                                        #print(f'{line}')
                                                mag = float(line.split()[12])
                                                neg = float(line.split()[13])
                                                pos = float(line.split()[14])
                                                return mag, neg, pos
                except:
                        skipped.append(pdbID)
        else:
                search_for = f'{atom} {chain} {number}'
                try:
                        #print('trying cofactor')
                        with open(os.path.join(metal_directory, 'Cofactor_metal_data')) as f:
                                  for line in f:
                                        if search_for in line:
                                        #print(f'{line}')
                                                mag = float(line.split()[12])
                                                neg = float(line.split()[13])
                                                pos = float(line.split()[14])
                                                return mag, neg, pos
                except:
                        skipped.append(pdbID)


        
# loop lets you set fixed file locations or enter new ones each run.         
if debug == 1:
        # csv list of pdb ids to analyze 
        fileIn = os.path.join(BASE_DIR, "missingids.txt")
        directory = BASE_DIR


else:
        fileIn = input ("Enter CSV file for PDB identifiers: ")
        directory = input("Enter directory containing files: ")


# make a list of PDB IDs to analyze
with open(fileIn, "r") as file:
	pdbString = file.read()
pdbList = [x.strip() for x in pdbString.split(",")]
pdbList = [s for s in pdbList if s]
#print(pdbList)

cluster = ['F3S', 'SF4', 'FES', 'SRM', 'HC1', 'HC0', 'CFM', 'CLP', 'CLF', 'FE2',
           'WCC', 'FS1', 'NFS', 'F4S', 'XCC', 'CU1', 'SF3', 'FSX', 'HCN', '1CL',
           'FS5', '402', '4WW', '4WX', '4WV', '6ML', '9S8', 'Q46', '9SQ', '8JU',
           'LPJ', 'FU8', 'CUV']


hemes = ['HEM', 'HEC', 'DHE', 'SF4', 'SRM', 'HEA', 'CCH', 'HAS', 'HEV',
         'HEO', 'HDD', 'HE6', 'FDE', 'FDD', 'HDE', 'FE2', 'HEB', 'CU1', 'HCO',
         'HFM', 'HIF', '2FH', 'POR', 'FMI', 'HME', '1FH', '7HE', '6HE', 'VEA',
         'VER', 'YT3', 'HDM', 'HKL', 'MOO', 'FEC', 'CLN', 'NTE', 'CUB', 'MH0',
         'ISW', 'OBV', 'UFE', '522', 'WVP', 'WYP', 'X8P', 'WUP', 'WXP', '89R',
         'HP5']


# set up columns for a dataframe to store info during each iteration of the loop
columns = ['distance',
           'zscore_to_avg',
           'literature closest distance',
           'zscore_to_closest',
           'zscore_pdbhr2',
           'zscore_pdbhr3',
           'zscore_csd2',
           'zscore_csd3',
           'sigma_mag',
           'sigma_neg',
           'sigma_pos',
           'atom_formulas', # atom name chain atom number
           'parent',
           'type',
           'bonded_to', # protein backbone or water
           'element',
           'neighbor',
           'residue number',
           'pdb id',
           'resolution'
           ]

all_bonds = pd.DataFrame()

# start over with empty list for each pdb entry - can print to see what kinds of bonds are
# common but no data exists for
no_bond_data = []
i = 0
while i < len(pdbList):
    pdbID = str(pdbList[i])
    print(pdbID)
   
    # reset data frame for each pdb ID 
    dist_info = pd.DataFrame(columns=columns)

    # analysis output
    metal_directory = os.path.join(directory, f'{pdbID}metals')


    # make parser, get pdb structure and make a list of the atoms
    parser = PDBParser(QUIET=True)
    #print('got parser')
    structure = parser.get_structure(f'{pdbID}', os.path.join(directory, f'{pdbID}_rszd.pdb'))
    #print('got structure')
    
    atoms=list(structure.get_atoms())

    # calculate DPI for later - returns DPI rounded to 4 decimal places
    # if the pdb file is missing or formatted wrong the entry is skipped
    # also stores resolution
    dpi_res = calculate_dpi(structure, pdbID)

    print(dpi_res)
    
    if dpi_res == None:
            print(f'No DPI, skipping {pdbID}')
            i += 1
            skipped.append(pdbID)
            continue
    else:
            dpi = dpi_res[0]
            resolution = dpi_res[1]

    # pull out Fe atoms - the if ( and ) statement ensures we only pull metal ions,
    # to ignore metals in a cofactor add second condition to if statement:
    # and atom.get_parent().get_resname() in metals
    metal_atoms = []
    for atom in atoms:
        element = atom.element.upper()
        if element == 'FE':
                metal_atoms.append(atom)
    #print(metal_atoms)

    # set up neighbor search
    ns = NeighborSearch(atoms)

    # print name of metal 
    for metal in metal_atoms:
        metal_id = (f'{metal.get_parent().get_resname()}'
                    f'{metal.get_parent().get_id()} {metal.get_full_id()[2]}')
        #print(f'\nmetal: {metal.element} at {metal.coord} ({metal_id})')
        

        # do the neighbor search
        neighbors = ns.search(metal.coord, cutoff, level='A')

        # look at every neighbor atom
        for neighbor in neighbors:

            
# skip bonds to other atoms in the cofactor - we only care about bonds
# to the protein atoms
            if metal.get_parent().get_resname() == (neighbor.get_parent().
                                                    get_resname()):
                pass
# skip neighbors in other cofactors - don't have bond info for them
            elif neighbor.get_parent().get_resname() not in aa:
                    pass

            else:
                neighbor_name = neighbor.get_parent().get_resname()
                print(neighbor.get_parent().get_full_id())

                # get distance between neighbor atom and metal atom
                # automatically includes distance to itself - skip it
                dist = float(metal - neighbor)
                dist2 = round(dist, 3)
                if dist > max_distance or dist2==0.0:
                        pass
                
                #print(dist2)

                else:
                        bonding_info = get_bonding_info(neighbor.get_name())

                        # call function to find literature bond distance 
                        result = find_dist(bonding_info, dpi)
                        print(result)

                        # keep track of bonds we dont have data for
                        if result is None:
                                no_bond_data.append(f'{metal.element} bonded to: {neighbor_name} '
                                                    f'{neighbor.element} {metal.element}')

                        else:
                                name = metal.get_parent().get_resname()
                                print(name)
                                chain = metal.get_full_id()[2]
                                number = metal.get_parent().get_id()[1]
                                atom_name = f'{metal.get_id()}{chain}{number}'
                                if name == 'FE':
                                        sig = get_sigma(metal.element, chain, number)
                                else:
                                        sig = get_sigma(name, chain, number)
                                        
                                        
                                
                                if sig == None:
                                        mag = 'NaN'
                                        neg = 'NaN'
                                        pos = 'NaN'
                                else:
                                        mag = sig[0]
                                        neg = sig[1]
                                        pos = sig[2]


                                if metal.get_parent().get_resname() in cluster:
                                        parent_type = 'cluster'
                                elif metal.get_parent().get_resname() == 'FE':
                                        parent_type= 'ion'
                                elif metal.get_parent().get_resname() in hemes:
                                        parent_type = 'heme'
                                else:
                                        parent_type = 'other'
                                
                                # decide what to tag the bonding neighbor for plot marker shape 
                                if neighbor_name == "HOH":
                                        bonded= "HOH"
                                elif neighbor_name in aa:
                                        bonded="P"
                                else:
                                        bonded=f'{neighbor_name}'
                                        #print(f'{metal.element} bonded to {neighbor.get_full_id()}')

                                # add the new row to the dataframe
                                new_row = {
                                        'distance': dist2,
                                        'zscore_to_avg': result[0],
                                        'literature closest distance': result[1],
                                        'zscore_to_closest': result[2],
                                        'zscore_pdbhr2': result[3],
                                        'zscore_pdbhr3': result[4],
                                        'zscore_csd2': result[5],
                                        'zscore_csd3': result[6],
                                        'sigma_mag': mag,
                                        'sigma_neg': neg,
                                        'sigma_pos': pos,
                                        'atom_formulas': atom_name,
                                        'parent': metal.get_parent().get_resname(),
                                        'type': parent_type,
                                        'bonded_to': bonded,
                                        'element': metal.element,
                                        'neighbor': neighbor_name,
                                        'residue number': metal.get_parent().get_id()[1],
                                        'pdb id': pdbID,
                                        'resolution': resolution
                                        }

                                #print(new_row)
                                dist_info = pd.concat([dist_info, pd.DataFrame([new_row])],
                                                      ignore_index=True)
    #print(dist_info)
                                                                        
    if not dist_info.empty:
            print(dist_info)
            dist_info.to_csv(os.path.join(directory, 'fe_bonds_all_final.csv'), mode='a', index=False, sep=',', header=False)


            
    i +=1
##
##print(fe_only)
#all_bonds.to_csv(os.path.join(directory, 'fe_bonds_all.txt'), index=False, sep='\t')

print(f'Couldnt find pdb files for {len(skipped)} pdbids')
#with open(os.path.join(directory, 'pdbfileonhd.txt'), 'w') as f:
#        f.write(skipped)

print(skipped)
missing = [s for s in pdbList if s not in skipped]



#print(stats_df(stats_df['prop'] > .37))
#stats_df.to_csv(os.path.join(directory, 'fe_sample_stats.csv'), index=False)
