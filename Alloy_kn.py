# Alloy 6/12/2025
# Script to download CCD and make list of all metallocofactors in the PDB
# CCD typically updated weekly with PDB, so script asks user if desired to
# download most recent CCD file
# need sh, gemmi python packages installed

# Could change search criteria to create custom lists of ligands to search for

import requests
from sh import gunzip
import os
import gemmi
import re
debug_mode = 1

URL_CCD = "https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz"

# function that returns True if the compound to be checked contains a metal and
# otherwise returns False
def find_metal_match(string):
    j = 0
    while j < len(metals):
        matches = re.findall(metals[j], string)
        if len(matches) == 0 :
            pass
        else:
            has_metal = True
            return True
        j += 1
    return False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if debug_mode == 1:
    directory = BASE_DIR
else:
    directory = input("What directory should file be downloaded and unzipped to?")



if debug_mode == 1:
    pass
else:
    
# loop asks for user input to determine if CCD file should be updated
# also unzips file
    while True:
        download_again = input("Would you like to download most recent CCD? y/n\n")

        if download_again == "y":
            response = requests.get(URL_CCD, stream=True)
            if response.status_code == 200:
                with open(os.path.join(directory, 'components.cif.gz'), 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                gunzip(os.path.join(directory, 'components.cif.gz'))
                print(f"File downloaded")
                break
            else:
                print(f"error downloading file: {response.status_code}")
                break
        elif download_again == "n":
            break

        else:
            print("Please enter only y or n")

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


# load and read CCD cif file

cifFile = os.path.join(directory, "components.cif")

print("Reading CCD")

#if debug_mode == 1:
#    ccd = gemmi.cif.read_file(os.path.join(directory, "testCIF.cif"))
#else:
ccd = gemmi.cif.read_file(cifFile)


# because we append info to files, delete old copies
if os.path.exists(os.path.join(directory, "metallocofactors_id.txt")):
    os.remove(os.path.join(directory, "metallocofactors_id.txt"))
if os.path.exists(os.path.join(directory, 'missing_formulas.txt')):
    os.remove(os.path.join(directory, 'missing_formulas.txt'))
if os.path.exists(os.path.join(directory, 'missingCIF.cif')):
    os.remove(os.path.join(directory, 'missingCIF.cif'))

# set up some variables to count how many components are anlyzed and
# an empty cif doc
missingCIF = gemmi.cif.Document()
count = 0
count2 = 0
count3 = 0
count4 = 0

#read through CCD file
print(f'Analyzing {len(ccd)} components in CCD')
for block in ccd:
#extract the values for the component that we are interested in maybe saving
    comp_id = block.find_value('_chem_comp.id')
    formula = block.find_value('_chem_comp.formula')
    upper_formula = formula.strip().upper()

# don't want to include CCD for a plain metal ion
    if upper_formula in metals:
        count4 += 1
        pass


# generic marker for unknown ligand - dont want to include in lists
    elif comp_id  == 'UNL':
        pass

# If there is no formula recorded, there is a ? returned.  We will instead
# analyze by individual atom if that's the case 
    elif formula == '?':
        count2 += 1

# adding blocks to a cif file for components missing formula - this would be a
# way to check for errors but can be commented out
        block = ccd.find_block(f'{comp_id}')
        add_block = missingCIF.add_copied_block(block, pos=-1)

# keep a doc with IDs of components missing formulas - also for checking and
# debugging purposes, can be commented out
        with open(os.path.join(directory, 'missing_formulas.txt'), 'a') as f_write:
            f_write.write(f"{comp_id}\n")

# use atom symbols instead to check if any metals are present
        atom_symbols = block.find_values('_chem_comp_atom.type_symbol')
        i = 0
        while i < len(atom_symbols):
            symbol = atom_symbols[i]
            has_metal = find_metal_match(symbol)
            if has_metal == True:
                count3 += 1
                break
            i += 1
# if theres a formula, search the string with each metal abbreviation until a
# match is found. since we don't care about recording every metal, just if at
# least one is present, we can stop checking as soon as the first is found
    else:
        has_metal = find_metal_match(upper_formula)
        if has_metal == True:
            count += 1

# save info to a seperate file if there's a metal
    if has_metal == True:
#            print(f'{comp_id} has metal\n')
        with open(os.path.join(directory, "metallocofactors_id.txt"), 'a') as f_write:
            f_write.write(f"{comp_id}\t{formula}\n")
    else:
        continue

missingCIF.write_file(os.path.join(directory, "missingCIF.cif"))

print(f"Found {count} components with metal")
print(f'Skipped {count4} metal ions')
print(f'Checked {count2} components missing formulas and added {count3} that contained metals to metallocofactors list')

    




    
        


