# Analysis v2: Updated capabilities to include reocgnizing cofactors containing
# metal atoms

# Extract metal data from Alchemy files

# make sure folders do not already exist when running the program, otherwise
# error is returned

import os
import glob

# Set directory containg Alchemy outputs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory = BASE_DIR

#common metals to search for
metals = ['NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN']

#uncommon metals to search for
uncommonMetals = ['CD', 'HG', 'PT', 'MO', 'AL', 'BE', 'BA', 'RU', 'V', 'SR', 'CS',
                  'W', 'AU', 'YB', 'LI', 'GD', 'PB', 'U', 'Y', 'LR', 'TI', 'RB',
                  'AG', 'SM', 'OS', 'PR', 'PD', 'EU', 'TB', 'RE', 'RH', 'TA', 'LU',
                  'HO', 'CR', 'GA', 'LA', 'SN', 'SB', 'CE', 'ZR', 'ER', 'TH', 'IN',
                  'HR', 'SC', 'DY', 'BI', 'PA', 'PU', 'AM', 'CM', 'CF', 'GE', 'NB',
                  'TC', 'ND', 'PM', 'TM', 'PO', 'FR', 'RA', 'AC', 'NP', 'BK', 'ES',
                  'FM', 'MD', 'NO', 'LR', 'RF', 'DB', 'SG']

# cofactors to search for
#cofactorFile = input("Enter Alloy output: ")
# if elif else checks to make sure it's actually an ID we're adding; the cif
# file contains names and IDs and someones split a ; or \n as a seperate entry
# so this step takes those out. Also filtered out in this step is the IDs for
# plain metal ions, which have an ID = atomic symbol (less than 3 char)
cofactors = []
formulas = []
cofactorFile = os.path.join(directory, "metallocofactors_id.txt")
with open(cofactorFile) as read_f:
        for line in read_f:
                fields = line.split(f"\t")
                cofactorid = fields[0].strip()
                cofactors.append(cofactorid)
                formula = fields[1].strip()
                formulas.append(formula)

                                
print(f"Searching common metals: {metals}\nuncommon metals: {uncommonMetals}\ncofactors: {cofactors}")
print(f'Searching for {len(metals)+len(uncommonMetals)} metals and {len(cofactors)} metal-containing cofactors')                


#fileIn = input ("Enter CSV file for PDB identifiers: ")
fileIn = os.path.join(BASE_DIR, "alchemyTest.txt")


with open(fileIn, "r") as file:
	pdbString = file.read()

# make a list of strings, each string is one pdb ID and clear each entry of any
# extra spaces

pdbList = [x.strip() for x in pdbString.split(",")]
pdbList = [s for s in pdbList if s]
print(pdbList)

print("Starting with: "+f'{metals}')

i = 0
while i<len(pdbList):
    pdbID=pdbList[i]

# make a directory for the pdb ID and set the path to metal_directory
    os.makedirs(os.path.join(directory, f'{pdbID}metals'))
    metal_directory=os.path.join(directory, pdbID+"metals")

#note: as is this would overwrite the data in each file every time program is
#run. not an issue here as different directories are created for each pdb ID
#and the txt metal files are placed inside, but if you want the program to
#do something like create a single file containing all data 'a' should be used
#instead of 'w'. 
    print("Searching for common metals")

    for metal in metals:
        
        output_file= f"{metal}_Data"
        input_file=f'{pdbID}_stats.out'
        with open(os.path.join(metal_directory, output_file), 'w') as out_f1:
            with open(os.path.join(directory, input_file)) as f:
                
                for line in f:
                    if line.startswith(f"{metal:<4}"):
                        out_f1.write(line[:100] + "\n")     
                        

#check for MSE residues
    input_file2=f'{pdbID}_stats.out'
    with open(os.path.join(metal_directory, "MSE_Data"), 'w') as out_f2:
        with open(os.path.join(directory, input_file2)) as f:
            for line in f:
                if line.startswith("MSE "):
                    out_f2.write(line[:100]+"\n")

    print("checking for uncommon metals")

# check for uncommon metals. should append each line to the file, and not overwrite
#previous entries for other metals. 
    for uncommonMetal in uncommonMetals:
        output_file2 = os.path.join(metal_directory, "Other_metal_data")
    
        with open(output_file2, 'a') as out_f3:
            with open(os.path.join(directory, input_file), 'r') as f:
                for line in f:
                    if line.startswith(f"{uncommonMetal:<4}"):
                        out_f3.write(line[:100]+"\n")

# check for metal containing cofactors - ** known issue rn is that new ccd ids are 5
# char long this script checks first 4 characters
    for cofactor in cofactors:
        output_file3 = os.path.join(metal_directory, "Cofactor_metal_data")
    
        with open(output_file3, 'a') as out_f3:
            with open(os.path.join(directory, input_file), 'r') as f:
                for line in f:
                    if line.startswith(f"{cofactor:<4}"):
                        out_f3.write(line[:100]+"\n")
                

    i+=1

# write the metals found to a common file

combined_common_metals=[]
for metal in metals:
    with open(os.path.join(metal_directory, f'{metal}_Data'), 'r') as f:
        combined_common_metals.extend(f.readlines())
        
with open(os.path.join(metal_directory, "Metal_data_difference"), 'w') as out_file:
    out_file.writelines(combined_common_metals)

# sort on sigma score into a new file

def get_sigma(line):
    fields = line.split()
    if len(fields) >= 13:
        try:
            return abs(float(fields[12]))
        except ValueError:
            return 0
    return 0

sorted_lines = sorted(combined_common_metals, key=get_sigma, reverse=True)

with open(os.path.join(metal_directory, "Metal_data_difference_sorted"), 'w') as sorted_file:
    sorted_file.writelines(sorted_lines)


# print summary

print("Metal ion counts")

total = 0

for metal in metals:
    with open(os.path.join(metal_directory, f'{metal}_Data'), 'r') as f:
        lines = f.readlines()
        count = len(lines)
        total += count
        print(f'Number of {metal.capitalize()} ions: {count}')

with open(os.path.join(metal_directory, 'MSE_Data'), 'r') as f:
    mse_count = len(f.readlines())
    print(f"Number of Se ions: {mse_count}")

with open(os.path.join(metal_directory, 'Other_metal_data'), 'r') as f:
    other_count = len(f.readlines())
    print(f"Number of other metal ions: {other_count}")

with open(os.path.join(metal_directory, 'Cofactor_metal_data'), 'r') as f:
    cofactor_count = len(f.readlines())
    print(f"Number of other metal-containing cofactors: {cofactor_count}")
        
