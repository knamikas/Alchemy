# Autoplot python version 6/16/2025
# plot structure and maps with CCP4mg, from edstats outputs
# *doesnt do anything with MSE data
# v2 updated to handle collecting coordinates for metal atoms in cofactors and
# also outputting images for these atoms


import os
from decimal import Decimal
import shutil
import fileinput
import subprocess
import re

# debug = 1 turns off image generation
debug = 0

# Set directory containg Alchemy outputs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory = BASE_DIR

metals = ['NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN']

allMetals = ['NA', 'MG', 'K', 'CA', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
          'CD', 'HG', 'PT', 'MO', 'AL', 'BE', 'BA', 'RU', 'V', 'SR',
          'CS', 'W', 'AU', 'YB', 'LI', 'GD', 'PB', 'U', 'Y', 'LR',
          'TI', 'RB', 'AG', 'SM', 'OS', 'PR', 'PD', 'EU', 'TB', 'RE',
          'RH', 'TA', 'LU', 'HO', 'CR', 'GA', 'LA', 'SN', 'SB', 'CE',
          'ZR', 'ER', 'TH', 'IN', 'HR', 'SC', 'DY', 'BI', 'PA', 'PU',
          'AM', 'CM', 'CF', 'GE', 'NB', 'TC', 'ND', 'PM', 'TM', 'PO',
          'FR', 'RA', 'AC', 'NP', 'BK', 'ES', 'FM', 'MD', 'NO', 'LR',
          'RF', 'DB', 'SG']


#fileIn = input ("Enter CSV file for PDB identifiers: ")
fileIn = os.path.join(BASE_DIR, "alchemyTest.txt")

with open(fileIn, "r") as file:
	pdbString = file.read()

# make a list of strings, each string is one pdb ID and clear each entry of any
# extra spaces

pdbList = [x.strip() for x in pdbString.split(",")]
pdbList = [s for s in pdbList if s]
print(pdbList)

# Set some variables

count = 0
chain=""
atom=""
number=""
target=""
filename=""

# set some more variables that set rms difference limits for the
# map plots

# set 2F-Fc to 2 sigma and difference maps to 3 sigma

SigmaFo=Decimal("2.0")
SigmaPdf=Decimal("3.0")
SigmaNdf=Decimal("-3.0")

# function to open pdb file, read each line and add the coordinates of the metal to a list.
# the input file must have a line for each metal ion - hence not used for cofactor search
# input is the list to append coordinates to and the name of the pdb file to search 

def coord_search(mylist, file_to_search):
        with open(file_to_search,'r') as f_read:
                for line in f_read:
                        if (f'{atom} {chain} {number}' in line or
                            f'{atom} {target}' in line):
# count from number index bc its always column immediately to left
                                fields2 = line.split()
                                ni = fields2.index(f'{number}') # number index
                                                                
                                xcoord = float(fields2[ni+1])*-1
                                ycoord = float(fields2[ni+2])*-1
                                zcoord = float(fields2[ni+3])*-1
                                coord = f'{xcoord}, {ycoord}, {zcoord}'
                                print(coord)
                                mylist.append(coord)

def read_file(filename, path):
    file = os.path.join(path, filename)
    file_size = os.path.getsize(file)
    if file_size == 0:
        print(f"No data from {filename}")
    else:
        with open(file, 'r') as f:
            for line in f:
    # read each line in the metal_Data file, extract the information we want 
                fields = line.split()
                atom = fields[0]
                chain = fields[1]
                number = fields[2]
                sigmaf = fields[12]
                target = f'{chain}{number}'
                print(f'atom {atom}, chain {chain}, number {number}')

    # set up these lists to make naming files more straightforward later on
                atoms.append(atom)
                chains.append(chain)
                numbers.append(number)

    #use the metal information to search the pdb file for coordinates:
                coord_search(metal_coords, os.path.join(directory, f'{pdbID}_rszd.pdb'))

    # make the legend and save it for later 
                legend = f'{pdbID} {atom} {target} {sigmaf} sigma'
                legends.append(legend)
                print(legend)

        

                                                        

# define function to pull RMS information from edstats.log file as well as calculate rmsp/rmsn
# rms is always positive so we start with a negative number to indicate it hasn't been
# updated. used decimal module to prevent float errors        
def rms(pdbID):
        rms_2fo = -1
        rms_dmf = -1
        with open(os.path.join(directory, pdbID+"_edstats.log"), 'r') as f:
                for line in f:
                        if "Rms deviation from mean density" in line:
                                fields3 = line.split()
                                if rms_2fo == -1:
                                        rms_2fo = Decimal(f'{fields3[-1]}')*SigmaFo
                                else:
                                        rms_dmf = Decimal(f'{fields3[-1]}')
                                        rms_dmfp= rms_dmf*SigmaPdf
                                        rms_dmfn=rms_dmf*SigmaNdf
                                        print(f'2Fo-Fc {rms_2fo}, Fo-Fc +ve and -ve {rms_dmfp}, {rms_dmfn}')
                                        return [rms_2fo, rms_dmfp, rms_dmfn]
                
# define function to copy then edit default.mgpic file with the info of a single metal.
# will be called 3 times per metal to make each orientation image

def make_images(filename, orientation):
        # copy template
        shutil.copy(os.path.join(directory, "default.mgpic"), filename)

        # define replacements
        replacements = [("DIRECTORY", directory),
                        ("DIRECTORY", directory),
                        ("DIRECTORY", directory),
                        ("DIRECTORY", directory),
                        ("COORDXYZ", coord),
                        ("RMS_2FO_LEVEL", rms_2fo),
                        ("RMS_DMFP_LEVEL", rms_dmfp),
                        ("RMS_DMFN_LEVEL", rms_dmfn),
                        ("LEGEND_INSERT", legend),
                        ("PDBID", pdbID),
                        ("PDBID", pdbID),
                        ("PDBID", pdbID),
                        ("PDBID", pdbID),
                        ("ORIENTATION", orientation)
                                ]
        # do the replacing
        for key, value in replacements:
                with fileinput.FileInput(filename, inplace=True) as file:
                        for line in file:
                                print(line.replace(key, value), end='')
                #print(f'{key} replaced with {value} in {filename}')

# define function to create the jpg image (broken on modern ubuntu bc of package updates)
# pretty sure ubuntu 16 would work directly
def run_ccp4mg(pic_file, img_file):
        subprocess.run([ "ccp4mg", "-norestore",
                        "-picture", pic_file,
                        "-R", img_file, "-RO",
                        '{"size":"1600x1600","transparent":"0","smoothribbons":"1","raytrace":"0"}',
                        "-quit" ], stdout=open("plot.log", "a"),
                        stderr=subprocess.STDOUT)
        print(f"CCP4MG file {pic_file} created and {img_file} written")


                    
# Loop through Analysis output

i = 0

while i < len(pdbList):
        pdbID = pdbList[i]
        metal_directory=os.path.join(directory, pdbID+"metals")
        os.mkdir(os.path.join(directory, pdbID+'images'))
        image_directory=os.path.join(directory, pdbID+'images')

# set up some empty lists
        atoms = []
        chains = []
        numbers = []
        metal_coords = []
        legends = []

# loop through common metals
        for metal in metals:
            read_file(f'{metal}_Data', metal_directory)

# check uncommon metals
        read_file("Other_metal_data", metal_directory)

        
            


        
