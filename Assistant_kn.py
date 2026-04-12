# Assistant 6/5/2025
# Program prompts user for name of a file. File should contain comma seperated
# PDB identifiers. Program downlowds files from PBD and PBD REDO (two different
# sites), then unzips files. Need to install sh package in order for unzipping
# functionality to work. 
#
#
#
# ***if you have an unzipped version of any of the files and try to download the
# files again, the program will not overwrite the file and stop running. Delete
# any duplicates or ensure they are not in the same directory before running
# Assistant.
# AI used to help write the ifelse statement to check the response code and see
# if the download was successful. 

import requests
from sh import gunzip
import os

# debug = 1 lets you enter an absolute file path, anything else will
# let user input file name in command line. 
debug = 1

BASE_URL="https://pdb-redo.eu/db/"      # PDB-REDO download directory


# home directory containing python script. files will also download and unzip
# to this directory
directory = "/home/knamikas/BioXFELproject/alchemyTesting_python/" 

#these two functions take a pdb ID 4 letter input and download a file for
#that ID. prints downloaded if the file exists and returns the error otherwise

# download function; tries to get mtz file; if that exists also downloads
# coordinates
def download_data(pdb):
        url = f'{BASE_URL}{pdb}/{pdb}_0cyc.mtz.gz"
        filename = pdb+"_0cyc.mtz.gz"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
                with open(directory+filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                print(f"MTZ file downloaded")
                url = f'{BASE_URL}{pdb}/{pdb}_0cyc.pdb.gz"
                filename = pdb+"_0cyc.pdb.gz"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                with open(directory+filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                print(f"PDB file downloaded")
                else:
                        print(f"error downloading pdb file: {response.status_code}")

        else:
                print(f"error downloading mtz file: {response.status_code}")
        


# function unzips the files for the input PDB id, if they exist. added this
# because one of my test files didn't have mtz data on the PDB REDO database

def unzip(pdbID):
        # unzip PDB file, if it exists otherwise print an error
        if os.path.exists(FILE_PATH+pdbID+"_0cyc.pdb.gz")==True:
                gunzip(FILE_PATH+pdbID+"_0cyc.pdb.gz")
        else:
                print("No pdb file exists for "+pdbID)
        #unzip MTZ file if it exists, otherwise print an error
        if os.path.exists(FILE_PATH+pdbID+"_0cyc.mtz.gz")==True:
                gunzip(FILE_PATH+pdbID+"_0cyc.mtz.gz")
                print("unzipped both files for "+pdbID)
        else:
                print("No mtz file exists for "+pdbID)
        

if debug = 1:
        fileIn = "alchemyTest.txt"
else:
        fileIn = input ("Enter CSV file for PDB identifiers: ")



with open(fileIn, "r") as file:
	pdbString = file.read()

# make a list of strings, each string is one pdb ID and clear each entry of any
# extra spaces

pdbList = [x.strip() for x in pdbString.split(",")]
pdbList = [s for s in pdbList if s]
print(pdbList)

#download and unzip files for each pdb ID:

i = 0

while i<len(pdbList):
        
        print("Downloading files for "+pdbList[i])
        download_data(pdbList[i])
        unzip(pdbList[i])
        i += 1

